import sys
import pathlib
import argparse
import random
import torch
import fnmatch
import json
from typing import Any
import os
from dataclasses import dataclass
import copy

from transformers import PreTrainedModel, PreTrainedTokenizer
from lm_eval.utils import handle_non_serializable
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.activation_extractor import ActivationManipulator
from src.utils.logging import create_logger, setup_logging, loglevel_names
from src.utils import env
from src.functional import project
from src.utils.torch import clear_memory
from scripts.utils.load_model import SUPPORTED_MODELS, load_model


SUPPORTED_TASKS_CHAT = [
    "metabench",  # Broad knowledge and reasoning (ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande).
    "xquad_en",  # English extractive reading comprehension.
    "xquad_ar",  # Arabic extractive reading comprehension.
    "xquad_ru",  # Russian extractive reading comprehension.
    "xquad_es",  # Spanish extractive reading comprehension.
    "ifeval",  # Instruction-following compliance.
    "wikitext",  # Language modeling perplexity.
    "blimp",  # Syntactic/grammatical competence.
    "anli",  # Adversarial natural language inference.
    "piqa",  # Physical commonsense reasoning.
    "mbpp",  # Python code generation.
    "jsonschema_bench",  # Schema-constrained JSON generation.
    "mastermind_easy",  # Symbolic logical deduction.
    "toxigen",  # Toxicity and bias sensitivity.
    "wmdp",  # Harmful knowledge and safety QA.
]

SUPPORTED_TASKS_BASE = [
    "metabench_arc",
    "metabench_hellaswag",
    "metabench_mmlu",
    "metabench_truthfulqa",
    "metabench_winogrande",
    "wikitext",
    "lambada_cloze",
    "lambada_multilingual_stablelm",
    "blimp",
    "anli",
    "piqa",
    "mbpp",
    "mastermind_easy",
]


TASK_PARAMS: dict[str, dict] = {
    "xquad_en": dict(limit=0.5),
    "xquad_ar": dict(limit=0.5),
    "xquad_ru": dict(limit=0.5),
    "xquad_es": dict(limit=0.5),
    "ifeval": dict(limit=0.25),
    "blimp": dict(limit=0.5),
    "jsonschema_bench": dict(limit=0.33),
}


# metabench: 3 mins
# squad_completion: 27 mins
# xquad: 47 mins
# ifeval: 30 mins
# wikitext: 1.5 mins
# blimp: 9 mins
# anli: 1 mins
# piqa: 1 mins
# mbpp: 5 mins
# toxigen: 30s
# jsonschema_bench: 15 mins
# mastermind_35_easy: 1 mins


logger = create_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "meta_paths",
        type=str,
        nargs="+",
        metavar="PATH",
        help="Paths to evaluation metadata folders.",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=SUPPORTED_TASKS_CHAT + SUPPORTED_TASKS_BASE + ["auto", "chat", "base"],
        default=["all"],
        metavar="TASKS",
        help=(
            "List of benchmark tasks to run. If not specified, will run all supported tasks. "
            " - If 'auto' is specified, runs all appropriate tasks for the model. "
            " - If 'chat' is specified, runs all chat model tasks. "
            " - If 'base' is specified, runs all base model tasks."
            f"Chat models tasks: {SUPPORTED_TASKS_CHAT}. "
            f"Base models tasks: {SUPPORTED_TASKS_BASE}. "
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        metavar="N",
        help="Batch size for data loading.",
    )

    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Whether the run is a test run (only runs on a small subset of the data for quick testing).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        choices=loglevel_names(),
        default="INFO",
        metavar="LEVEL",
        help=f"Logging level to python-logger. Available levels: {loglevel_names()}",
    )

    parser.add_argument(
        "--recurse",
        action="store_true",
        help="Recursively search directories for data files.",
    )

    parser.add_argument(
        "--allow_code",
        action="store_true",
        help="Whether to allow code execution during evaluation (for tasks that require it, e.g. MBPP).",
    )

    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=["*"],
        metavar="PATTERN",
        help="List of filename patterns to include when searching directories.",
    )

    args = parser.parse_args()

    # print the parsed arguments
    print()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    return args


@dataclass
class Meta:
    model_name: str
    layer_name: str
    direction: torch.Tensor
    is_chat_model: bool


def _read_meta(path: pathlib.Path) -> Meta | None:
    if path.suffix != ".json":
        return None

    metadata = json.load(path.open())

    # make sure the meta has the required fields
    if not all(field in metadata for field in ["model_name", "layer_name", "is_chat_model"]):
        return None

    # check if direction.pt in the same folder as the meta json
    direction_path = path.parent / "direction.pt"
    if not direction_path.exists():
        logger.warning(f"Direction file not found for meta {path}. Skipping...")
        return None

    direction = torch.load(direction_path, weights_only=True, map_location="cuda:0")

    return Meta(
        model_name=metadata["model_name"],
        layer_name=metadata["layer_name"],
        is_chat_model=metadata["is_chat_model"],
        direction=direction,
    )


def read_data(paths: list[str], recurse: bool, patterns: list[str]) -> tuple[list[str], list[Meta]]:
    path_list = []
    meta_list = []

    def matches_patterns(file_path: pathlib.Path) -> bool:
        """Check if file path matches any of the patterns."""
        # Use forward-slash normalized path for consistent pattern matching
        path_str = file_path.as_posix()
        return any(fnmatch.fnmatch(path_str, pattern) for pattern in patterns)

    user_dir = pathlib.Path.cwd()
    for path in paths:
        # make all paths relative to CWD
        path = pathlib.Path(path).resolve()
        path = path.relative_to(user_dir, walk_up=True)

        if path.is_dir():
            inner_paths = [p.as_posix() for p in path.iterdir() if (p.is_file() and matches_patterns(p)) or (p.is_dir() and recurse)]
            sub_paths, sub_data = read_data(inner_paths, recurse, patterns)
            if len(sub_data) == 0:
                continue

            logger.info(f"Loaded {len(sub_data)} files from directory: {path}")
            path_list.extend(sub_paths)
            meta_list.extend(sub_data)
            continue

        try:
            meta = _read_meta(path)
            if meta is None:
                continue

        except Exception as e:
            logger.error(f"Error reading file {path}: {e}. Skipping...")
            continue

        path_list.append(path.as_posix())
        meta_list.append(meta)

    return path_list, meta_list


def prepare_environment(seed: int | None, args: argparse.Namespace):
    if seed is None:
        seed = random.randint(0, 10000)
    logger.info(f"Random seed: {seed}")

    if args.allow_code:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        logger.warning("Code execution is enabled for benchmarks.")

    torch.set_float32_matmul_precision("high")
    env.prepare_environment()
    env.set_seed(seed)


@torch.inference_mode()
def run_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    meta: Meta,
    task: str,
    args: argparse.Namespace,
) -> dict[str, Any]:

    task_params = copy.deepcopy(TASK_PARAMS.get(task, {}))

    if args.test_run:
        task_params["limit"] = 10

    def subtract_projection(activations: torch.Tensor) -> torch.Tensor:
        direction = meta.direction.to(activations.device, activations.dtype)
        projection = project(activations, direction, normalize=True)
        return activations - projection

    manipulator = ActivationManipulator(model, meta.layer_name, manipulation_fn=subtract_projection)

    with manipulator.capture():
        bench_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            device="cuda:0",
            enable_thinking=False,
            max_length=2048,
        )

        bench_results = evaluator.simple_evaluate(
            model=bench_model,
            tasks=[task],
            limit=task_params.pop("limit", None),
            log_samples=task_params.pop("log_samples", False),
            gen_kwargs=task_params.pop("gen_kwargs", None),
            bootstrap_iters=task_params.pop("bootstrap_iters", 0),
            confirm_run_unsafe_code=args.allow_code,
            **task_params,
        )

    if bench_results is None:
        bench_results = {}

    return bench_results


def _get_tasks(meta: Meta, args: argparse.Namespace) -> list[str]:

    tasks = args.tasks

    if ("auto" in args.tasks and meta.is_chat_model) or "chat" in args.tasks:
        tasks += SUPPORTED_TASKS_CHAT
    if ("auto" in args.tasks and not meta.is_chat_model) or "base" in args.tasks:
        tasks += SUPPORTED_TASKS_BASE

    tasks = list(set(tasks))
    tasks = [task for task in tasks if task not in {"auto", "chat", "base"}]
    return tasks


def save_results(results: dict[str, Any], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(results, f, indent=4, default=handle_non_serializable)
    logger.info(f"Saved benchmark results to: {path}")


def main(args: argparse.Namespace):

    # read data
    path_list, data_list = read_data(args.meta_paths, args.recurse, args.patterns)
    logger.info(f"Total files loaded: {len(data_list)}")

    # group paths and metas by model name
    model_names = sorted(set(meta.model_name for meta in data_list))
    model_to_paths: dict[str, list[pathlib.Path]] = {model_name: [] for model_name in model_names}
    model_to_metas: dict[str, list[Meta]] = {model_name: [] for model_name in model_names}

    if any(model_name not in SUPPORTED_MODELS for model_name in model_names):
        unsupported = [model_name for model_name in model_names if model_name not in SUPPORTED_MODELS]
        logger.error(f"Unsupported models found in data: {unsupported}. Supported models: {SUPPORTED_MODELS}")
        return

    for p, m in zip(path_list, data_list):
        model_to_paths[m.model_name].append(p)
        model_to_metas[m.model_name].append(m)

    total_runs = sum(len(metas) for metas in model_to_metas.values())
    current_run = 1

    for model_name in model_names:
        logger.info(f"Processing model: {model_name}")
        paths = model_to_paths[model_name]
        metas = model_to_metas[model_name]
        logger.info(f"Found {len(metas)} meta files for model {model_name}.")

        logger.info(f"Loading model: {model_name}")
        model, tokenizer = load_model(model_name, dtype=torch.bfloat16, device_map="cuda:0")

        for meta, path in zip(metas, paths):
            model_tasks = _get_tasks(meta, args)
            logger.info(f"Tasks to run for meta {path}: {model_tasks}")

            for j, task_name in enumerate(model_tasks):
                print()
                logger.info(f"Processing (model, layer) = ({meta.model_name}, {meta.layer_name}) ({current_run}/{total_runs})")
                logger.info(f"Running benchmark for task: {task_name} ({j + 1}/{len(model_tasks)})")

                try:
                    task_results = run_benchmark(
                        model=model,
                        tokenizer=tokenizer,
                        meta=meta,
                        task=task_name,
                        args=args,
                    )

                except Exception as e:
                    logger.error(f"Error running benchmark for task {task_name} with meta {path}: {e}. Skipping...")
                    continue

                if not args.test_run:
                    result_path = pathlib.Path(path).parent.parent / "benchmarks" / f"{task_name}.json"
                    save_results(task_results, path=result_path)

                else:
                    logger.info(f"Test run - {task_name} results:")
                    print(json.dumps(task_results["results"], indent=4, default=str))

            # prepare for next run
            clear_memory()
            current_run += 1


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level)
    prepare_environment(seed=args.seed, args=args)
    main(args)
