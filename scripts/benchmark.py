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
from functools import partial
from contextlib import nullcontext

from transformers import PreTrainedModel, PreTrainedTokenizer
from lm_eval.utils import handle_non_serializable
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

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


TASK_MANAGER: TaskManager | None = None  # global task manager instance to be used for all benchmarks

SUPPORTED_TASKS_CHAT = [
    "jsonschema_bench",  # Schema-constrained JSON generation.
    "metabench_arc",  # Multiple choice question answering.
    "metabench_gsm8k",  # Grade school math problem solving.
    "metabench_hellaswag",  # Commonsense reasoning about grounded situations.
    "metabench_mmlu",  # Multiple choice question answering across 57 subjects.
    "metabench_truthfulqa",  #  question answering benchmark focused on measuring truthfulness and factual accuracy.
    "metabench_winogrande",  # commonsense reasoning benchmark with pronoun resolution questions.
    "wmdp",  # Harmful knowledge and safety QA.
    "mbpp",  # Python code generation.
    "ifeval",  # Instruction-following compliance.
    "xquad_en",  # English extractive reading comprehension.
    "xquad_ar",  # Arabic extractive reading comprehension.
    "xquad_ru",  # Russian extractive reading comprehension.
    "xquad_es",  # Spanish extractive reading comprehension.
    "xquad_zh",  # Chinese extractive reading comprehension.
    "anli",  # Adversarial natural language inference.
    "piqa",  # Physical commonsense reasoning.
    "mastermind_easy",  # Symbolic logical deduction.
    "toxigen",  # Toxicity and bias sensitivity.
    "blimp",  # Syntactic/grammatical competence.
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
    "metabench_arc": dict(batch_scale=1.0),
    "metabench_gsm8k": dict(batch_scale=6.0),
    "metabench_hellaswag": dict(batch_scale=1.0),
    "metabench_mmlu": dict(batch_scale=1.0),
    "metabench_truthfulqa": dict(batch_scale=1.0),
    "metabench_winogrande": dict(batch_scale=3.0),
    "metabench": dict(batch_scale=1.0),
    "xquad_en": dict(batch_scale=2.0, limit=0.5),
    "xquad_ar": dict(batch_scale=1.35, limit=0.5),
    "xquad_ru": dict(batch_scale=1.5, limit=0.5),
    "xquad_es": dict(batch_scale=2.0, limit=0.5),
    "xquad_zh": dict(batch_scale=1.5, limit=0.5),
    "ifeval": dict(batch_scale=1.5, limit=0.25),
    "wikitext": dict(batch_scale=1.0),
    "blimp": dict(batch_scale=15.0, limit=0.5),
    "anli": dict(batch_scale=6.0),
    "piqa": dict(batch_scale=6.0),
    "mbpp": dict(batch_scale=1.35),
    "jsonschema_bench": dict(batch_scale=1.35, limit=0.33, gen_kwargs=dict(max_gen_toks=256, until=["\n\n"])),
    "mastermind_easy": dict(batch_scale=6.0),
    "toxigen": dict(batch_scale=6.0),
    "wmdp": dict(batch_scale=1.0),
}


logger = create_logger(__name__)


@dataclass
class Meta:
    model_name: str
    layer_name: str
    direction: torch.Tensor | None
    is_chat_model: bool
    path: str


class Benchmarker:
    def __init__(self):
        self._parsed_args = None

    def args(self) -> argparse.Namespace:
        if self._parsed_args is None:
            raise ValueError("Arguments have not been parsed yet. Call _parse_args() first.")
        return self._parsed_args

    def parse_args(self):
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
            default=["auto"],
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
            default=16,
            metavar="N",
            help="Batch size for data loading.",
        )

        parser.add_argument(
            "--log_samples",
            action="store_true",
            help="Whether to log samples and predictions during evaluation",
        )

        parser.add_argument(
            "--test_run",
            action="store_true",
            help="Whether the run is a test run (only runs on a small subset of the data for quick testing).",
        )

        parser.add_argument(
            "--clean_run",
            action="store_true",
            help="Whether benchmark a clean model (without any projection applied).",
        )

        parser.add_argument(
            "--do_sample",
            choices=["true", "false"],
            default="true",
            help="Whether to use sampling instead of greedy decoding during evaluation (for tasks that require generation).",
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
            "--disable_code",
            action="store_true",
            help="Whether to disable code execution during evaluation (for tasks that require it, e.g. MBPP).",
        )

        parser.add_argument(
            "--patterns",
            type=str,
            nargs="+",
            default=["*"],
            metavar="PATTERN",
            help="List of filename patterns to include when searching directories.",
        )

        self._parsed_args = parser.parse_args()

        # print the parsed arguments
        print()
        print("Parsed arguments:")
        for arg, value in vars(self.args()).items():
            print(f"  {arg}: {value}")
        print()

    def _validate_args(self):
        args = self.args()
        for path in args.meta_paths:
            path = pathlib.Path(path).resolve()

            if not path.exists():
                raise FileNotFoundError(f"Meta path does not exist: {path}")

            if path.is_dir() and not args.recurse:
                raise IsADirectoryError(f"Meta path is a directory but --recurse is not set: {path}")

    def _subtract_projection(self, activations: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        projection = project(activations, direction, normalize=True)
        return activations - projection

    def _read_meta(self, path: pathlib.Path) -> Meta | None:
        if path.suffix != ".json":
            return None

        metadata = json.load(path.open())

        # make sure the meta has the required fields
        if not all(field in metadata for field in ["model_name", "layer_name", "is_chat_model"]):
            return None

        direction = None

        if not self.args().clean_run:
            # check if direction.pt in the same folder as the meta json
            direction_path = path.parent / "direction.pt"
            if not direction_path.exists():
                logger.warning(f"Direction file not found for meta {path}. Skipping...")
                return None

            direction = torch.load(direction_path, weights_only=True)

        return Meta(
            model_name=metadata["model_name"],
            layer_name=metadata["layer_name"],
            is_chat_model=metadata["is_chat_model"],
            direction=direction,
            path=path.as_posix(),
        )

    def read_data(self, paths: list[str], recurse: bool, patterns: list[str]) -> list[Meta]:
        meta_list: list[Meta] = []

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
                sub_data = self.read_data(inner_paths, recurse, patterns)
                if len(sub_data) == 0:
                    continue

                logger.info(f"Loaded {len(sub_data)} files from directory: {path}")
                meta_list.extend(sub_data)
                continue

            try:
                meta = self._read_meta(path)

                if meta is None:
                    continue

            except Exception as e:
                logger.error(f"Error reading file {path}: {e}. Skipping...")
                continue

            meta_list.append(meta)

        return meta_list

    def prepare_environment(self):
        seed = self.args().seed
        if seed is None:
            seed = random.randint(0, 10000)
        logger.info(f"Random seed: {seed}")

        if not self.args().disable_code:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            logger.warning("Code execution is enabled for benchmarks.")

        torch.set_float32_matmul_precision("high")
        env.prepare_environment()
        env.set_seed(seed)

        logger.info("Initializing a global TaskManager...")
        global TASK_MANAGER
        TASK_MANAGER = TaskManager()

    @torch.inference_mode()
    def _run_benchmark(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        meta: Meta,
        task: str,
        batch_size: int,
    ) -> dict[str, Any]:

        clear_memory()
        args = self.args()
        task_params = copy.deepcopy(TASK_PARAMS.get(task, {}))
        batch_size = int(batch_size * task_params.pop("batch_scale", 1.0))

        if args.test_run:
            task_params["limit"] = batch_size * 2

        if args.do_sample.lower() == "true":  # enable sampling for generation tasks
            gen_kwargs: dict = task_params.get("gen_kwargs", {})
            task_params["gen_kwargs"] = {
                **gen_kwargs,
                "do_sample": True,
                "temperature": model.generation_config.temperature if model.generation_config.temperature is not None else 1.0,
            }

        if meta.direction is not None:
            meta.direction = meta.direction.to(model.device, model.dtype)

            manipulator = ActivationManipulator(
                model,
                meta.layer_name,
                manipulation_fn=partial(self._subtract_projection, direction=meta.direction),
            )

            model_ctx = manipulator.capture()

        else:
            model_ctx = nullcontext()  # no-op context manager

        with model_ctx:
            # create model wrapper for evaluation
            bench_model = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                device="cuda:0",
                enable_thinking=False,
                max_length=2048,
            )

            # run the benchmark and get results
            bench_results = evaluator.simple_evaluate(
                model=bench_model,
                tasks=[task],
                limit=task_params.pop("limit", None),
                log_samples=args.log_samples,
                bootstrap_iters=task_params.pop("bootstrap_iters", 0),
                confirm_run_unsafe_code=not args.disable_code,
                apply_chat_template=meta.is_chat_model,
                fewshot_as_multiturn=meta.is_chat_model,
                task_manager=TASK_MANAGER,
                use_cache=None,
                cache_requests=False,  # otherwise causes weird issues since kwargs are baked-in within the requests
                # set seeds
                random_seed=args.seed,
                numpy_random_seed=args.seed + 1,
                torch_random_seed=args.seed + 1,
                fewshot_random_seed=args.seed + 1,
                **task_params,
            )

        if bench_results is None:
            bench_results = {}

        return bench_results

    def get_tasks(self, meta: Meta) -> list[str]:

        args = self.args()
        tasks = copy.deepcopy(args.tasks)

        if ("auto" in args.tasks and meta.is_chat_model) or "chat" in args.tasks:
            tasks += SUPPORTED_TASKS_CHAT
        if ("auto" in args.tasks and not meta.is_chat_model) or "base" in args.tasks:
            tasks += SUPPORTED_TASKS_BASE

        tasks = list(dict.fromkeys(tasks).keys())  # remove duplicates while preserving order
        tasks = [task for task in tasks if task not in {"auto", "chat", "base"}]
        return tasks

    def save_results(self, results: dict[str, Any], path: pathlib.Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(results, f, indent=4, default=handle_non_serializable)
        logger.info(f"Saved benchmark results to: {path}")

    def run_benchmark(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        meta: Meta,
        task_name: str,
        batch_size: int,
    ):
        """
        Returns the updated batch size after running the benchmark (in case it had to be reduced due to OOM).
        """

        while True:
            try:
                task_results = self._run_benchmark(
                    model=model,
                    tokenizer=tokenizer,
                    meta=meta,
                    task=task_name,
                    batch_size=batch_size,
                )

            except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError) as e:
                if batch_size == 1:
                    logger.error(f"OOM error running benchmark for task {task_name} even with batch size 1. Skipping...")
                    return batch_size

                batch_size = max(1, int(batch_size / 2))
                logger.warning((f"OOM error running benchmark for task {task_name}. Decreasing batch size to {batch_size}"))
                continue

            except Exception:
                logger.exception("Error running benchmark for task %s with meta %s. Skipping...", task_name, meta.path, exc_info=True)
                return batch_size

            break  # success

        if not self.args().test_run:
            result_path = pathlib.Path(meta.path).parent.parent / "benchmarks" / f"{task_name}.json"
            self.save_results(task_results, path=result_path)

        else:
            logger.info(f"Test run - {task_name} results:")
            print(json.dumps(task_results["results"], indent=4, default=str))

        return batch_size

    def run(self):

        # read data
        args = self.args()
        data_list = self.read_data(args.meta_paths, args.recurse, args.patterns)
        logger.info(f"Total files loaded: {len(data_list)}")

        # group paths and metas by model name
        model_names = sorted(set(meta.model_name for meta in data_list))

        if any(model_name not in SUPPORTED_MODELS for model_name in model_names):
            unsupported = [model_name for model_name in model_names if model_name not in SUPPORTED_MODELS]
            logger.error(f"Unsupported models found in data: {unsupported}. Supported models: {SUPPORTED_MODELS}")
            return

        model_to_metas: dict[str, list[Meta]] = {model_name: [] for model_name in model_names}
        for m in data_list:
            model_to_metas[m.model_name].append(m)

            if args.clean_run and len(model_to_metas[m.model_name]) > 1:
                logger.warning(f"Multiple metas found for model {m.model_name} in clean run.")

        total_runs = sum(len(metas) for metas in model_to_metas.values())
        current_run = 1

        for model_name in model_names:
            logger.info(f"Processing model: {model_name}")

            # each model begins with the default batch size from the config, and may be reduced if OOM is encountered
            batch_size: int = args.batch_size
            metas = model_to_metas[model_name]
            logger.info(f"Found {len(metas)} meta files for model {model_name}.")
            logger.info(f"Loading model: {model_name}")

            model, tokenizer = load_model(model_name, device_map="cuda:0")

            for meta in metas:
                model_tasks = self.get_tasks(meta)

                print()
                logger.info(f"Tasks to run for meta {meta.path}: {model_tasks}")
                logger.info(f"Processing (model, layer) = ({meta.model_name}, {meta.layer_name}) ({current_run / total_runs:.2%})")

                for j, task_name in enumerate(model_tasks):
                    logger.info(f"Running benchmark for task: {task_name} ({j + 1}/{len(model_tasks)})")
                    logger.info(f"Model batch size: {batch_size}")

                    batch_size = self.run_benchmark(
                        model=model,
                        tokenizer=tokenizer,
                        meta=meta,
                        task_name=task_name,
                        batch_size=batch_size,
                    )

                # prepare for next run
                current_run += 1

    def main(self):

        import evaluate.config as eval_config
        import tempfile

        # NOTE: sorcery to support multi-process evaluation.
        # some benchmarks in lm_eval use HF evaluate library. when launching benchmark
        # from multiple processes and both processes run the same benchmark, sometimes
        # they will try to access the same cache files in HF_METRICS_CACHE which causes crashes.
        with tempfile.TemporaryDirectory(dir=eval_config.HF_METRICS_CACHE) as tmp_dir:
            eval_config.HF_METRICS_CACHE = tmp_dir
            os.environ["HF_METRICS_CACHE"] = tmp_dir

            self.parse_args()
            setup_logging(level=self.args().log_level)
            self.prepare_environment()
            self.run()


if __name__ == "__main__":
    runner = Benchmarker()
    runner.main()
