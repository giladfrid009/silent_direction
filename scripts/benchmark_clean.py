import sys
import pathlib
import argparse
import torch
import json
from typing import Any
import copy

from transformers import PreTrainedModel, PreTrainedTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.utils.logging import create_logger, setup_logging, loglevel_names
from src.utils.torch import clear_memory
from scripts.utils.load_model import SUPPORTED_MODELS, load_model

from scripts.benchmark import (
    SUPPORTED_TASKS_CHAT,
    SUPPORTED_TASKS_BASE,
    TASK_PARAMS,
    save_results,
    prepare_environment,
)


logger = create_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--model_name",
        type=str,
        choices=SUPPORTED_MODELS,
    )

    parser.add_argument(
        "--is_chat",
        choices=["true", "false"],
        help="Whether the model is a chat model (i.e. whether to apply chat templates during evaluation).",
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
        "--allow_code",
        action="store_true",
        help="Whether to allow code execution during evaluation (for tasks that require it, e.g. MBPP).",
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Optional path to save the results file (overrides default path).",
    )

    args = parser.parse_args()

    # print the parsed arguments
    print()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    return args


@torch.inference_mode()
def run_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: str,
    args: argparse.Namespace,
) -> dict[str, Any]:

    clear_memory()
    task_params = copy.deepcopy(TASK_PARAMS.get(task, {}))
    batch_size = int(args.batch_size * task_params.pop("batch_scale", 1.0))

    if args.test_run:
        task_params["limit"] = batch_size * 2

    bench_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
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
        apply_chat_template=(args.is_chat.lower() == "true"),
        **task_params,
    )

    if bench_results is None:
        bench_results = {}

    return bench_results


def _get_tasks(args: argparse.Namespace) -> list[str]:

    tasks = args.tasks

    if ("auto" in args.tasks and args.is_chat) or "chat" in args.tasks:
        tasks += SUPPORTED_TASKS_CHAT
    if ("auto" in args.tasks and not args.is_chat) or "base" in args.tasks:
        tasks += SUPPORTED_TASKS_BASE

    tasks = list(dict.fromkeys(tasks).keys())  # remove duplicates while preserving order
    tasks = [task for task in tasks if task not in {"auto", "chat", "base"}]
    return tasks


def main(args: argparse.Namespace):

    model_name = args.model_name
    logger.info(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device_map="cuda:0")

    model_tasks = _get_tasks(args)
    logger.info(f"Tasks to run: {model_tasks}")

    for i, task_name in enumerate(model_tasks):
        print()
        logger.info(f"Running benchmark for task: {task_name} ({i + 1}/{len(model_tasks)})")

        try:
            task_results = run_benchmark(
                model=model,
                tokenizer=tokenizer,
                task=task_name,
                args=args,
            )

        except Exception as e:
            logger.exception("Error running benchmark for task %s. Skipping...", task_name, exc_info=True)
            continue

        if not args.test_run:
            root_dir = pathlib.Path(args.root_dir) if args.root_dir else pathlib.Path(__file__).parent.parent
            result_path = root_dir / "clean_benchmarks" / "benchmarks" / model_name / f"{task_name}.json"
            save_results(task_results, path=result_path)

        else:
            logger.info(f"Test run - {task_name} results:")
            print(json.dumps(task_results["results"], indent=4, default=str))


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level)
    prepare_environment(seed=args.seed, args=args)
    main(args)
