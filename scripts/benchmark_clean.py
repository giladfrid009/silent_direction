import sys
import pathlib
import argparse
import random
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import json
from typing import Any

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.utils.logging import create_logger, setup_logging, loglevel_names
from src.utils import env
from scripts.utils.load_model import SUPPORTED_MODELS, load_model
from scripts.benchmark import SUPPORTED_TASKS


logger = create_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=SUPPORTED_MODELS,
        metavar="MODELS",
        help=f"List of models to evaluate. Supported models: {SUPPORTED_MODELS}",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=SUPPORTED_TASKS,
        default=SUPPORTED_TASKS,
        metavar="TASKS",
        help="List of benchmark tasks to run.",
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
        "--results_dir",
        type=str,
        metavar="DIR",
        default="clean_results",
        help="Directory to save benchmark results.",
    )

    args = parser.parse_args()

    # print the parsed arguments
    print()
    print("Parsed arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    return args


def prepare_environment(seed: int | None):
    if seed is None:
        seed = random.randint(0, 10000)
    logger.info(f"Random seed: {seed}")

    torch.set_float32_matmul_precision("high")
    env.prepare_environment()
    env.set_seed(seed)


def run_benchmark(
    model,
    tokenizer,
    task: str,
    args: argparse.Namespace,
) -> dict[str, Any]:

    bench_model = HFLM(
        model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
        device="cuda:0",
        enable_thinking=None,
    )

    bench_results = evaluator.simple_evaluate(
        model=bench_model,
        tasks=[task],
        limit=10 if args.test_run else None,
        log_samples=False,
        gen_kwargs=None,
        bootstrap_iters=0,
    )

    if bench_results is None:
        bench_results = {}

    return bench_results


def save_results(results: dict[str, Any], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(results, f, indent=4, default=str)
    logger.info(f"Saved benchmark results to: {path}")


def main(args: argparse.Namespace):

    for model_name in args.models:
        logger.info(f"Processing model: {model_name}")

        logger.info(f"Loading model: {model_name}")
        model, tokenizer = load_model(model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")

        for j, task_name in enumerate(args.tasks):
            logger.info(f"Running benchmark for task: {task_name} ({j + 1}/{len(args.tasks)})")

            try:
                task_results = run_benchmark(
                    model=model,
                    tokenizer=tokenizer,
                    task=task_name,
                    args=args,
                )

            except Exception as e:
                logger.error(f"Error running benchmark for task {task_name}: {e}.")
                continue

            if not args.test_run:
                task_results.pop("configs", None)
                result_path = pathlib.Path(args.output_dir) / model_name.split("/")[-1] / "benchmarks" / f"{task_name}.json"
                save_results(task_results, path=result_path)
                
            else:
                print(f"Test run - {task_name} results:")
                print(json.dumps(task_results["results"], indent=4, default=str))


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level)
    prepare_environment(seed=args.seed)
    main(args)
