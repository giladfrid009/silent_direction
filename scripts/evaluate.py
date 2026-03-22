from sqlite3 import NotSupportedError
import sys
import pathlib
import argparse
import random
import torch
import fnmatch
import json
from typing import Any
from dataclasses import dataclass
import copy

import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))


from src.config import StopCriteria
from src.utils.logging import create_logger, setup_logging, loglevel_names
from src.utils import env
from src.functional import project
from src.utils.torch import clear_memory
from src.data import TableLoader
from src.evaluate import evaluate
from src.model import TargetedModel
from scripts.utils.load_model import SUPPORTED_MODELS, load_model
from scripts.utils.load_dataset import SUPPORTED_DATASETS, load_dataset, is_chat_dataset


SUPPORTED_DATASETS_CHAT = SUPPORTED_DATASETS
SUPPORTED_DATASETS_BASE = []


logger = create_logger(__name__)


@dataclass
class Meta:
    model_name: str
    layer_name: str
    direction: torch.Tensor
    is_chat_model: bool
    path: str


class Evaluator:
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
            "--datasets",
            type=str,
            nargs="+",
            default=["auto"],
            choices=SUPPORTED_DATASETS_CHAT + SUPPORTED_DATASETS_BASE + ["auto", "chat", "base"],
            help=(
                "List of datasets to run. If not specified, will run all supported datasets. "
                " - If 'auto' is specified, runs all appropriate datasets for the model. "
                " - If 'chat' is specified, runs all chat model datasets. "
                " - If 'base' is specified, runs all base model datasets. "
                f"Chat models datasets: {SUPPORTED_DATASETS_CHAT}. "
                f"Base models datasets: {SUPPORTED_DATASETS_BASE}. "
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
            "--max_time",
            type=int,
            default=15,
            metavar="MINUTES",
            help="The maximum evaluation time in minutes.",
        )

        parser.add_argument(
            "--max_steps",
            type=int,
            default=250,
            metavar="NUM",
            help="The maximum number of evaluation steps.",
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

        torch.set_float32_matmul_precision("high")
        env.prepare_environment()
        env.set_seed(seed)

    @torch.inference_mode()
    def _run_evaluation(
        self,
        targeted_model: TargetedModel,
        meta: Meta,
        ds_eval: pd.DataFrame,
        batch_size: int,
        max_steps: int = 250,
    ) -> tuple[dict[str, Any], pd.DataFrame]:

        clear_memory()
        args = self.args()

        dl_eval = TableLoader(
            df=ds_eval,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        meta.direction = meta.direction.to(
            targeted_model.device,
            targeted_model.dtype,
        )

        # increase max steps by the same factor we decreased batch size
        batch_ratio = args.batch_size / batch_size
        max_steps = int(args.max_steps * batch_ratio)

        stop_criteria = StopCriteria(
            max_steps=max_steps if not args.test_run else 10,
            max_time=args.max_time if not args.test_run else 2,
        )

        metrics, sample_data = evaluate(
            targeted_model=targeted_model,
            layer=meta.layer_name,
            dl_eval=dl_eval,
            direction=meta.direction,
            stop_criteria=stop_criteria,
        )

        return metrics, sample_data

    def get_datasets(self, meta: Meta) -> list[str]:

        args = self.args()
        datasets = copy.deepcopy(args.datasets)

        if ("auto" in args.datasets and meta.is_chat_model) or "chat" in args.datasets:
            datasets += SUPPORTED_DATASETS_CHAT
        if ("auto" in args.datasets and not meta.is_chat_model) or "base" in args.datasets:
            datasets += SUPPORTED_DATASETS_BASE
            raise NotSupportedError("Base (non-chat) datasets are not supported yet.")

        datasets = list(dict.fromkeys(datasets).keys())  # remove duplicates while preserving order
        datasets = [ds_name for ds_name in datasets if ds_name not in {"auto", "chat", "base"}]
        return datasets

    def save_benchmarks(
        self,
        bench_name: str,
        base_dir: pathlib.Path,
        results: dict[str, Any],
    ):

        if base_dir is None:
            logger.warning("No log directory found. Skipping saving benchmarks.")
            return

        bench_dir = pathlib.Path(base_dir) / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)

        with open(bench_dir / f"{bench_name}.json", "w") as f:
            json.dump({"results": {bench_name: results}}, f, indent=4)

        logger.info(f"Saved metrics results to {bench_dir} folder.")

    def run_evaluation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        meta: Meta,
        ds_name: str,
        batch_size: int,
    ):
        """
        Returns the updated batch size after running the evaluation (in case it had to be reduced due to OOM).
        """

        ds_train, ds_eval, ds_test = load_dataset(ds_name)
        targeted_model = TargetedModel(model, tokenizer, is_chat=is_chat_dataset(ds_test))

        while True:
            try:
                results, sample_data = self._run_evaluation(
                    targeted_model=targeted_model,
                    meta=meta,
                    ds_eval=ds_test,
                    batch_size=batch_size,
                )

            except (torch.OutOfMemoryError, torch.cuda.OutOfMemoryError) as e:
                if batch_size == 1:
                    logger.error(f"OOM error running evaluation for dataset {ds_name} even with batch size 1. Skipping...")
                    return batch_size

                batch_size = max(1, int(batch_size / 2))
                logger.warning((f"OOM error running evaluation for dataset {ds_name}. Decreasing batch size to {batch_size}"))
                continue

            except Exception:
                logger.exception("Error running evaluation for dataset %s with meta %s. Skipping...", ds_name, meta.path, exc_info=True)
                return batch_size

            break  # success

        if not self.args().test_run:
            self.save_benchmarks(
                bench_name=f"eval-{ds_name}",
                base_dir=pathlib.Path(meta.path).parent.parent,
                results=results,
            )

        else:
            logger.info(f"Test run - {ds_name} results:")
            print(json.dumps(results, indent=4, default=str))

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
                dataset_names = self.get_datasets(meta)

                print()
                logger.info(f"Tasks to run for meta {meta.path}: {dataset_names}")
                logger.info(f"Processing (model, layer) = ({meta.model_name}, {meta.layer_name}) ({current_run / total_runs:.2%})")

                for j, ds_name in enumerate(dataset_names):
                    logger.info(f"Running evaluation for dataset: {ds_name} ({j + 1}/{len(dataset_names)})")
                    logger.info(f"Model batch size: {batch_size}")

                    batch_size = self.run_evaluation(
                        model=model,
                        tokenizer=tokenizer,
                        meta=meta,
                        ds_name=ds_name,
                        batch_size=batch_size,
                    )

                # prepare for next run
                current_run += 1

    def main(self):
        self.parse_args()
        setup_logging(level=self.args().log_level)
        self.prepare_environment()
        self.run()


if __name__ == "__main__":
    runner = Evaluator()
    runner.main()
