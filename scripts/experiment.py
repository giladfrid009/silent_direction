from abc import abstractmethod, ABC
import torch
import random
import argparse
import sys
import time
from contextlib import ExitStack
import json
import pathlib

from src.utils import env
from src.utils.logging import create_logger, setup_logging, loglevel_names
from src.data import TableLoader
from src.utils.trackers import MetricTracker
from src.utils.torch import clear_memory
from src.config import StopCriteria
from src.model import TargetedModel

from scripts.utils.load_model import SUPPORTED_MODELS, load_model
from scripts.utils.load_dataset import SUPPORTED_DATASETS, load_dataset, is_chat_dataset


logger = create_logger(__name__)


class Experiment(ABC):
    def __init__(self) -> None:
        self._parsed_args = None

    def args(self) -> argparse.Namespace:
        if self._parsed_args is None:
            raise ValueError("Arguments have not been parsed yet. Call _parse_args() first.")
        return self._parsed_args

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Override to add custom command line arguments."""
        pass

    def _parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "--model",
            type=str,
            choices=SUPPORTED_MODELS,
            metavar="MODEL",
            help=f"The model name to attack. Available models: {SUPPORTED_MODELS}",
        )

        parser.add_argument(
            "--layers",
            type=str,
            nargs="+",
            metavar="LAYERS",
            help="Layers for which the attack is performed. A distinct attack is performed for each specified layer.",
        )

        parser.add_argument(
            "--dataset",
            type=str,
            choices=SUPPORTED_DATASETS,
            default="hh-rlhf",
            metavar="DATASET",
            help=f"The datasets to use. Available datasets: {SUPPORTED_DATASETS}",
        )

        parser.add_argument(
            "--test_datasets",
            type=str,
            choices=SUPPORTED_DATASETS,
            nargs="+",
            default=[],
            metavar="DATASET",
            help=f"Additional datasets for final evaluation. Available datasets: {SUPPORTED_DATASETS}",
        )

        parser.add_argument(
            "--run_name",
            type=str,
            default=time.strftime("%Y-%m-%d_%H-%M-%S"),
            metavar="NAME",
            help="The name of the run, used for logging.",
        )

        parser.add_argument(
            "--train_batch",
            type=int,
            default=8,
            metavar="SIZE",
            help="The training batch size.",
        )

        parser.add_argument(
            "--eval_batch",
            type=int,
            default=8,
            metavar="SIZE",
            help="The evaluation batch size.",
        )

        parser.add_argument(
            "--drop_last",
            choices=["true", "false"],
            metavar="BOOL",
            default="false",
            help="Whether to drop the last incomplete batch from the training data.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=random.randint(0, 1000000),
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
            "--project_name",
            type=str,
            default="silent-direction",
            metavar="NAME",
            help="The name of the project for logging purposes.",
        )

        parser.add_argument(
            "--log_dir",
            type=str,
            default="logs",
            metavar="PATH",
            help="Directory to save logs and results.",
        )

        parser.add_argument(
            "--test_run",
            action="store_true",
            help="If set, the experiment will run a quick test with reduced epochs and time.",
        )

        stop_args = parser.add_argument_group("Stopping Criteria")

        stop_args.add_argument(
            "--train_time",
            type=int,
            default=20,
            metavar="MINUTES",
            help="The maximum training time in minutes.",
        )

        stop_args.add_argument(
            "--train_steps",
            type=int,
            default=200,
            metavar="NUM",
            help="The maximum number of training steps.",
        )

        stop_args.add_argument(
            "--train_patience",
            type=int,
            default=100,
            metavar="NUM",
            help="Early stopping patience in number of evaluations. If 0 then no early stopping is applied.",
        )

        stop_args.add_argument(
            "--eval_time",
            type=int,
            default=5,
            metavar="MINUTES",
            help="The maximum evaluation time in minutes.",
        )

        stop_args.add_argument(
            "--eval_steps",
            type=int,
            default=200,
            metavar="NUM",
            help="The maximum number of evaluation steps.",
        )

        self.add_arguments(parser)
        self._parsed_args = parser.parse_args()
        args = self.args()

        # print the parsed arguments
        print()
        print("Parsed arguments:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print()

        return args

    def prepare_environment(self, seed: int | None):
        if seed is None:
            seed = random.randint(0, 10000)
        logger.info(f"Random seed: {seed}")

        torch.set_float32_matmul_precision("high")
        env.prepare_environment()
        env.set_seed(seed)

    def save_metadata(
        self,
        model_name: str,
        layer_name: str,
        dataset_name: str,
        direction: torch.Tensor,
        metric_tracker: MetricTracker,
        **kwargs,
    ):
        """
        Saves direction and its metadata to disk. Metadata saved as a JSON file.
        """
        log_dir = metric_tracker.log_dir

        if log_dir is None:
            logger.warning("No log directory found. Skipping saving metadata.")
            return

        metadata = {
            "model_name": model_name,
            "layer_name": layer_name,
            "dataset_name": dataset_name,
            **{k: v for k, v in kwargs.items()},
        }

        meta_dir = pathlib.Path(log_dir) / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)

        with open(meta_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        torch.save(direction.cpu().detach(), meta_dir / "direction.pt")

    def save_benchmarks(
        self,
        bench_name: str,
        bench_metrics: dict[str, float],
        metric_tracker: MetricTracker,
    ):
        log_dir = metric_tracker.log_dir

        if log_dir is None:
            logger.warning("No log directory found. Skipping saving benchmarks.")
            return

        bench_dir = pathlib.Path(log_dir) / "benchmarks"
        bench_dir.mkdir(parents=True, exist_ok=True)

        with open(bench_dir / f"{bench_name}.json", "w") as f:
            json.dump({"results": {bench_name: bench_metrics}}, f, indent=4)

        logger.info(f"Saved metrics results to {bench_dir} folder.")

    @torch.inference_mode()
    def final_evaluation(
        self,
        targeted_model: TargetedModel,
        layer_name: str,
        direction: torch.Tensor,
        stop_criteria: StopCriteria,
        metric_tracker: MetricTracker,
    ):
        args = self.args()
        dataset_names = args.test_datasets
        if args.dataset not in dataset_names:
            dataset_names.append(args.dataset)

        logger.info(f"Running final evaluation on datasets: {dataset_names}")

        for ds_name in dataset_names:
            _, _, ds_test = load_dataset(ds_name)
            dl_test = TableLoader(ds_test, batch_size=args.eval_batch, shuffle=False)
            logger.info(f"Loaded test dataset: {ds_name} with {len(ds_test)} samples.")
            logger.info(f"Evaluating on test dataset: {ds_name}")

            clear_memory()
            test_metrics = self.run_evaluation(
                targeted_model=targeted_model,
                layer_name=layer_name,
                direction=direction,
                dl_test=dl_test,
                stop_criteria=stop_criteria,
            )

            metric_tracker.report_globals({f"{ds_name}/{k}": v for k, v in test_metrics.items()})

            if not args.test_run:
                self.save_benchmarks(
                    bench_name=f"eval-{ds_name}",
                    bench_metrics=test_metrics,
                    metric_tracker=metric_tracker,
                )

            else:
                print(f"Test run - {ds_name} results:")
                print(json.dumps(test_metrics, indent=4, default=str))

    @abstractmethod
    def run_training(
        self,
        targeted_model: TargetedModel,
        layer_name: str,
        dl_train: TableLoader,
        dl_eval: TableLoader,
        stop_criteria: StopCriteria,
        metric_tracker: MetricTracker,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def run_evaluation(
        self,
        targeted_model: TargetedModel,
        layer_name: str,
        direction: torch.Tensor,
        dl_test: TableLoader,
        stop_criteria: StopCriteria,
    ) -> dict[str, float]:
        pass

    def run(self):
        args = self.args()

        if not torch.cuda.is_available():
            logger.error("No GPU available. Exiting.")
            sys.exit(1)

        logger.info(f"Loading dataset: {args.dataset}")
        ds_train, ds_val, ds_test = load_dataset(args.dataset)
        dl_train = TableLoader(ds_train, batch_size=args.train_batch, shuffle=True, drop_last=args.drop_last)
        dl_eval = TableLoader(ds_val, batch_size=args.eval_batch, shuffle=False)
        dl_test = TableLoader(ds_test, batch_size=args.eval_batch, shuffle=False)
        is_chat = is_chat_dataset(dl_train)
        logger.info(f"Dataset identified as {'chat' if is_chat else 'non-chat'} format.")
        logger.info(f"Loaded datasets with sample counts: (train, val, test) = ({len(ds_train)}, {len(ds_val)}, {len(ds_test)}).")

        logger.info(f"Loading model: {args.model}")
        model, tokenizer = load_model(args.model, device_map="cuda:0")
        targeted_model = TargetedModel(model=model, tokenizer=tokenizer, is_chat=is_chat)
        logger.info(f"\nModel architecture: {targeted_model.model}\n")

        for layer_name in args.layers:
            logger.info(f"Running experiment for layer: {layer_name}")

            with ExitStack() as stack:
                metric_tracker = MetricTracker.create(
                    args.model.split("/")[-1],
                    args.dataset,
                    layer_name,
                    self.args().run_name,
                    kind="wandb",
                    root_dir=args.log_dir,
                    project=args.project_name,
                    disabled=args.test_run,
                )

                metric_tracker = stack.enter_context(metric_tracker)

                metric_tracker.set_tags(
                    model=args.model,
                    layer=layer_name,
                    dataset=args.dataset,
                )

                if main_file := getattr(sys.modules.get("__main__"), "__file__", None):
                    metric_tracker.upload_code(main_file)
                if expr_file := getattr(sys.modules.get(__name__), "__file__", None):
                    metric_tracker.upload_code(expr_file)

                train_stop = StopCriteria(
                    max_steps=args.train_steps if not args.test_run else 10,
                    max_time=args.train_time if not args.test_run else 2,
                    patience=args.train_patience if args.train_patience > 0 else None,
                )

                direction = self.run_training(
                    targeted_model=targeted_model,
                    layer_name=layer_name,
                    dl_train=dl_train,
                    dl_eval=dl_eval,
                    stop_criteria=train_stop,
                    metric_tracker=metric_tracker,
                )

                # save metadata and direction
                if not args.test_run:
                    self.save_metadata(
                        model_name=args.model,
                        layer_name=layer_name,
                        dataset_name=args.dataset,
                        direction=direction,
                        metric_tracker=metric_tracker,
                        is_chat_model=targeted_model.is_chat,
                    )

                eval_stop = StopCriteria(
                    max_steps=args.eval_steps if not args.test_run else 10,
                    max_time=args.eval_time if not args.test_run else 2,
                )

                # perform full evaluation and save benchmarks
                self.final_evaluation(
                    targeted_model=targeted_model,
                    layer_name=layer_name,
                    direction=direction,
                    stop_criteria=eval_stop,
                    metric_tracker=metric_tracker,
                )

    def main(self):
        try:
            self._parse_args()
            setup_logging(level=self.args().log_level)
            self.prepare_environment(seed=self.args().seed)
            self.run()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            sys.exit(0)
