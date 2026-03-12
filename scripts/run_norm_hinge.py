from argparse import ArgumentParser
import sys
import pathlib
import pandas as pd

# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

import torch
from src.model import TargetedModel
from src.data import TableLoader
from src.utils.trackers import MetricTracker
from src.config import StopCriteria
from src.evaluate import evaluate

from src.norm_hinge.train import train_norm_hinge
from scripts.experiment import Experiment


class NormHingeExperiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:

        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.01,
            metavar="LR",
            help="Learning rate for the optimization of the direction.",
        )

        parser.add_argument(
            "--target_kl",
            type=float,
            metavar="KL",
            help="Target KL divergence for the projection of the activations.",
        )

        parser.add_argument(
            "--kl_coef",
            type=float,
            metavar="FLOAT",
            default=1.0,
            help="Coefficient for the KL divergence penalty.",
        )

        parser.add_argument(
            "--tol_factor",
            type=float,
            default=2.0,
            metavar="FLOAT",
            help="Tolerance factor for the KL constraint in the scoring function.",
        )

        parser.add_argument(
            "--loss_kind",
            type=str,
            choices=["mse", "mae"],
            default="mae",
            help="Kind of loss to use for the KL penalty (mse or mae).",
        )

        parser.add_argument(
            "--loss_reduction",
            type=str,
            choices=["none", "mean", "samplemean"],
            default="none",
            help="Reduction method for the loss computation.",
        )

        parser.set_defaults(
            project_name="silent-norm-hinge",
        )

    def run_training(
        self,
        targeted_model: TargetedModel,
        layer_name: str,
        dl_train: TableLoader,
        dl_eval: TableLoader,
        stop_criteria: StopCriteria,
        metric_tracker: MetricTracker,
    ) -> torch.Tensor:

        args = self.args()

        metric_tracker.set_tags(target_kl=args.target_kl)

        metric_tracker.report_hparams(
            "main_params",
            model_name=targeted_model.model.name_or_path,
            layer_name=layer_name,
            target_kl=args.target_kl,
            learning_rate=args.learning_rate,
            kl_coef=args.kl_coef,
            tol_factor=args.tol_factor,
            loss_kind=args.loss_kind,
            loss_reduction=args.loss_reduction,
        )

        direction, history = train_norm_hinge(
            targeted_model=targeted_model,
            layer=layer_name,
            dl_train=dl_train,
            stop_criteria=stop_criteria,
            target_kl=args.target_kl,
            learning_rate=args.learning_rate,
            hinge_coef=args.kl_coef,
            tol_factor=args.tol_factor,
            loss_kind=args.loss_kind,
            loss_reduction=args.loss_reduction,
        )

        for step, metrics in enumerate(history):
            metric_tracker.report_scalars(metrics, step=step)

        return direction

    def run_evaluation(
        self,
        targeted_model: TargetedModel,
        layer_name: str,
        direction: torch.Tensor,
        dl_test: TableLoader,
        stop_criteria: StopCriteria,
    ) -> tuple[dict[str, float], pd.DataFrame]:

        metrics, sample_data = evaluate(
            targeted_model=targeted_model,
            layer=layer_name,
            direction=direction,
            dl_eval=dl_test,
            stop_criteria=stop_criteria,
        )

        outliers = self.collect_outliers(sample_data)

        return metrics, outliers

    def collect_outliers(self, sample_data: pd.DataFrame, sigma: float = 3.0) -> pd.DataFrame:
        mean = sample_data["kl_div"].mean()
        std = sample_data["kl_div"].std()
        outlier_samples = sample_data[sample_data["kl_div"] > mean + sigma * std]
        return outlier_samples


if __name__ == "__main__":
    experiment = NormHingeExperiment()
    experiment.main()
