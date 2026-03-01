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

from src.norm.train import train_norm
from src.norm.utils import redundancy_score_norm
from scripts.experiment import Experiment


class NormExperiment(Experiment):
    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.01,
            metavar="LR",
            help="Learning rate for the optimization of the direction.",
        )

        parser.add_argument(
            "--proj_weight",
            type=float,
            default=0.1,
            metavar="FLOAT",
            help="Weight for the projection term in the loss function.",
        )

        parser.add_argument(
            "--kl_weight",
            type=float,
            default=1.0,
            metavar="FLOAT",
            help="Weight for the KL divergence term in the loss function.",
        )

        parser.set_defaults(
            project_name="silent-norm",
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

        metric_tracker.report_hparams(
            "main_params",
            model_name=targeted_model.model.name_or_path,
            layer_name=layer_name,
            learning_rate=args.learning_rate,
            proj_weight=args.proj_weight,
        )

        direction, history = train_norm(
            targeted_model=targeted_model,
            layer=layer_name,
            dl_train=dl_train,
            stop_criteria=stop_criteria,
            learning_rate=args.learning_rate,
            proj_weight=args.proj_weight,
            kl_weight=args.kl_weight,
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

        metrics["score"] = redundancy_score_norm(
            proj_norm=metrics["proj_l2_rel"],
            top1_acc=metrics["top1_acc"],
            top10_agr=metrics["top10_agr"],
        )

        outliers = self.collect_outliers(sample_data)

        return metrics, outliers

    def collect_outliers(self, sample_data: pd.DataFrame, sigma: float = 3.0) -> pd.DataFrame:
        mean = sample_data["kl_div"].mean()
        std = sample_data["kl_div"].std()
        outlier_samples = sample_data[sample_data["kl_div"] > mean + sigma * std]
        return outlier_samples


if __name__ == "__main__":
    experiment = NormExperiment()
    experiment.main()
