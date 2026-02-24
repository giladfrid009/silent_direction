import subprocess
import argparse
import sys
import copy
from typing import Dict, List, Any
import torch
import gc

# =============================================================================
# Global Constants & Configuration
# =============================================================================

PROJECT_NAME = "silent-norm-ablations"
SCRIPT_PATH = "scripts/run_norm.py"

# Default Arguments shared across all runs unless overridden
DEFAULT_RUN_ARGS = {
    "log_dir": "logs",
    "dataset": "hh-rlhf",
    "test_datasets": ["hh-rlhf", "slim-orca", "oasst2", "tulu-v2", "lmsys-1m"],
    "train_batch": 8,
    "train_steps": 20000,
    "train_time": 40,
    "train_patience": 100,
    "eval_batch": 8,
    "eval_steps": 250,
}

# =============================================================================
# Experiment Setups
# =============================================================================

# Define reusable experiment configurations
# Using clear names for standard setups.
# These dictionaries are essentially just sets of CLI arguments.

# baseline setting
EXP_SETUP_BASELINE_TULU = dict(
    learning_rate=0.1,
    kl_weight=10.0,
    proj_weigh=1.0,
    dataset="tulu-v2",
)

# lower LR by 5x
EXP_SETUP_SMALL_LR_TULU = dict(
    learning_rate=0.02,
    kl_weight=10.0,
    proj_weigh=1.0,
    dataset="tulu-v2",
)

# lower LR by 2.5x
EXP_SETUP_MEDIUM_LR_TULU = dict(
    learning_rate=0.04,
    kl_weight=10.0,
    proj_weigh=1.0,
    dataset="tulu-v2",
)

# higher LR by 2.5x
EXP_SETUP_LARGE_LR_TULU = dict(
    learning_rate=0.25,
    kl_weight=10.0,
    proj_weigh=1.0,
    dataset="tulu-v2",
)

# higher kl_weight by 2x
EXP_SETUP_HIGH_KL_TULU = dict(
    learning_rate=0.1,
    kl_weight=20.0,
    proj_weigh=1.0,
    dataset="tulu-v2",
)

# lower kl_weight by 2x
EXP_SETUP_SMALL_KL_TULU = dict(
    learning_rate=0.1,
    kl_weight=5.0,
    proj_weigh=1.0,
    dataset="tulu-v2",
)

# TODO: WITHOUT GRAD SCHEDULER AND W.O EARLY STOPPING
EXP_SETUP_NO_EARLY_TULU = dict(
    learning_rate=0.1,
    kl_weight=10.0,
    proj_weigh=1.0,
    dataset="tulu-v2",
    train_patience=0,
)

########


EXP_SETUP_HH_RLHF = dict(
    learning_rate=0.1,
    kl_weight=10.0,
    proj_weigh=1.0,
    dataset="hh-rlhf",
)

EXP_SETUP_SLIM_ORCA = dict(
    learning_rate=0.1,
    kl_weight=10.0,
    proj_weigh=1.0,
    dataset="slim-orca",
)

EXP_SETUP_OASST2 = dict(
    learning_rate=0.1,
    kl_weight=10.0,
    proj_weigh=1.0,
    dataset="oasst2",
)

EXP_SETUP_TULU_V2 = dict(
    learning_rate=0.1,
    kl_weight=10.0,
    proj_weigh=1.0,
    dataset="tulu-v2",
)

EXP_SETUP_LMSYS = dict(
    learning_rate=0.1,
    kl_weight=10.0,
    proj_weigh=1.0,
    dataset="lmsys-1m",
)


# =============================================================================
# Model Registry
# =============================================================================
# Structure:
#   key: Internal identifier
#   value: Dict containing:
#       - "nick": (Required) Used for run_name generation
#       - "experiments": (Required) Dict of {experiment_suffix: specific_args_dict}
#       - ...Any other key represents a CLI argument for the script


def get_search_locations(
    num_layers: int,
    num_probes: int = 2,
    block_path: str | None = "model.layers.{i}",
    attn_path: str | None = "model.layers.{i}.self_attn",
    mlp_path: str | None = "model.layers.{i}.mlp",
    special_locations: list[str] = ["model.embed_tokens", "model.norm"],
) -> list[str]:
    """
    Define architecturally meaningful locations to search for redundant directions.
    """
    num_probes = min(num_probes, num_layers)
    indices = [int(round(i * num_layers / num_probes, 0)) for i in range(num_probes)]
    indices = [min(i, num_layers - 1) for i in indices]  # Ensure indices are within bounds

    locations = []

    if block_path is not None:
        locations += [block_path.format(i=i) for i in indices]

    if attn_path is not None:
        locations += [attn_path.format(i=i) for i in indices]

    if mlp_path is not None:
        locations += [mlp_path.format(i=i) for i in indices]

    special_locations = ["model.embed_tokens", "model.norm"]

    locations += special_locations

    return locations


MODELS = {
    "llama-2-7b-chat": {
        "experiments": {
            "baseline-tulu": EXP_SETUP_BASELINE_TULU,
            "small-lr-tulu": EXP_SETUP_SMALL_LR_TULU,
            "medium-lr-tulu": EXP_SETUP_MEDIUM_LR_TULU,
            "large-lr-tulu": EXP_SETUP_LARGE_LR_TULU,
            "small-kl-tulu": EXP_SETUP_SMALL_KL_TULU,
            "high-kl-tulu": EXP_SETUP_HIGH_KL_TULU,
            "no-early-stop-tulu": EXP_SETUP_NO_EARLY_TULU,
        },
        # CLI Arguments
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "layers": get_search_locations(
            num_layers=32,
            num_probes=3,
            attn_path=None,
            mlp_path=None,
        ),
    },
    "phi-3-mini-it": {
        "experiments": {
            "baseline-tulu": EXP_SETUP_BASELINE_TULU,
            "small-lr-tulu": EXP_SETUP_SMALL_LR_TULU,
            "medium-lr-tulu": EXP_SETUP_MEDIUM_LR_TULU,
            "large-lr-tulu": EXP_SETUP_LARGE_LR_TULU,
            "small-kl-tulu": EXP_SETUP_SMALL_KL_TULU,
            "high-kl-tulu": EXP_SETUP_HIGH_KL_TULU,
            "no-early-stop-tulu": EXP_SETUP_NO_EARLY_TULU,
        },
        # CLI Arguments
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "layers": get_search_locations(
            num_layers=32,
            num_probes=3,
            attn_path=None,
            mlp_path=None,
        ),
    },
    "qwen-2.5-3b-instruct": {
        "experiments": {
            "baseline-tulu": EXP_SETUP_BASELINE_TULU,
            "small-lr-tulu": EXP_SETUP_SMALL_LR_TULU,
            "medium-lr-tulu": EXP_SETUP_MEDIUM_LR_TULU,
            "large-lr-tulu": EXP_SETUP_LARGE_LR_TULU,
            "small-kl-tulu": EXP_SETUP_SMALL_KL_TULU,
            "high-kl-tulu": EXP_SETUP_HIGH_KL_TULU,
            "no-early-stop-tulu": EXP_SETUP_NO_EARLY_TULU,
        },
        # CLI Arguments
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "layers": get_search_locations(
            num_layers=36,
            num_probes=3,
            attn_path=None,
            mlp_path=None,
        ),
    },
    # "llama-2-7b-chat": {
    #     "experiments": {
    #         "exp-rlhb": EXP_SETUP_HH_RLHF,
    #         "exp-orca": EXP_SETUP_SLIM_ORCA,
    #         "exp-oasst2": EXP_SETUP_OASST2,
    #         "exp-tulu": EXP_SETUP_TULU_V2,
    #         "exp-lmsys": EXP_SETUP_LMSYS,
    #     },
    #     # CLI Arguments
    #     "model": "meta-llama/Llama-2-7b-chat-hf",
    #     "layers": get_search_locations(
    #         num_layers=32,
    #         num_probes=3,
    #         attn_path=None,
    #         mlp_path=None,
    #     ),
    # },
    # "phi-3-mini-it": {
    #     "experiments": {
    #         "exp-rlhb": EXP_SETUP_HH_RLHF,
    #         "exp-orca": EXP_SETUP_SLIM_ORCA,
    #         "exp-oasst2": EXP_SETUP_OASST2,
    #         "exp-tulu": EXP_SETUP_TULU_V2,
    #         "exp-lmsys": EXP_SETUP_LMSYS,
    #     },
    #     # CLI Arguments
    #     "model": "microsoft/Phi-3-mini-4k-instruct",
    #     "layers": get_search_locations(
    #         num_layers=32,
    #         num_probes=3,
    #         attn_path=None,
    #         mlp_path=None,
    #     ),
    # },
    # "qwen-2.5-3b-instruct": {
    #     "experiments": {
    #         "exp-rlhb": EXP_SETUP_HH_RLHF,
    #         "exp-orca": EXP_SETUP_SLIM_ORCA,
    #         "exp-oasst2": EXP_SETUP_OASST2,
    #         "exp-tulu": EXP_SETUP_TULU_V2,
    #         "exp-lmsys": EXP_SETUP_LMSYS,
    #     },
    #     # CLI Arguments
    #     "model": "Qwen/Qwen2.5-3B-Instruct",
    #     "layers": get_search_locations(
    #         num_layers=36,
    #         num_probes=3,
    #         attn_path=None,
    #         mlp_path=None,
    #     ),
    # },
}


# =============================================================================
# Helper Functions
# =============================================================================


def build_command(run_config: Dict[str, Any], script_path: str, run_name: str, project_name: str) -> List[str]:
    """Constructs the subprocess command list from configuration dictionary."""
    cmd_args = [
        "python",
        script_path,
        "--run_name",
        run_name,
        "--project_name",
        project_name,
    ]

    for arg_name, arg_value in run_config.items():
        cmd_args.append(f"--{arg_name}")
        # Handle lists (repeated arguments or specific formatting based on implementation)
        # Assuming argparse nargs='+' style inputs
        if isinstance(arg_value, list):
            cmd_args.extend(map(str, arg_value))
        else:
            cmd_args.append(str(arg_value))

    return cmd_args


def collect_garbage():
    """Utility to clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def run_experiment(model_key: str, iteration: int, total_iters: int, name_suffix: str):
    """Executes experiments for a single model configuration."""

    # 1. Retrieve Model Configuration
    # Safe copy to avoid side effects
    model_conf = copy.deepcopy(MODELS[model_key])

    # Extract special control fields
    model_nick = model_conf["model"].split("/")[-1]
    model_experiments = model_conf.pop("experiments", {})

    print(f"\n>> Processing Model: {model_key} ({model_nick})")

    if not model_experiments:
        print(f"!! No experiments defined for {model_key}. Skipping.")
        return

    # 2. Iterate over experiments defined for this model
    for exp_setup_name, exp_params in model_experiments.items():
        full_run_name = f"{model_nick}-{exp_setup_name}{name_suffix}-iter{iteration}"

        print("=" * 60)
        print(f"▶️ Running Experiment: {full_run_name} ({iteration}/{total_iters})")
        print("=" * 60)

        # 3. Construct Run Parameters Strategy:
        #    Global Defaults
        #    -> MERGED WITH -> Model Params
        #    -> MERGED WITH -> Experiment Params

        run_params = copy.deepcopy(DEFAULT_RUN_ARGS)

        # Add all remaining model config items as CLI arguments
        run_params.update(model_conf)

        # Add experiment-specific overrides
        run_params.update(exp_params)

        # 4. Build and Execute Command
        cmd = build_command(run_config=run_params, script_path=SCRIPT_PATH, run_name=full_run_name, project_name=PROJECT_NAME)

        try:
            # print("Executing:", " ".join(cmd))
            collect_garbage()
            subprocess.run(cmd, shell=False, check=True)
        except KeyboardInterrupt:
            print("\n🚨 Execution Interrupted by User")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during execution of {full_run_name}: {e}")
            # Continue to next experiment/model
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run consolidated IML experiments.")
    parser.add_argument("--models", nargs="+", default=["all"], choices=list(MODELS.keys()) + ["all"], help="List of model keys to run, or 'all'")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix to append to experiment names.")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations to run each experiment.")

    args = parser.parse_args()

    target_models = list(MODELS.keys()) if "all" in args.models else args.models

    print(f"🚀 Starting IML Runs for models: {target_models}")
    print(f"📋 Global Default Args: {DEFAULT_RUN_ARGS}")

    for model_key in target_models:
        for iteration in range(1, args.iters + 1):
            print(f"\n--- Iteration {iteration} / {args.iters} ---")
            run_experiment(model_key, iteration, args.iters, args.name_suffix)
