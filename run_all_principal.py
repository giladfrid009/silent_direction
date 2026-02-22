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

PROJECT_NAME = "silent-principal"
SCRIPT_PATH = "scripts/run_principal.py"

# Default Arguments shared across all runs unless overridden
DEFAULT_RUN_ARGS = {
    "log_dir": "logs",
    "dataset": "hh-rlhf",
    "train_batch": 4,
    "eval_batch": 4,
    "train_time": 40,
    "train_steps": 2000,
    "train_patience": 100,
}

# =============================================================================
# Experiment Setups
# =============================================================================

# Define reusable experiment configurations
# Using clear names for standard setups.
# These dictionaries are essentially just sets of CLI arguments.

EXP_SETUP_PLACEHOLDER = dict(
    learning_rate=0.01,
    kl_weight=1.0,
    proj_weigh=10.0,
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
            "setup": EXP_SETUP_PLACEHOLDER,
        },
        # CLI Arguments
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "layers": get_search_locations(
            num_layers=32,
            num_probes=3,
        ),
    },
    "llama-3.1-8b": {
        "experiments": {
            "setup": EXP_SETUP_PLACEHOLDER,
        },
        # CLI Arguments
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "layers": get_search_locations(
            num_layers=32,
            num_probes=4,
        ),
    },
    "gemma-2b-it": {
        "experiments": {
            "setup": EXP_SETUP_PLACEHOLDER,
        },
        # CLI Arguments
        "model": "google/gemma-2b-it",
        "layers": get_search_locations(
            num_layers=18,
            num_probes=4,
        ),
    },
    "gemma-3-1b-it": {
        "experiments": {
            "setup": EXP_SETUP_PLACEHOLDER,
        },
        # CLI Arguments
        "model": "google/gemma-3-1b-it",
        "layers": get_search_locations(
            num_layers=26,
            num_probes=4,
        ),
    },
    "qwen-1.5-7b-chat": {
        "experiments": {
            "setup": EXP_SETUP_PLACEHOLDER,
        },
        # CLI Arguments
        "model": "Qwen/Qwen1.5-7B-Chat",
        "layers": get_search_locations(
            num_layers=32,
            num_probes=4,
        ),
    },
    "qwen-3-8b-chat": {
        "experiments": {
            "setup": EXP_SETUP_PLACEHOLDER,
        },
        # CLI Arguments
        "model": "Qwen/Qwen3-8B",
        "layers": get_search_locations(
            num_layers=36,
            num_probes=4,
        ),
    },
    "phi-3-mini-it": {
        "experiments": {
            "setup": EXP_SETUP_PLACEHOLDER,
        },
        # CLI Arguments
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "layers": get_search_locations(
            num_layers=32,
            num_probes=4,
        ),
    },
    "phi-4-mini-it": {
        "experiments": {
            "setup": EXP_SETUP_PLACEHOLDER,
        },
        # CLI Arguments
        "model": "microsoft/Phi-4-mini-instruct",
        "layers": get_search_locations(
            num_layers=32,
            num_probes=4,
        ),
    },
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
