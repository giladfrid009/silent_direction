import subprocess
import argparse
import copy
from typing import Dict, List, Any
import torch
import gc

# =============================================================================
# Global Constants & Configuration
# =============================================================================

PROJECT_NAME = "silent-norm-runs-v3"
SCRIPT_PATH = "scripts/run_norm.py"

# Default Arguments shared across all runs unless overridden
DEFAULT_RUN_ARGS = {
    "log_dir": "logs",
    "train_batch": 8,
    "train_steps": 20000,
    "train_time": 40,
    "train_patience": 200,
    "eval_batch": 8,
    "eval_steps": 250,
}

# =============================================================================
# Experiment Setups
# =============================================================================

# Define reusable experiment configurations
# Using clear names for standard setups.
# These dictionaries are essentially just sets of CLI arguments.

EXP_SETUP_BASE = dict(
    learning_rate=0.1,
    kl_weight=10.0,
    proj_weigh=1.0,
    dataset=["oasst2", "tulu-v3"],
    train_patience_delta=0.005,
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

    locations += special_locations

    return locations


MODELS = {
    "llama-2-7b-chat": {
        "experiments": {
            # "KL-0.0": {**EXP_SETUP_BASE, **dict(kl_weight=0.0)},
            # "KL-0.125": {**EXP_SETUP_BASE, **dict(kl_weight=0.125)},
            # "KL-0.25": {**EXP_SETUP_BASE, **dict(kl_weight=0.25)},
            # "KL-0.5": {**EXP_SETUP_BASE, **dict(kl_weight=0.5)},
            "KL-1.0": {**EXP_SETUP_BASE, **dict(kl_weight=1.0)},
            # "KL-2.0": {**EXP_SETUP_BASE, **dict(kl_weight=2.0)},
            # "KL-4.0": {**EXP_SETUP_BASE, **dict(kl_weight=4.0)},
            # "KL-8.0": {**EXP_SETUP_BASE, **dict(kl_weight=8.0)},
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
            # "KL-0.0": {**EXP_SETUP_BASE, **dict(kl_weight=0.0)},
            # "KL-0.125": {**EXP_SETUP_BASE, **dict(kl_weight=0.125)},
            # "KL-0.25": {**EXP_SETUP_BASE, **dict(kl_weight=0.25)},
            # "KL-0.5": {**EXP_SETUP_BASE, **dict(kl_weight=0.5)},
            "KL-1.0": {**EXP_SETUP_BASE, **dict(kl_weight=1.0)},
            # "KL-2.0": {**EXP_SETUP_BASE, **dict(kl_weight=2.0)},
            # "KL-4.0": {**EXP_SETUP_BASE, **dict(kl_weight=4.0)},
            # "KL-8.0": {**EXP_SETUP_BASE, **dict(kl_weight=8.0)},
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
            # "KL-0.0": {**EXP_SETUP_BASE, **dict(kl_weight=0.0)},
            # "KL-0.125": {**EXP_SETUP_BASE, **dict(kl_weight=0.125)},
            # "KL-0.25": {**EXP_SETUP_BASE, **dict(kl_weight=0.25)},
            # "KL-0.5": {**EXP_SETUP_BASE, **dict(kl_weight=0.5)},
            "KL-1.0": {**EXP_SETUP_BASE, **dict(kl_weight=1.0)},
            # "KL-2.0": {**EXP_SETUP_BASE, **dict(kl_weight=2.0)},
            # "KL-4.0": {**EXP_SETUP_BASE, **dict(kl_weight=4.0)},
            # "KL-8.0": {**EXP_SETUP_BASE, **dict(kl_weight=8.0)},
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
    "gemma-2b-it": {
        "experiments": {
            # "KL-0.0": {**EXP_SETUP_BASE, **dict(kl_weight=0.0)},
            # "KL-0.125": {**EXP_SETUP_BASE, **dict(kl_weight=0.125)},
            # "KL-0.25": {**EXP_SETUP_BASE, **dict(kl_weight=0.25)},
            # "KL-0.5": {**EXP_SETUP_BASE, **dict(kl_weight=0.5)},
            "KL-1.0": {**EXP_SETUP_BASE, **dict(kl_weight=1.0)},
            # "KL-2.0": {**EXP_SETUP_BASE, **dict(kl_weight=2.0)},
            # "KL-4.0": {**EXP_SETUP_BASE, **dict(kl_weight=4.0)},
            # "KL-8.0": {**EXP_SETUP_BASE, **dict(kl_weight=8.0)},
        },
        # CLI Arguments
        "model": "google/gemma-2b-it",
        "layers": get_search_locations(
            num_layers=18,
            num_probes=3,
            attn_path=None,
            mlp_path=None,
        ),
    },
    "qwen-2.5-0.5b-instruct-runs1": {
        "experiments": {
            "KL-0.0": {**EXP_SETUP_BASE, **dict(kl_weight=0.0)},
            # "KL-0.125": {**EXP_SETUP_BASE, **dict(kl_weight=0.125)},
            "KL-0.25": {**EXP_SETUP_BASE, **dict(kl_weight=0.25)},
            # "KL-0.5": {**EXP_SETUP_BASE, **dict(kl_weight=0.5)},
            # "KL-1.0": {**EXP_SETUP_BASE, **dict(kl_weight=1.0)},
            # "KL-2.0": {**EXP_SETUP_BASE, **dict(kl_weight=2.0)},
            # "KL-4.0": {**EXP_SETUP_BASE, **dict(kl_weight=4.0)},
            # "KL-8.0": {**EXP_SETUP_BASE, **dict(kl_weight=8.0)},
        },
        # CLI Arguments
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers": get_search_locations(
            num_layers=24,
            num_probes=3,
            attn_path=None,
            mlp_path=None,
        ),
    },
    "qwen-2.5-0.5b-instruct-runs2": {
        "experiments": {
            # "KL-0.0": {**EXP_SETUP_BASE, **dict(kl_weight=0.0)},
            # "KL-0.125": {**EXP_SETUP_BASE, **dict(kl_weight=0.125)},
            # "KL-0.25": {**EXP_SETUP_BASE, **dict(kl_weight=0.25)},
            # "KL-0.5": {**EXP_SETUP_BASE, **dict(kl_weight=0.5)},
            "KL-1.0": {**EXP_SETUP_BASE, **dict(kl_weight=1.0)},
            # "KL-2.0": {**EXP_SETUP_BASE, **dict(kl_weight=2.0)},
            "KL-4.0": {**EXP_SETUP_BASE, **dict(kl_weight=4.0)},
            # "KL-8.0": {**EXP_SETUP_BASE, **dict(kl_weight=8.0)},
        },
        # CLI Arguments
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers": get_search_locations(
            num_layers=24,
            num_probes=3,
            attn_path=None,
            mlp_path=None,
        ),
    },
    "qwen-2.5-14b-instruct-runs1": {
        "experiments": {
            "KL-0.0": {
                **EXP_SETUP_BASE,
                **dict(
                    kl_weight=0.0,
                    train_batch=2,
                    train_time=60*3,
                    eval_batch=2,
                    eval_time=15,
                    window_size=40,
                ),
            },
            # "KL-0.125": {**EXP_SETUP_BASE, **dict(kl_weight=0.125)},
            # "KL-0.25": {**EXP_SETUP_BASE, **dict(kl_weight=0.25)},
            # "KL-0.5": {**EXP_SETUP_BASE, **dict(kl_weight=0.5)},
            # "KL-1.0": {**EXP_SETUP_BASE, **dict(kl_weight=1.0)},
            # "KL-2.0": {**EXP_SETUP_BASE, **dict(kl_weight=2.0)},
            # "KL-4.0": {**EXP_SETUP_BASE, **dict(kl_weight=4.0)},
            # "KL-8.0": {**EXP_SETUP_BASE, **dict(kl_weight=8.0)},
        },
        # CLI Arguments
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "layers": get_search_locations(
            num_layers=48,
            num_probes=0,
            attn_path=None,
            mlp_path=None,
            special_locations=["model.norm"],
        ),
    },
    "qwen-2.5-14b-instruct-runs2": {
        "experiments": {
            # "KL-0.0": {**EXP_SETUP_BASE, **dict(kl_weight=0.0)},
            # "KL-0.125": {**EXP_SETUP_BASE, **dict(kl_weight=0.125)},
            "KL-0.25": {
                **EXP_SETUP_BASE,
                **dict(
                    kl_weight=0.25,
                    train_batch=2,
                    train_time=60 * 3,
                    eval_batch=2,
                    eval_time=15,
                    window_size=40,
                ),
            },
            # "KL-0.5": {**EXP_SETUP_BASE, **dict(kl_weight=0.5)},
            # "KL-1.0": {**EXP_SETUP_BASE, **dict(kl_weight=1.0)},
            # "KL-2.0": {**EXP_SETUP_BASE, **dict(kl_weight=2.0)},
            # "KL-4.0": {**EXP_SETUP_BASE, **dict(kl_weight=4.0)},
            # "KL-8.0": {**EXP_SETUP_BASE, **dict(kl_weight=8.0)},
        },
        # CLI Arguments
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "layers": get_search_locations(
            num_layers=48,
            num_probes=0,
            attn_path=None,
            mlp_path=None,
            special_locations=["model.norm"],
        ),
    },
}


# MODELS = {
#     "llama-2-7b-chat": {
#         "experiments": {
#             "exp-rlhb": EXP_SETUP_HH_RLHF,
#             "exp-orca": EXP_SETUP_SLIM_ORCA,
#             "exp-oasst2": EXP_SETUP_OASST2,
#             "exp-tulu": EXP_SETUP_TULU_V2,
#             "exp-lmsys": EXP_SETUP_LMSYS,
#         },
#         # CLI Arguments
#         "model": "meta-llama/Llama-2-7b-chat-hf",
#         "layers": get_search_locations(
#             num_layers=32,
#             num_probes=8,
#             attn_path=None,
#             mlp_path=None,
#         ),
#     },
#     "phi-3-mini-it": {
#         "experiments": {
#             "exp-rlhb": EXP_SETUP_HH_RLHF,
#             "exp-orca": EXP_SETUP_SLIM_ORCA,
#             "exp-oasst2": EXP_SETUP_OASST2,
#             "exp-tulu": EXP_SETUP_TULU_V2,
#             "exp-lmsys": EXP_SETUP_LMSYS,
#         },
#         # CLI Arguments
#         "model": "microsoft/Phi-3-mini-4k-instruct",
#         "layers": get_search_locations(
#             num_layers=32,
#             num_probes=8,
#             attn_path=None,
#             mlp_path=None,
#         ),
#     },
#     "gemma-2b-it": {
#         "experiments": {
#             # "exp-rlhb": EXP_SETUP_HH_RLHF,
#             # "exp-orca": EXP_SETUP_SLIM_ORCA,
#             # "exp-oasst2": EXP_SETUP_OASST2,
#             "exp-tulu": EXP_SETUP_TULU_V2,
#             # "exp-lmsys": EXP_SETUP_LMSYS,
#         },
#         # CLI Arguments
#         "model": "google/gemma-2b-it",
#         "layers": get_search_locations(
#             num_layers=18,
#             num_probes=8,
#             attn_path=None,
#             mlp_path=None,
#         ),
#         "test_datasets": ["hh-rlhf", "oasst2", "tulu-v2", "lmsys-1m"],
#     },
#     "gemma-2-2b-it": {
#         "experiments": {
#             # "exp-rlhb": EXP_SETUP_HH_RLHF,
#             # "exp-orca": EXP_SETUP_SLIM_ORCA,
#             # "exp-oasst2": EXP_SETUP_OASST2,
#             "exp-tulu": EXP_SETUP_TULU_V2,
#             # "exp-lmsys": EXP_SETUP_LMSYS,
#         },
#         # CLI Arguments
#         "model": "google/gemma-2-2b-it",
#         "layers": get_search_locations(
#             num_layers=26,
#             num_probes=8,
#             attn_path=None,
#             mlp_path=None,
#         ),
#     },
#     "gemma-2-27b-it": {
#         "experiments": {
#             # "exp-rlhb": EXP_SETUP_HH_RLHF,
#             # "exp-orca": EXP_SETUP_SLIM_ORCA,
#             # "exp-oasst2": EXP_SETUP_OASST2,
#             "exp-tulu": EXP_SETUP_TULU_V2,
#             # "exp-lmsys": EXP_SETUP_LMSYS,
#         },
#         # CLI Arguments
#         "model": "google/gemma-2-27b-it",
#         "layers": get_search_locations(
#             num_layers=46,
#             num_probes=8,
#             attn_path=None,
#             mlp_path=None,
#         ),
#     },
#     "qwen-2.5-3b-instruct": {
#         "experiments": {
#             # "exp-rlhb": EXP_SETUP_HH_RLHF,
#             # "exp-orca": EXP_SETUP_SLIM_ORCA,
#             # "exp-oasst2": EXP_SETUP_OASST2,
#             "exp-tulu": EXP_SETUP_TULU_V2,
#             # "exp-lmsys": EXP_SETUP_LMSYS,
#         },
#         # CLI Arguments
#         "model": "Qwen/Qwen2.5-3B-Instruct",
#         "layers": get_search_locations(
#             num_layers=36,
#             num_probes=8,
#             attn_path=None,
#             mlp_path=None,
#         ),
#     },
#     "phi-3-medium-it": {
#         "experiments": {
#             # "exp-rlhb": EXP_SETUP_HH_RLHF,
#             # "exp-orca": EXP_SETUP_SLIM_ORCA,
#             # "exp-oasst2": EXP_SETUP_OASST2,
#             "exp-tulu": EXP_SETUP_TULU_V2,
#             # "exp-lmsys": EXP_SETUP_LMSYS,
#         },
#         # CLI Arguments
#         "model": "microsoft/Phi-3-medium-4k-instruct",
#         "layers": get_search_locations(
#             num_layers=40,
#             num_probes=8,
#             attn_path=None,
#             mlp_path=None,
#         ),
#     },
# }


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


def run_experiment(model_key: str, iteration: int, total_iters: int, name_suffix: str, args):
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

        except KeyboardInterrupt as e:
            print("\n🚨 Execution Interrupted by User")
            raise e

        except subprocess.CalledProcessError as e:
            print(f"❌ Error during execution of {full_run_name}: {e}")

            if not args.allow_exceptions:
                print("Aborting further runs due to error.")
                raise e

            # Continue to next experiment/model
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

            if not args.allow_exceptions:
                print("Aborting further runs due to error.")
                raise e


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run consolidated IML experiments.")
    parser.add_argument("--models", nargs="+", default=["all"], choices=list(MODELS.keys()) + ["all"], help="List of model keys to run, or 'all'")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix to append to experiment names.")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations to run each experiment.")
    parser.add_argument("--allow_exceptions", action="store_true", help="Whether to continue running other experiments if one fails.")

    args = parser.parse_args()

    target_models = list(MODELS.keys()) if "all" in args.models else args.models

    print(f"🚀 Starting IML Runs for models: {target_models}")
    print(f"📋 Global Default Args: {DEFAULT_RUN_ARGS}")

    for model_key in target_models:
        for iteration in range(1, args.iters + 1):
            print(f"\n--- Iteration {iteration} / {args.iters} ---")
            run_experiment(model_key, iteration, args.iters, args.name_suffix, args)
