import subprocess
import argparse
import os

from typing import List
import torch
import gc

# =============================================================================
# Global Constants & Configuration
# =============================================================================

SCRIPT_PATH = "scripts/benchmark.py"

MODEL_CONFIGS = {
    "gemma-1": dict(
        batch_size=14,
        paths=[
            "logs/silent-norm-runs-v1/gemma-2b-it/oasst2_tulu-v3/model.embed_tokens",
            "logs/silent-norm-runs-v1/gemma-2b-it/oasst2_tulu-v3/model.layers.0",
            "logs/silent-norm-runs-v1/gemma-2b-it/oasst2_tulu-v3/model.layers.6",
        ],
    ),
    "gemma-2": dict(
        batch_size=14,
        paths=[
            "logs/silent-norm-runs-v1/gemma-2b-it/oasst2_tulu-v3/model.layers.12",
            "logs/silent-norm-runs-v1/gemma-2b-it/oasst2_tulu-v3/model.norm",
        ],
    ),
    "llama-1": dict(
        batch_size=8,
        paths=[
            "logs/silent-norm-runs-v1/Llama-2-7b-chat-hf/oasst2_tulu-v3/model.embed_tokens",
            "logs/silent-norm-runs-v1/Llama-2-7b-chat-hf/oasst2_tulu-v3/model.layers.0",
            "logs/silent-norm-runs-v1/Llama-2-7b-chat-hf/oasst2_tulu-v3/model.layers.11",
        ],
    ),
    "llama-2": dict(
        batch_size=8,
        paths=[
            "logs/silent-norm-runs-v1/Llama-2-7b-chat-hf/oasst2_tulu-v3/model.layers.21",
            "logs/silent-norm-runs-v1/Llama-2-7b-chat-hf/oasst2_tulu-v3/model.norm",
        ],
    ),    
    "phi-1": dict(
        batch_size=14,
        paths=[
            "logs/silent-norm-runs-v1/Phi-3-mini-4k-instruct/oasst2_tulu-v3/model.embed_tokens",
            "logs/silent-norm-runs-v1/Phi-3-mini-4k-instruct/oasst2_tulu-v3/model.layers.0",
            "logs/silent-norm-runs-v1/Phi-3-mini-4k-instruct/oasst2_tulu-v3/model.layers.11",
        ],
    ),
    "phi-2": dict(
        batch_size=14,
        paths=[
            "logs/silent-norm-runs-v1/Phi-3-mini-4k-instruct/oasst2_tulu-v3/model.layers.21",
            "logs/silent-norm-runs-v1/Phi-3-mini-4k-instruct/oasst2_tulu-v3/model.norm",
        ],
    ),
    "qwen-1": dict(
        batch_size=14,
        paths=[
            "logs/silent-norm-runs-v1/Qwen2.5-3B-Instruct/oasst2_tulu-v3/model.embed_tokens",
            "logs/silent-norm-runs-v1/Qwen2.5-3B-Instruct/oasst2_tulu-v3/model.layers.0",
            "logs/silent-norm-runs-v1/Qwen2.5-3B-Instruct/oasst2_tulu-v3/model.layers.6",
        ],
    ),
    "qwen-2": dict(
        batch_size=14,
        paths=[
            "logs/silent-norm-runs-v1/Qwen2.5-3B-Instruct/oasst2_tulu-v3/model.layers.12",
            "logs/silent-norm-runs-v1/Qwen2.5-3B-Instruct/oasst2_tulu-v3/model.norm",
        ],
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================


def validate_paths():
    """Checks that all specified paths in MODEL_CONFIGS exist."""

    result = True

    for model_key, config in MODEL_CONFIGS.items():
        for path in config["paths"]:
            if not os.path.exists(path):
                print(f"⚠️ Warning: Path does not exist for {model_key}: {path}")
                result = False

    return result


def build_command(model_config: dict) -> List[str]:
    """Constructs the subprocess command list from configuration dictionary."""
    cmd_args = [
        "python",
        SCRIPT_PATH,
        *model_config["paths"],
        "--batch_size",
        model_config["batch_size"],
        "--recurse",
    ]

    cmd_args = [str(arg) for arg in cmd_args]

    return cmd_args


def collect_garbage():
    """Utility to clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def run_benchmarks(model_key: str, args):
    """Executes benchmarks for a single model configuration."""

    cmd = build_command(MODEL_CONFIGS[model_key])

    try:
        print("Executing:", " ".join(cmd))
        collect_garbage()
        subprocess.run(cmd, shell=False, check=True)

    except KeyboardInterrupt as e:
        print("\n🚨 Execution Interrupted by User")
        raise e

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during execution of {model_key}: {e}")

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
    res = validate_paths()
    print(f"✅ Path validation result: {res}")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        help="List of model keys to run, or 'all'",
    )

    parser.add_argument(
        "--allow_exceptions",
        action="store_true",
        help="Whether to continue running other experiments if one fails.",
    )

    args = parser.parse_args()

    target_models = list(MODEL_CONFIGS.keys()) if "all" in args.models else args.models

    print(f"🚀 Starting Runs for models: {target_models}")

    for model_key in target_models:
        run_benchmarks(model_key, args)



