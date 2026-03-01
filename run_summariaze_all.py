import subprocess
import argparse
import os

from typing import List
import torch
import gc

# =============================================================================
# Global Constants & Configuration
# =============================================================================

SCRIPT_PATH = "scripts/summarize.py"

MODEL_CONFIGS = {
    "llama-2-7b-chat": dict(
        batch_size=8,
        paths=[
            # lmsys-1m runs
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/lmsys-1m/model.embed_tokens/Llama-2-7b-chat-hf-lmsys-1m_kl=0.5-embed_tokens-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/lmsys-1m/model.layers.0/Llama-2-7b-chat-hf-lmsys-1m_kl=0.5-L0-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/lmsys-1m/model.layers.11/Llama-2-7b-chat-hf-lmsys-1m_kl=0.5-L11-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/lmsys-1m/model.layers.21/Llama-2-7b-chat-hf-lmsys-1m_kl=0.5-L21-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/lmsys-1m/model.norm/Llama-2-7b-chat-hf-lmsys-1m_kl=0.5-norm-iter1/benchmarks",
            # oasst2 runs
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/oasst2/model.embed_tokens/Llama-2-7b-chat-hf-oasst2_kl=0.5-embed_tokens-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/oasst2/model.layers.0/Llama-2-7b-chat-hf-oasst2_kl=0.5-L0-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/oasst2/model.layers.11/Llama-2-7b-chat-hf-oasst2_kl=0.5-L11-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/oasst2/model.layers.21/Llama-2-7b-chat-hf-oasst2_kl=0.5-L21-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/oasst2/model.norm/Llama-2-7b-chat-hf-oasst2_kl=0.5-norm-iter1/benchmarks",
            # tulu-v2 runs
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/tulu-v2/model.embed_tokens/Llama-2-7b-chat-hf-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/tulu-v2/model.layers.0/Llama-2-7b-chat-hf-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/tulu-v2/model.layers.11/Llama-2-7b-chat-hf-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/tulu-v2/model.layers.21/Llama-2-7b-chat-hf-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Llama-2-7b-chat-hf/tulu-v2/model.norm/Llama-2-7b-chat-hf-small-kl-tulu-iter1/benchmarks",
        ],
    ),
    "phi-3-mini-it": dict(
        batch_size=10,
        paths=[
            # lmsys-1m runs
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/lmsys-1m/model.embed_tokens/Phi-3-mini-4k-instruct-lmsys-1m_kl=0.5-embed_tokens-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/lmsys-1m/model.layers.0/Phi-3-mini-4k-instruct-lmsys-1m_kl=0.5-L0-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/lmsys-1m/model.layers.11/Phi-3-mini-4k-instruct-lmsys-1m_kl=0.5-L11-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/lmsys-1m/model.layers.21/Phi-3-mini-4k-instruct-lmsys-1m_kl=0.5-L21-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/lmsys-1m/model.norm/Phi-3-mini-4k-instruct-lmsys-1m_kl=0.5-norm-iter1/benchmarks",
            # oasst2 runs
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/oasst2/model.embed_tokens/Phi-3-mini-4k-instruct-oasst2_kl=0.5-embed_tokens-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/oasst2/model.layers.0/Phi-3-mini-4k-instruct-oasst2_kl=0.5-L0-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/oasst2/model.layers.11/Phi-3-mini-4k-instruct-oasst2_kl=0.5-L11-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/oasst2/model.layers.21/Phi-3-mini-4k-instruct-oasst2_kl=0.5-L21-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/oasst2/model.norm/Phi-3-mini-4k-instruct-oasst2_kl=0.5-norm-iter1/benchmarks",
            # tulu-v2 runs
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/tulu-v2/model.embed_tokens/Phi-3-mini-4k-instruct-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/tulu-v2/model.layers.0/Phi-3-mini-4k-instruct-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/tulu-v2/model.layers.11/Phi-3-mini-4k-instruct-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/tulu-v2/model.layers.21/Phi-3-mini-4k-instruct-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/Phi-3-mini-4k-instruct/tulu-v2/model.norm/Phi-3-mini-4k-instruct-small-kl-tulu-iter1/benchmarks",
        ],
    ),
    "qwen-2.5-3b-instruct": dict(
        batch_size=14,
        paths=[],
    ),
    "gemma-2b-it": dict(
        batch_size=16,
        paths=[
            # lmsys-1m runs
            "logs/silent-norm-ablations-v2/gemma-2b-it/lmsys-1m/model.embed_tokens/gemma-2b-it-lmsys-1m_kl=0.5-embed_tokens-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/lmsys-1m/model.layers.0/gemma-2b-it-lmsys-1m_kl=0.5-L0-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/lmsys-1m/model.layers.6/gemma-2b-it-lmsys-1m_kl=0.5-L6-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/lmsys-1m/model.layers.12/gemma-2b-it-lmsys-1m_kl=0.5-L12-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/lmsys-1m/model.norm/gemma-2b-it-lmsys-1m_kl=0.5-norm-iter1/benchmarks",
            # oasst2 runs
            "logs/silent-norm-ablations-v2/gemma-2b-it/oasst2/model.embed_tokens/gemma-2b-it-oasst2_kl=0.5-embed_tokens-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/oasst2/model.layers.0/gemma-2b-it-oasst2_kl=0.5-L0-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/oasst2/model.layers.6/gemma-2b-it-oasst2_kl=0.5-L6-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/oasst2/model.layers.12/gemma-2b-it-oasst2_kl=0.5-L12-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/oasst2/model.norm/gemma-2b-it-oasst2_kl=0.5-norm-iter1/benchmarks",
            # tulu-v2 runs
            "logs/silent-norm-ablations-v2/gemma-2b-it/tulu-v2/model.embed_tokens/gemma-2b-it-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/tulu-v2/model.layers.0/gemma-2b-it-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/tulu-v2/model.layers.6/gemma-2b-it-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/tulu-v2/model.layers.12/gemma-2b-it-small-kl-tulu-iter1/benchmarks",
            "logs/silent-norm-ablations-v2/gemma-2b-it/tulu-v2/model.norm/gemma-2b-it-small-kl-tulu-iter1/benchmarks",
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


def build_command(paths: list[str]) -> List[str]:
    """Constructs the subprocess command list from configuration dictionary."""
    cmd_args = [
        "python",
        SCRIPT_PATH,
        *paths,
        "--output_path",
        "res.json"
    ]

    cmd_args = [str(arg) for arg in cmd_args]

    return cmd_args


def collect_garbage():
    """Utility to clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def run_summarize(model_keys: list[str], args):
    """Executes benchmarks for a list of model configurations."""

    paths = []
    for model_key in model_keys:
        paths.extend(MODEL_CONFIGS[model_key]["paths"])

    cmd = build_command(paths)

    print("Executing:", " ".join(cmd))
    collect_garbage()
    subprocess.run(cmd, shell=False, check=True)


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

    args = parser.parse_args()

    target_models = list(MODEL_CONFIGS.keys()) if "all" in args.models else args.models

    print(f"🚀 Starting Runs for models: {target_models}")

    run_summarize(target_models, args)
