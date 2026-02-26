import subprocess
import argparse
import sys
import copy
from typing import Dict, List, Any
from datetime import datetime
import time

# =============================================================================
# Global Constants & Configuration
# =============================================================================

PROJECT_NAME = "silent-norm-ablations-v2"
SCRIPT_PATH = "scripts/run_norm.py"
SBATCH_SCRIPT = "run_sbatch.sh"

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

kl_vals = [0.5, 1.0, 2.0, 10.0, 20.0]

EXPS_SETUP_OASST2 = {f"oasst2_kl={kl}": dict(
    learning_rate=0.01,
    kl_weight=kl,
    proj_weigh=1.0,
    dataset="oasst2",
) for kl in kl_vals}

EXPS_SETUP_LM_SYS = {f"lmsys-1m_kl={kl}": dict(
    learning_rate=0.01,
    kl_weight=kl,
    proj_weigh=1.0,
    dataset="lmsys-1m",
) for kl in kl_vals}



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
            **EXPS_SETUP_OASST2,
            **EXPS_SETUP_LM_SYS
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
            **EXPS_SETUP_OASST2,
            **EXPS_SETUP_LM_SYS
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
            **EXPS_SETUP_OASST2,
            **EXPS_SETUP_LM_SYS
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
    # "gemma-2b-it": {
    #     "experiments": {
    #         **EXPS_SETUP_OASST2,
    #         **EXPS_SETUP_LM_SYS
    #     },
    #     # CLI Arguments
    #     "model": "google/gemma-2b-it",
    #     "layers": get_search_locations(
    #         num_layers=18,
    #         num_probes=3,
    #         attn_path=None,
    #         mlp_path=None,
    #     ),
    #     "test_datasets": ["hh-rlhf", "oasst2", "tulu-v2", "lmsys-1m"],
    # },
}


# =============================================================================
# Helper Functions
# =============================================================================


def layer_short_tag(layer_path: str) -> str:
    """Create a compact tag from a full layer path for use in job / run names.

    Examples:
        model.layers.4          -> L4
        model.layers.4.self_attn -> L4-attn
        model.layers.4.mlp      -> L4-mlp
        model.embed_tokens      -> embed
        model.norm              -> norm
    """
    parts = layer_path.split(".")
    # model.layers.<idx>[.suffix]
    if "layers" in parts:
        idx = parts[parts.index("layers") + 1]
        suffix_parts = parts[parts.index("layers") + 2 :]
        suffix = "-" + "_".join(suffix_parts) if suffix_parts else ""
        return f"L{idx}{suffix}"
    # Fallback: last meaningful component (e.g. embed_tokens, norm)
    return parts[-1]


def build_script_args(run_config: Dict[str, Any], script_path: str, run_name: str, project_name: str) -> List[str]:
    """Constructs the script arguments (without 'python' prefix, since run_sbatch.sh handles that)."""
    cmd_args = [
        script_path,
        "--run_name",
        run_name,
        "--project_name",
        project_name,
    ]

    for arg_name, arg_value in run_config.items():
        cmd_args.append(f"--{arg_name}")
        if isinstance(arg_value, list):
            cmd_args.extend(map(str, arg_value))
        else:
            cmd_args.append(str(arg_value))

    return cmd_args


def submit_sbatch_job(
    script_args: List[str],
    job_name: str,
    gpu: str = "1",
    partition: str | None = None,
    cpus: int | None = None,
    dry_run: bool = False,
) -> str | None:
    """Submit a job to SLURM via sbatch.

    Args:
        script_args: Arguments to pass to run_sbatch.sh (i.e. script path + its args).
        job_name: SLURM job name.
        gpu: GPU resource spec for --gres (e.g. '1', 'A100:1', 'PRO6000:1').
        partition: Optional SLURM partition.
        cpus: Optional override for --cpus-per-task.
        dry_run: If True, print the command without submitting.

    Returns:
        The SLURM job ID on success, or None on failure.
    """
    sbatch_cmd: List[str] = [
        "sbatch",
        "--job-name", job_name,
    ]

    if gpu:
        sbatch_cmd.extend(["--gres", f"gpu:{gpu}"])
    if partition:
        sbatch_cmd.extend(["--partition", partition])
    if cpus is not None:
        sbatch_cmd.extend(["--cpus-per-task", str(cpus)])

    # The batch script followed by arguments it should forward
    sbatch_cmd.append(SBATCH_SCRIPT)
    sbatch_cmd.extend(script_args)

    if dry_run:
        print(f"  [DRY RUN] {' '.join(sbatch_cmd)}")
        return "DRY_RUN"

    try:
        result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"  ✓ Submitted: Job ID {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"  ✗ sbatch failed: {e.stderr.strip()}", file=sys.stderr)
        return None


def run_experiment(
    model_key: str,
    iteration: int,
    total_iters: int,
    name_suffix: str,
    gpu: str = "1",
    partition: str | None = None,
    cpus: int | None = None,
    dry_run: bool = False,
) -> List[str]:
    """Submits sbatch jobs for every experiment defined on a single model.

    Returns:
        List of submitted SLURM job IDs.
    """
    job_ids: List[str] = []

    # 1. Retrieve Model Configuration
    model_conf = copy.deepcopy(MODELS[model_key])

    # Extract special control fields
    model_nick = model_conf["model"].split("/")[-1]
    model_experiments = model_conf.pop("experiments", {})

    print(f"\n>> Processing Model: {model_key} ({model_nick})")

    if not model_experiments:
        print(f"!! No experiments defined for {model_key}. Skipping.")
        return job_ids

    # 2. Iterate over experiments defined for this model
    for exp_setup_name, exp_params in model_experiments.items():
        # 3. Construct base run parameters:
        #    Global Defaults -> Model Params -> Experiment Params
        base_params = copy.deepcopy(DEFAULT_RUN_ARGS)
        base_params.update(model_conf)
        base_params.update(exp_params)

        # Extract the layers list; each layer becomes its own job
        all_layers = base_params.pop("layers", [None])
        if not all_layers:
            all_layers = [None]  # single job with no --layers arg

        for layer in all_layers:
            # Build a per-layer run name & job name
            tag = layer_short_tag(layer) if layer else "all"
            full_run_name = f"{model_nick}-{exp_setup_name}{name_suffix}-{tag}-iter{iteration}"

            print("=" * 60)
            print(f"▶️ Submitting: {full_run_name} ({iteration}/{total_iters})")
            print("=" * 60)

            run_params = copy.deepcopy(base_params)
            if layer is not None:
                run_params["layers"] = [layer]  # single-element list

            # 4. Build script args and submit via sbatch
            script_args = build_script_args(
                run_config=run_params,
                script_path=SCRIPT_PATH,
                run_name=full_run_name,
                project_name=PROJECT_NAME,
            )

            job_id = submit_sbatch_job(
                script_args=script_args,
                job_name=full_run_name,
                gpu=gpu,
                partition=partition,
                cpus=cpus,
                dry_run=dry_run,
            )

            if job_id:
                job_ids.append(job_id)

    return job_ids


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit consolidated IML experiments via SLURM sbatch.")
    parser.add_argument("--models", nargs="+", default=["all"], choices=list(MODELS.keys()) + ["all"], help="List of model keys to run, or 'all'")
    parser.add_argument("--name_suffix", type=str, default="", help="Suffix to append to experiment names.")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations to run each experiment.")
    # SLURM options
    parser.add_argument("--gpu", type=str, default="L40:1", help="GPU resource for --gres (e.g. '1', 'A100:1', 'PRO6000:1').")
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition to submit to.")
    parser.add_argument("--cpus", type=int, default=8, help="Override --cpus-per-task.")
    parser.add_argument("--dry_run", action="store_true", help="Print sbatch commands without submitting.")

    args = parser.parse_args()

    target_models = list(MODELS.keys()) if "all" in args.models else args.models

    print(f"🚀 Submitting IML jobs for models: {target_models}")
    if args.dry_run:
        print("⚠️  DRY RUN — no jobs will actually be submitted.")
    print(f"📋 Global Default Args: {DEFAULT_RUN_ARGS}")

    all_job_ids: List[str] = []

    for model_key in target_models:
        for iteration in range(1, args.iters + 1):
            print(f"\n--- Iteration {iteration} / {args.iters} ---")
            ids = run_experiment(
                model_key,
                iteration,
                args.iters,
                args.name_suffix,
                gpu=args.gpu,
                partition=args.partition,
                cpus=args.cpus,
                dry_run=args.dry_run,
            )
            all_job_ids.extend(ids)

    print(f"\n{'='*60}")
    print(f"✅ Total jobs submitted: {len(all_job_ids)}")
    if all_job_ids and not args.dry_run:
        print(f"   Job IDs: {', '.join(all_job_ids)}")
