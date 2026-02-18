# NOTE: GPT5 Generated

import argparse
import pathlib
import sys
import json
import pandas as pd


# set pythonpath to the main module directory
module_dir = pathlib.Path(__file__).parent.resolve().parent
if str(module_dir) not in sys.path:
    sys.path.append(str(module_dir))

from src.utils.logging import create_logger, setup_logging, loglevel_names  # noqa: E402


logger = create_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect benchmark JSON results into one DataFrame.")

    parser.add_argument(
        "root",
        type=str,
        help="Root folder to search recursively.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Optional output file (.csv/.parquet/.jsonl).",
    )

    parser.add_argument(
        "--log_level",
        type=str,
        choices=loglevel_names(),
        default="INFO",
        metavar="LEVEL",
        help=f"Logging level to python-logger. Available levels: {loglevel_names()}",
    )

    return parser.parse_args()


def _read_result_file(path: pathlib.Path) -> pd.DataFrame | None:
    """Read numeric results from a benchmark result file."""
    all_data = []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        results = data.get("results")
        if not isinstance(results, dict):
            return None

        # results is a dict of dicts:
        #   key   = benchmark name
        #   value = dict(metric_name -> metric_value)
        for bench_name, bench_results in results.items():
            if not isinstance(bench_results, dict):
                continue

            for metric_name, metric_value in bench_results.items():
                if isinstance(metric_value, (int, float)):
                    all_data.append(
                        {
                            "benchmark": str(bench_name),
                            "metric": str(metric_name),
                            "value": float(metric_value),
                        }
                    )

        if not all_data:
            return pd.DataFrame(columns=["benchmark", "metric", "value"])

        return pd.DataFrame(all_data)

    except Exception:
        return None


def find_benchmarks_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    """
    Find directories named 'benchmarks' that contain at least one *.json file.
    """
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    dirs: list[pathlib.Path] = []

    # rglob is simple and robust; we look for "benchmarks" directories.
    for p in root.rglob("benchmarks"):
        if not p.is_dir():
            continue
        # Require at least one json file directly inside benchmarks/
        if any(child.is_file() and child.suffix.lower() == ".json" for child in p.iterdir()):
            dirs.append(p.resolve())

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def collect_from_benchmarks_dir(bench_dir: pathlib.Path, root: pathlib.Path) -> pd.DataFrame:
    """
    Read all *.json files in a benchmarks directory into one DataFrame,
    adding a `path` column (full path to the benchmarks directory) and `file` column.
    """
    json_files = sorted([p for p in bench_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"])

    per_file_dfs: list[pd.DataFrame] = []
    for jf in json_files:
        df = _read_result_file(jf)
        if df is None:
            logger.warning("Failed to parse JSON (skipping): %s", jf)
            continue

        # Add file-level provenance; helpful when multiple jsons exist per dir.
        df.insert(0, "file", jf.name)
        rel_path = bench_dir.relative_to(root, walk_up=True)

        df.insert(0, "path", str(rel_path))
        per_file_dfs.append(df)

    if not per_file_dfs:
        return pd.DataFrame(columns=["path", "file", "benchmark", "metric", "value"])

    out = pd.concat(per_file_dfs, ignore_index=True)

    # Ensure stable dtypes
    out["path"] = out["path"].astype("string")
    out["file"] = out["file"].astype("string")
    out["benchmark"] = out["benchmark"].astype("string")
    out["metric"] = out["metric"].astype("string")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    # Drop any rows that somehow became NaN (e.g., weird metric types)
    out = out.dropna(subset=["value"]).reset_index(drop=True)

    return out


def collect_all(root: pathlib.Path) -> pd.DataFrame:
    """
    Find all benchmarks directories under root and return one unified DataFrame.
    """
    root = root.resolve()
    bench_dirs = find_benchmarks_dirs(root)
    logger.info("Found %d benchmarks directories under %s", len(bench_dirs), root)

    all_dfs: list[pd.DataFrame] = []
    for d in bench_dirs:
        df = collect_from_benchmarks_dir(d, root)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["path", "file", "benchmark", "metric", "value"])

    out = pd.concat(all_dfs, ignore_index=True)

    # Optional: sort for readability
    out = out.sort_values(["path", "file", "benchmark", "metric"], kind="stable").reset_index(drop=True)

    return out


def write_output(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(out_path, index=False)
    elif suffix in (".parquet", ".pq"):
        df.to_parquet(out_path, index=False)
    elif suffix in (".jsonl", ".json"):
        # JSON Lines is usually best for big outputs; .json will still be lines here.
        df.to_json(out_path, orient="records", lines=True, indent=4)
    else:
        raise ValueError(f"Unsupported output extension: {suffix} (use .csv, .parquet, .jsonl)")


def main(args) -> int:
    root = pathlib.Path(args.root)
    df = collect_all(root)

    logger.info("Collected %d rows", len(df))
    if len(df) > 0:
        # Quick sanity summary
        logger.info(
            "Unique benchmarks=%d, metrics=%d, benchmark dirs=%d",
            df["benchmark"].nunique(dropna=True),
            df["metric"].nunique(dropna=True),
            df["path"].nunique(dropna=True),
        )

    if args.output_path:
        write_output(df, pathlib.Path(args.output_path))
        logger.info("Wrote output to %s", pathlib.Path(args.output_path).resolve())
    else:
        # Print a compact preview
        with pd.option_context("display.max_rows", 20, "display.max_columns", 20, "display.width", 140):
            print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    args = parse_args()
    setup_logging(level=args.log_level)
    main(args)
