from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


KNOWN_METRIC_KEYWORDS: tuple[str, ...] = (
	"acc",
	"f1",
	"match",
	"validity",
	"compliance",
	"pass",
)


def is_known_range_metric(metric_name: str, keywords: Iterable[str] = KNOWN_METRIC_KEYWORDS) -> bool:
	metric_name = str(metric_name).lower()
	return any(keyword in metric_name for keyword in keywords)


def get_benchmark_columns_by_keywords(
	df: pd.DataFrame,
	keywords: Iterable[str] = KNOWN_METRIC_KEYWORDS,
) -> list[str]:
	return [column_name for column_name in df.columns if is_known_range_metric(column_name, keywords=keywords)]


def compute_metric_statistics_by_group(
	df: pd.DataFrame,
	metric_columns: list[str] | None = None,
	group_columns: tuple[str, ...] = ("model_name", "train_dataset"),
	benchmark_metric_sep: str = " / ",
) -> pd.DataFrame:
	"""Compute per-metric summary stats for each group."""
	missing_group_cols = [column_name for column_name in group_columns if column_name not in df.columns]
	if missing_group_cols:
		raise ValueError(f"Missing group columns: {missing_group_cols}")

	if metric_columns is None:
		metric_columns = [column_name for column_name in df.columns if benchmark_metric_sep in str(column_name)]
	metric_columns = [column_name for column_name in metric_columns if column_name in df.columns]
	if not metric_columns:
		raise ValueError("No valid metric columns found for statistics")

	long_df = df[list(group_columns) + metric_columns].melt(
		id_vars=list(group_columns),
		value_vars=metric_columns,
		var_name="benchmark_metric",
		value_name="metric_value",
	)
	long_df["metric_value"] = pd.to_numeric(long_df["metric_value"], errors="coerce")
	long_df = long_df.dropna(subset=["metric_value"]).copy()

	split_cols = long_df["benchmark_metric"].str.split(benchmark_metric_sep, n=1, expand=True)
	long_df["benchmark"] = split_cols[0]
	long_df["metric"] = split_cols[1]

	agg_cols = list(group_columns) + ["benchmark_metric", "benchmark", "metric"]

	def _summarize(series: pd.Series) -> pd.Series:
		return pd.Series(
			{
				"mean": float(series.mean()),
				"q25": float(series.quantile(0.25)),
				"q50": float(series.quantile(0.50)),
				"q75": float(series.quantile(0.75)),
				"variance": float(series.var(ddof=1)) if len(series) > 1 else 0.0,
				"std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
				"count": int(series.count()),
			}
		)

	raw_stats = (
		long_df.groupby(agg_cols, dropna=False)["metric_value"]
		.apply(_summarize)
		.reset_index()
	)
	raw_stats["value_transform"] = "raw"

	abs_df = long_df.copy()
	abs_df["metric_value"] = abs_df["metric_value"].abs()
	abs_stats = (
		abs_df.groupby(agg_cols, dropna=False)["metric_value"]
		.apply(_summarize)
		.reset_index()
	)
	abs_stats["value_transform"] = "abs"

	out = pd.concat([raw_stats, abs_stats], ignore_index=True)
	out = out.sort_values(
		by=[*group_columns, "benchmark_metric", "value_transform"],
		ascending=[True] * len(group_columns) + [True, True],
	).reset_index(drop=True)
	return out


def compute_clean_dirty_difference_statistics(
	dirty_df: pd.DataFrame,
	clean_df: pd.DataFrame,
	group_columns: list[str] | None = None,
	benchmark_metric_sep: str = " / ",
	known_metric_keywords: Iterable[str] = KNOWN_METRIC_KEYWORDS,
) -> pd.DataFrame:
	"""Compute grouped summary stats on dirty-clean benchmark differences.

	For known-range metrics, differences are stored as percentage points.
	For unknown-range metrics, differences are stored as raw deltas.
	"""
	if group_columns is None:
		group_columns = []

	missing_group_cols = [column_name for column_name in group_columns if column_name not in dirty_df.columns]
	if missing_group_cols:
		raise ValueError(f"Missing group columns in dirty_df: {missing_group_cols}")

	if "model_name" not in dirty_df.columns or "model_name" not in clean_df.columns:
		raise ValueError("Both dirty_df and clean_df must contain a 'model_name' column")

	rows: list[dict] = []

	clean_by_model: dict[str, pd.Series] = {}
	for model_name, clean_rows in clean_df.groupby("model_name", dropna=False):
		if len(clean_rows) == 1:
			clean_by_model[str(model_name)] = clean_rows.iloc[0]

	for _, dirty_row in dirty_df.iterrows():
		model_name = str(dirty_row["model_name"])
		clean_row = clean_by_model.get(model_name)
		if clean_row is None:
			continue

		group_values = {column_name: dirty_row[column_name] for column_name in group_columns}
		metric_columns = [column_name for column_name in dirty_row.index if benchmark_metric_sep in str(column_name)]

		for benchmark_metric in metric_columns:
			if benchmark_metric not in clean_row.index:
				continue

			dirty_value = pd.to_numeric(pd.Series([dirty_row[benchmark_metric]]), errors="coerce").iloc[0]
			clean_value = pd.to_numeric(pd.Series([clean_row[benchmark_metric]]), errors="coerce").iloc[0]
			if pd.isna(dirty_value) or pd.isna(clean_value):
				continue

			benchmark_name, metric_name = str(benchmark_metric).split(benchmark_metric_sep, 1)
			diff_raw = float(dirty_value - clean_value)
			known_range = is_known_range_metric(metric_name, keywords=known_metric_keywords)

			rows.append(
				{
					**group_values,
					"model_name": dirty_row["model_name"],
					"benchmark": benchmark_name,
					"metric": metric_name,
					"benchmark_metric": benchmark_metric,
					"value_clean": float(clean_value),
					"value_dirty": float(dirty_value),
					"diff_raw": diff_raw,
					"diff_percent": diff_raw * 100.0 if known_range else None,
					"is_known_range": bool(known_range),
				}
			)

	diffs_df = pd.DataFrame(rows)
	if diffs_df.empty:
		return diffs_df

	groupby_cols = group_columns + ["benchmark", "metric", "is_known_range"]
	stats_rows: list[dict] = []

	for group_vals, group_df in diffs_df.groupby(groupby_cols, dropna=False):
		is_known_range = bool(group_vals[-1])
		diff_col = "diff_percent" if is_known_range else "diff_raw"
		diffs = pd.to_numeric(group_df[diff_col], errors="coerce").dropna()
		if diffs.empty:
			continue

		row_dict = {column_name: value for column_name, value in zip(groupby_cols, group_vals)}
		row_dict.update(
			{
				"mean_diff": float(diffs.mean()),
				"std_diff": float(diffs.std()),
				"median_diff": float(diffs.median()),
				"min_diff": float(diffs.min()),
				"max_diff": float(diffs.max()),
				"q25_diff": float(diffs.quantile(0.25)),
				"q75_diff": float(diffs.quantile(0.75)),
				"count": int(len(diffs)),
				"diff_unit": "% change" if is_known_range else "raw diff",
			}
		)
		stats_rows.append(row_dict)

	stats_df = pd.DataFrame(stats_rows)
	if not stats_df.empty:
		sort_cols = group_columns + ["benchmark", "metric"]
		stats_df = stats_df.sort_values(by=sort_cols).reset_index(drop=True)

	return stats_df


def as_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
	"""Return a Series, keeping first column when duplicate labels return a DataFrame."""
	if isinstance(obj, pd.DataFrame):
		return obj.iloc[:, 0]
	return obj


def safe_corr_xy(x: pd.Series, y: pd.Series) -> float:
	pair = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
	if len(pair) < 2:
		return np.nan
	if pair["x"].nunique(dropna=True) < 2 or pair["y"].nunique(dropna=True) < 2:
		return np.nan
	return float(pair["x"].corr(pair["y"]))


def compute_kl_target_correlations(
	dirty_df: pd.DataFrame,
	metric_columns: list[str] | None = None,
	kl_target_columns: list[str] | None = None,
	hue_column: str = "model_name",
	train_dataset_column: str = "train_dataset",
	combine_train_datasets: bool = True,
	exclude_metric_prefix: str | None = "eval",
	benchmark_metric_sep: str = " / ",
) -> pd.DataFrame:
	"""Compute metric-to-KL correlations using per-hue grouping, with optional dataset pooling."""
	if hue_column not in dirty_df.columns:
		raise ValueError(f"Missing hue column: {hue_column}")

	if not combine_train_datasets and train_dataset_column not in dirty_df.columns:
		raise ValueError(
			f"Missing train dataset column '{train_dataset_column}' while combine_train_datasets=False"
		)

	if metric_columns is None:
		metric_columns = [
			column_name
			for column_name in dirty_df.columns
			if benchmark_metric_sep in str(column_name)
		]
	else:
		metric_columns = [column_name for column_name in metric_columns if column_name in dirty_df.columns]

	if exclude_metric_prefix:
		metric_columns = [
			column_name for column_name in metric_columns if not str(column_name).startswith(exclude_metric_prefix)
		]

	if not metric_columns:
		raise ValueError("No valid metric columns were found in dataframe")

	if kl_target_columns is None:
		kl_target_columns = [
			column_name
			for column_name in dirty_df.columns
			if benchmark_metric_sep in str(column_name) and str(column_name).endswith(" / kl_div")
		]
	else:
		kl_target_columns = [column_name for column_name in kl_target_columns if column_name in dirty_df.columns]

	if not kl_target_columns:
		raise ValueError("No valid KL target columns were found in dataframe")

	if combine_train_datasets:
		scopes = [("ALL_TRAIN_DATASETS", dirty_df)]
	else:
		scopes = [(str(dataset_name), dataset_df) for dataset_name, dataset_df in dirty_df.groupby(train_dataset_column, dropna=False)]

	rows = []
	for train_scope, scope_df in scopes:
		grouped = scope_df.groupby(hue_column, dropna=False)
		group_names = list(grouped.groups.keys())

		for kl_col in kl_target_columns:
			for metric_col in metric_columns:
				if metric_col == kl_col:
					continue

				x_all = as_series(scope_df[kl_col])
				y_all = as_series(scope_df[metric_col])
				pooled_pair = pd.concat([x_all, y_all], axis=1, keys=["x", "y"]).dropna()
				pooled_corr = safe_corr_xy(x_all, y_all)
				n_rows_used_pooled = int(len(pooled_pair))

				group_corrs = []
				for _, group_df in grouped:
					x_group = as_series(group_df[kl_col])
					y_group = as_series(group_df[metric_col])
					corr_val = safe_corr_xy(x_group, y_group)
					if pd.notna(corr_val):
						group_corrs.append(float(corr_val))

				avg_group_corr = float(np.mean(group_corrs)) if group_corrs else np.nan

				rows.append(
					{
						"train_scope": train_scope,
						"kl_target": kl_col,
						"metric": metric_col,
						"pooled_corr": pooled_corr,
						"avg_group_corr": avg_group_corr,
						"n_groups_total": len(group_names),
						"n_groups_with_corr": len(group_corrs),
						"n_rows_total": int(len(scope_df)),
						"n_rows_used_pooled": n_rows_used_pooled,
					}
				)

	out = pd.DataFrame(rows)
	if out.empty:
		return out

	out["abs_avg_group_corr"] = out["avg_group_corr"].abs()
	out["abs_pooled_corr"] = out["pooled_corr"].abs()
	out = out.sort_values(
		by=["train_scope", "kl_target", "abs_avg_group_corr"],
		ascending=[True, True, False],
	).reset_index(drop=True)
	return out


def summarize_kl_target_correlations(
	dirty_df: pd.DataFrame,
	metric_columns: list[str] | None = None,
	kl_target_columns: list[str] | None = None,
	hue_column: str = "model_name",
	train_dataset_column: str = "train_dataset",
	combine_train_datasets: bool = True,
	exclude_metric_prefix: str | None = "eval",
	top_k: int = 20,
	benchmark_metric_sep: str = " / ",
) -> pd.DataFrame:
	corr_df = compute_kl_target_correlations(
		dirty_df=dirty_df,
		metric_columns=metric_columns,
		kl_target_columns=kl_target_columns,
		hue_column=hue_column,
		train_dataset_column=train_dataset_column,
		combine_train_datasets=combine_train_datasets,
		exclude_metric_prefix=exclude_metric_prefix,
		benchmark_metric_sep=benchmark_metric_sep,
	)

	if corr_df.empty:
		return corr_df

	return corr_df
