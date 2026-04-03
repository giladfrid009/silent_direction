from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from notebooks.utils.stats import as_series, safe_corr_xy


KNOWN_METRIC_KEYWORDS: tuple[str, ...] = (
	"acc",
	"f1",
	"match",
	"validity",
	"compliance",
	"pass",
)
PLOT_COLUMNS = [
	"benchmark",
	"metric",
	"benchmark_metric",
	"value_dirty",
	"value_clean",
	"value_plot",
	"is_known_range",
]


def is_known_range_metric(metric_name: str) -> bool:
	metric_name = str(metric_name).lower()
	return any(keyword in metric_name for keyword in KNOWN_METRIC_KEYWORDS)


def resolve_clean_row(clean_df: pd.DataFrame, model_name: str) -> pd.Series:
	model_rows = clean_df[clean_df["model_name"] == model_name]
	if model_rows.empty:
		raise ValueError(f"No clean row found for model_name='{model_name}'")
	if len(model_rows) > 1:
		raise ValueError(f"Expected one clean row for model_name='{model_name}', got {len(model_rows)}")
	return model_rows.iloc[0]


def as_float_or_none(value, column_name: str, source_name: str) -> float | None:
	if pd.isna(value):
		return None
	try:
		return float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"Non-numeric value in {source_name} column '{column_name}': {value!r}") from exc


def build_plot_frames(
	dirty_row: pd.Series,
	clean_row: pd.Series,
	benchmark_metric_sep: str = " / ",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	empty = pd.DataFrame(columns=PLOT_COLUMNS)

	dirty_metric_columns = [column_name for column_name in dirty_row.index if benchmark_metric_sep in str(column_name)]
	clean_metric_columns = set(column_name for column_name in clean_row.index if benchmark_metric_sep in str(column_name))

	common_rows = []
	dirty_only_rows = []

	for column_name in dirty_metric_columns:
		benchmark_name, metric_name = column_name.split(benchmark_metric_sep, 1)
		dirty_value = as_float_or_none(dirty_row[column_name], column_name, "dirty")

		if column_name in clean_metric_columns:
			clean_value = as_float_or_none(clean_row[column_name], column_name, "clean")
			if dirty_value is None or clean_value is None:
				continue

			known_range = is_known_range_metric(metric_name)
			value_plot = (dirty_value - clean_value) * 100.0 if known_range else (dirty_value - clean_value)
			common_rows.append(
				{
					"benchmark": benchmark_name,
					"metric": metric_name,
					"benchmark_metric": column_name,
					"value_dirty": dirty_value,
					"value_clean": clean_value,
					"value_plot": value_plot,
					"is_known_range": known_range,
				}
			)
		else:
			if dirty_value is None:
				continue

			dirty_only_rows.append(
				{
					"benchmark": benchmark_name,
					"metric": metric_name,
					"benchmark_metric": column_name,
					"value_dirty": dirty_value,
					"value_clean": pd.NA,
					"value_plot": dirty_value,
					"is_known_range": False,
				}
			)

	if common_rows:
		common_df = pd.DataFrame(common_rows).sort_values("benchmark_metric").reset_index(drop=True)
		known_df = common_df[common_df["is_known_range"]].copy().reset_index(drop=True)
		unknown_df = common_df[~common_df["is_known_range"]].copy().reset_index(drop=True)
	else:
		known_df = empty.copy()
		unknown_df = empty.copy()

	if dirty_only_rows:
		dirty_only_df = pd.DataFrame(dirty_only_rows).sort_values("benchmark_metric").reset_index(drop=True)
		dirty_only_df = dirty_only_df[PLOT_COLUMNS]
	else:
		dirty_only_df = empty.copy()

	return known_df, unknown_df, dirty_only_df


def plot_panel(
	ax: Axes,
	df: pd.DataFrame,
	ylabel: str,
	panel_title: str,
	label_mode: str,
	fontsize: int = 14,
) -> None:
	ax.set_title(panel_title, pad=12, fontsize=fontsize*1.5, fontweight="bold")
	# ax.set_xlabel("")
	ax.set_ylabel(ylabel, fontsize=fontsize*1.5, fontweight="bold")
	ax.tick_params(axis="x", rotation=90, labelsize=fontsize)
	ax.tick_params(axis="y", labelsize=fontsize)
	ax.grid(axis="y", alpha=0.3)

	colors = ["#d62728" if value < 0 else "#1f77b4" for value in df["value_plot"]]
	
	ax.bar(df["benchmark_metric"], df["value_plot"], color=colors)

	if "error_bar" in df.columns:
		err = pd.to_numeric(df["error_bar"], errors="coerce")
		# if label_mode == "percent":
		# 	err = err * 100
		
		# Plot error bars only for valid values
		for i, x_val in enumerate(df["benchmark_metric"]):
			e = err.iloc[i]
			if pd.notna(e) and e > 0:
				ax.errorbar(
					x=x_val,
					y=0,
					yerr=e,
					fmt="none",
					ecolor="black",
					capsize=3,
					alpha=0.6,
				)

	for idx, bar in enumerate(ax.patches):
		if idx >= len(df):
			break

		if not isinstance(bar, Rectangle):
			continue

		dirty_value = df.iloc[idx]["value_dirty"]
		clean_value = df.iloc[idx]["value_clean"]

		if label_mode == "percent":
			label = f"{(dirty_value - clean_value) * 100:.1f}"
		elif label_mode == "diff":
			label = f"{dirty_value:.3f} - {clean_value:.3f}"
		elif label_mode == "raw":
			label = f"{dirty_value:.3f}"
		else:
			raise ValueError(f"Unsupported label_mode: {label_mode}")

		x_center = bar.get_x() + bar.get_width() / 2
		y_anchor = max(bar.get_height(), 0)
		ax.annotate(
			label,
			xy=(x_center, y_anchor),
			xytext=(0, 2),
			textcoords="offset points",
			ha="center",
			va="bottom",
			fontsize=fontsize,
		)


def plot_run_comparisons(
	dirty_filtered_df: pd.DataFrame,
	clean_df: pd.DataFrame,
	benchmark_metric_sep: str = " / ",
	clean_statistics: pd.DataFrame | None = None,
	error_bar_col: str | None = None,
	save: bool = False,
	save_dir: str = "",
) -> None:
	run_ids = dirty_filtered_df["run_id"].tolist()
	print("Number of runs to plot:", len(run_ids))

	for _, dirty_row in dirty_filtered_df.iterrows():
		model_name = dirty_row["model_name"]
		train_dataset = dirty_row["train_dataset"]
		layer_name = dirty_row["layer_name"]

		clean_row = resolve_clean_row(clean_df, model_name=model_name)
		known_df, unknown_df, dirty_only_df = build_plot_frames(
			dirty_row,
			clean_row,
			benchmark_metric_sep=benchmark_metric_sep,
		)

		if clean_statistics is not None and error_bar_col is not None and not known_df.empty:
			stats_subset = clean_statistics[clean_statistics["model_name"] == model_name]
			if not stats_subset.empty and error_bar_col in stats_subset.columns:
				# Left merge so we don't drop rows from known_df missing in stats
				known_df = known_df.merge(
					stats_subset[["benchmark_metric", error_bar_col]],
					on="benchmark_metric",
					how="left"
				).rename(columns={error_bar_col: "error_bar"})

		panel_specs = []
		if not known_df.empty:
			panel_specs.append(
				{
					"df": known_df,
					"ylabel": "Difference %",
					"title": "(dirty - clean) x 100",
					"label_mode": "percent",
				}
			)
		# if not unknown_df.empty:
		# 	panel_specs.append(
		# 		{
		# 			"df": unknown_df,
		# 			"ylabel": "Difference (raw)",
		# 			"title": "Unknown-range metrics: (dirty - clean)",
		# 			"label_mode": "diff",
		# 		}
		# 	)
		# if not dirty_only_df.empty:
		# 	panel_specs.append(
		# 		{
		# 			"df": dirty_only_df,
		# 			"ylabel": "Value (raw)",
		# 			"title": "Dirty-only metrics (not present in clean)",
		# 			"label_mode": "raw",
		# 		}
		# 	)

		if not panel_specs:
			continue

		fig, axes = plt.subplots(len(panel_specs), 1, figsize=(18, 7 * len(panel_specs)), dpi=150)
		if len(panel_specs) == 1:
			axes = [axes]

		relative_l2 = (dirty_row.get("eval-oasst2 / proj_l2_rel", -1)*3 + dirty_row.get('eval-tulu-v3 / proj_l2_rel', -1)*10) / 13
		kl_divergence = (dirty_row.get("eval-oasst2 / kl_div", -1)*3 + dirty_row.get("eval-tulu-v3 / kl_div", -1)*10) / 13
		fig.suptitle(
			(
				f"Model: {model_name} | Layer: {layer_name} "
				f"| KL-Weight: {dirty_row.get('kl_value', 'N/A')} \n"
				# f"Run ID: {dirty_row.get('run_id', 'N/A')}"
				f"Rel L2: {relative_l2 * 100:.2f}% | KL Div: {kl_divergence:.4f}"
			),
			# y=1.04,
			fontsize=18,
			fontweight="bold",
		)
	
		for ax, spec in zip(axes, panel_specs):
			plot_panel(
				ax=ax,
				df=spec["df"],
				ylabel=spec["ylabel"],
				panel_title=spec["title"],
				label_mode=spec["label_mode"],
				fontsize=10,
			)

		plt.tight_layout()
		if save:
			import os
			if not save_dir:
				save_dir = "."
			os.makedirs(save_dir, exist_ok=True)
			kl_val = dirty_row.get("kl_value", "NA")
			safe_model_name = str(model_name).replace("/", "_")
			safe_train_dataset = str(train_dataset).replace("/", "_")
			safe_layer_name = str(layer_name).replace("/", "_")
			filename = f"{safe_model_name}:{safe_train_dataset}:{safe_layer_name}:kl_{kl_val}.png"
			filepath = os.path.join(save_dir, filename)
			plt.savefig(filepath, bbox_inches="tight")

		plt.show()


def extract_known_run_comparisons(
	dirty_filtered_df: pd.DataFrame,
	clean_df: pd.DataFrame,
	benchmark_metric_sep: str = " / ",
	clean_statistics: pd.DataFrame | None = None,
	error_bar_col: str | None = None,
) -> pd.DataFrame:
	"""
	Extracts and concatenates the known_df from all runs in dirty_filtered_df into a single DataFrame.
	This is useful for further analysis of the differences between dirty and clean runs.
	"""
	all_known_dfs = []

	for _, dirty_row in dirty_filtered_df.iterrows():
		model_name = dirty_row["model_name"]

		clean_row = resolve_clean_row(clean_df, model_name=model_name)
		known_df, _, _ = build_plot_frames(
			dirty_row,
			clean_row,
			benchmark_metric_sep=benchmark_metric_sep,
		)

		if clean_statistics is not None and error_bar_col is not None and not known_df.empty:
			stats_subset = clean_statistics[clean_statistics["model_name"] == model_name]
			if not stats_subset.empty and error_bar_col in stats_subset.columns:
				known_df = known_df.merge(
					stats_subset[["benchmark_metric", error_bar_col]],
					on="benchmark_metric",
					how="left"
				).rename(columns={error_bar_col: "error_bar"})

		if not known_df.empty:
			known_df = known_df.copy()
			known_df["run_id"] = dirty_row.get("run_id", pd.NA)
			known_df["model_name"] = model_name
			known_df["train_dataset"] = dirty_row.get("train_dataset", pd.NA)
			known_df["layer_name"] = dirty_row.get("layer_name", pd.NA)
			known_df["kl_value"] = dirty_row.get("kl_value", pd.NA)
			
			relative_l2 = (dirty_row.get("eval-oasst2 / proj_l2_rel", pd.NA)*3 + dirty_row.get('eval-tulu-v3 / proj_l2_rel', pd.NA)*10) / 13
			kl_divergence = (dirty_row.get("eval-oasst2 / kl_div", pd.NA)*3 + dirty_row.get("eval-tulu-v3 / kl_div", pd.NA)*10) / 13
			known_df["kl_div"] = kl_divergence
			known_df["proj_l2_rel"] = relative_l2
			
			all_known_dfs.append(known_df)

	if all_known_dfs:
		return pd.concat(all_known_dfs, ignore_index=True)
	
	# Return empty DataFrame with expected columns if nothing is found
	expected_cols = PLOT_COLUMNS + [
		"run_id", "model_name", "train_dataset", "layer_name", "kl_value", 
		"kl_div", "proj_l2_rel"
	]
	if error_bar_col is not None:
		expected_cols.append("error_bar")
	return pd.DataFrame(columns=expected_cols)


def plot_difference_statistics(
	stats_df: pd.DataFrame,
	separate_by_range: bool = True,
	top_k: int | None = None,
	figsize: tuple[float, float] = (14, 6),
	dirty_df: pd.DataFrame | None = None,
) -> None:
	if stats_df.empty:
		print("No statistics to plot")
		return

	title_suffix = ""
	if dirty_df is not None:
		relative_l2 = (dirty_df.get("eval-oasst2 / proj_l2_rel", pd.Series([-1])).mean() * 3 + dirty_df.get('eval-tulu-v3 / proj_l2_rel', pd.Series([-1])).mean() * 10) / 13
		kl_divergence = (dirty_df.get("eval-oasst2 / kl_div", pd.Series([-1])).mean() * 3 + dirty_df.get("eval-tulu-v3 / kl_div", pd.Series([-1])).mean() * 10) / 13
		title_suffix = f"\nAvg Rel L2: {relative_l2 * 100:.2f}% | Avg KL Div: {kl_divergence:.4f}"

	plot_data = stats_df.copy()
	if top_k is not None:
		if "abs_mean_diff" in plot_data.columns:
			plot_data = plot_data.nlargest(top_k, "abs_mean_diff")
		else:
			plot_data = plot_data.assign(abs_mean_diff=plot_data["mean_diff"].abs()).nlargest(top_k, "abs_mean_diff")

	if separate_by_range:
		for is_known_range in [True, False]:
			subset = plot_data[plot_data["is_known_range"] == is_known_range].copy()
			if subset.empty:
				continue

			range_label = " (% change)" if is_known_range else " (raw diff)"

			fig, ax = plt.subplots(figsize=figsize)

			subset = subset.sort_values("mean_diff")
			x_pos = range(len(subset))

			mean_vals = pd.to_numeric(subset["mean_diff"], errors="coerce")
			if "q25_diff" not in subset.columns or "q75_diff" not in subset.columns:
				raise ValueError("Expected quantile columns 'q25_diff' and 'q75_diff' for error bars")
			q25_vals = pd.to_numeric(subset["q25_diff"], errors="coerce")
			q75_vals = pd.to_numeric(subset["q75_diff"], errors="coerce")

			lower_err = (mean_vals - q25_vals).clip(lower=0).fillna(0.0)
			upper_err = (q75_vals - mean_vals).clip(lower=0).fillna(0.0)
			xerr = np.vstack([lower_err.to_numpy(), upper_err.to_numpy()])

			colors = ["#d62728" if value < 0 else "#1f77b4" for value in subset["mean_diff"]]
			ax.barh(
				x_pos,
				subset["mean_diff"],
				xerr=xerr,
				color=colors,
				capsize=3,
				alpha=0.8,
				error_kw={"elinewidth": 1},
			)

			ax.set_yticks(x_pos)
			ax.set_yticklabels(
				[f"{benchmark} / {metric}" for benchmark, metric in zip(subset["benchmark"], subset["metric"])],
				fontsize=9,
			)
			ax.set_xlabel(
				f"Mean Difference ({range_label.split('(')[1].rstrip(')')})",
				fontsize=10,
			)
			ax.set_title(
				f"{title_suffix}",
				fontsize=12,
				fontweight="bold",
			)
			ax.axvline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
			ax.grid(axis="x", alpha=0.3)

			for i, (mean_diff, count) in enumerate(zip(subset["mean_diff"], subset["count"])):
				x_offset = mean_diff + upper_err.iloc[i]
				ax.text(x_offset, i, f" n={count}", va="center", fontsize=8, alpha=0.7)

			plt.tight_layout()
			plt.show()
	else:
		fig, ax = plt.subplots(figsize=figsize)

		plot_data = plot_data.sort_values("mean_diff")
		x_pos = range(len(plot_data))

		mean_vals = pd.to_numeric(plot_data["mean_diff"], errors="coerce")
		if "q25_diff" not in plot_data.columns or "q75_diff" not in plot_data.columns:
			raise ValueError("Expected quantile columns 'q25_diff' and 'q75_diff' for error bars")
		q25_vals = pd.to_numeric(plot_data["q25_diff"], errors="coerce")
		q75_vals = pd.to_numeric(plot_data["q75_diff"], errors="coerce")

		lower_err = (mean_vals - q25_vals).clip(lower=0).fillna(0.0)
		upper_err = (q75_vals - mean_vals).clip(lower=0).fillna(0.0)
		xerr = np.vstack([lower_err.to_numpy(), upper_err.to_numpy()])

		colors = ["#d62728" if value < 0 else "#1f77b4" for value in plot_data["mean_diff"]]
		ax.barh(
			x_pos,
			plot_data["mean_diff"],
			xerr=xerr,
			color=colors,
			capsize=3,
			alpha=0.8,
			error_kw={"elinewidth": 1},
		)

		ax.set_yticks(x_pos)
		ax.set_yticklabels(
			[f"{benchmark} / {metric}" for benchmark, metric in zip(plot_data["benchmark"], plot_data["metric"])],
			fontsize=9,
		)
		ax.set_xlabel("Mean Difference", fontsize=10)
		ax.set_title(f"Clean-Dirty Differences by Metric{title_suffix}", fontsize=12, fontweight="bold")
		ax.axvline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
		ax.grid(axis="x", alpha=0.3)

		plt.tight_layout()
		plt.show()


def plot_grouped_difference_statistics(
	stats_df: pd.DataFrame,
	group_columns: list[str] | None = None,
	group_by: str | None = None,
	separate_by_range: bool = True,
	top_k: int | None = None,
	figsize_per_group: tuple[float, float] = (14, 6),
	hue_column: str | None = None,
) -> None:
	if stats_df.empty:
		print("No statistics to plot")
		return

	if group_columns is None:
		standard_cols = {
			"benchmark",
			"metric",
			"is_known_range",
			"mean_diff",
			"std_diff",
			"median_diff",
			"min_diff",
			"max_diff",
			"q25_diff",
			"q75_diff",
			"count",
			"diff_unit",
			"model_name",
			"abs_mean_diff",
		}
		group_columns = [column_name for column_name in stats_df.columns if column_name not in standard_cols]

	if group_by is None and group_columns:
		group_by = group_columns[0]

	if group_by is None:
		plot_difference_statistics(
			stats_df,
			separate_by_range=separate_by_range,
			top_k=top_k,
			figsize=figsize_per_group,
		)
		return

	unique_groups = stats_df[group_by].dropna().unique()
	n_groups = len(unique_groups)

	print(f"Plotting statistics for {n_groups} unique values of '{group_by}':")
	print(f"  {sorted(unique_groups)}")
	print()

	range_types = [True, False] if separate_by_range else [None]

	for is_known_range in range_types:
		range_label = (
			"Known-range metrics (% change)"
			if is_known_range is True
			else "Unknown-range metrics (raw diff)"
			if is_known_range is False
			else "All metrics"
		)

		fig, axes = plt.subplots(
			nrows=1,
			ncols=n_groups,
			figsize=(figsize_per_group[0] / 1.5 * n_groups, figsize_per_group[1]),
			squeeze=False,
			dpi=150,
		)
		axes = axes.flatten()

		for ax_idx, group_val in enumerate(sorted(unique_groups)):
			subset = stats_df[stats_df[group_by] == group_val].copy()

			if is_known_range is not None:
				subset = subset[subset["is_known_range"] == is_known_range].copy()

			if subset.empty:
				ax = axes[ax_idx]
				ax.text(
					0.5,
					0.5,
					f"No data for {group_by}={group_val}",
					ha="center",
					va="center",
					transform=ax.transAxes,
				)
				ax.set_axis_off()
				continue

			if top_k is not None:
				subset = subset.assign(_abs_mean_diff=subset["mean_diff"].abs())
				subset = subset.nlargest(top_k, "_abs_mean_diff").drop(columns=["_abs_mean_diff"])

			subset = subset.sort_values("mean_diff")
			ax = axes[ax_idx]

			# Build asymmetric error bars from quantiles around the mean.
			mean_vals = pd.to_numeric(subset["mean_diff"], errors="coerce")
			if "q25_diff" not in subset.columns or "q75_diff" not in subset.columns:
				raise ValueError("Expected quantile columns 'q25_diff' and 'q75_diff' for grouped error bars")
			q25_vals = pd.to_numeric(subset["q25_diff"], errors="coerce")
			q75_vals = pd.to_numeric(subset["q75_diff"], errors="coerce")

			lower_err = (mean_vals - q25_vals).clip(lower=0).fillna(0.0)
			upper_err = (q75_vals - mean_vals).clip(lower=0).fillna(0.0)
			xerr = np.vstack([lower_err.to_numpy(), upper_err.to_numpy()])

			if hue_column and hue_column in subset.columns:
				unique_hues = subset[hue_column].unique()
				palette = sns.color_palette("tab10", n_colors=len(unique_hues))
				color_map = {hue_value: palette[idx] for idx, hue_value in enumerate(unique_hues)}
				colors = [color_map.get(hue_value, "#1f77b4") for hue_value in subset[hue_column]]
			else:
				colors = ["#d62728" if value < 0 else "#1f77b4" for value in subset["mean_diff"]]

			x_pos = range(len(subset))
			ax.barh(
				x_pos,
				subset["mean_diff"],
				xerr=xerr,
				color=colors,
				capsize=3,
				alpha=0.8,
				error_kw={"elinewidth": 1},
			)

			ax.set_yticks(x_pos)
			ax.set_yticklabels(
				[f"{benchmark} / {metric}" for benchmark, metric in zip(subset["benchmark"], subset["metric"])],
				fontsize=8,
			)
			ax.set_xlabel("Mean Difference", fontsize=9)
			ax.set_title(f"{group_by}={group_val}\n(n={subset['count'].sum()})", fontsize=10, fontweight="bold")
			ax.axvline(0, color="black", linewidth=1, linestyle="-", alpha=0.5)
			ax.grid(axis="x", alpha=0.3)

		for idx in range(n_groups, len(axes)):
			axes[idx].set_axis_off()

		fig.suptitle(
			f"Clean-Dirty Differences by {group_by}: {range_label}",
			fontsize=12,
			fontweight="bold",
			y=1.02,
		)
		plt.tight_layout()
		plt.show()


def annotate_bar_values(ax: Axes, fontsize: int = 7) -> None:
	"""Add small numeric labels above bars in a seaborn/matplotlib bar plot."""
	for container in ax.containers:
		if not isinstance(container, BarContainer):
			continue

		values = getattr(container, "datavalues", None)
		if values is None:
			continue

		labels = []
		for value in values:
			if pd.isna(value):
				labels.append("")
			elif abs(value) >= 1:
				labels.append(f"{value:.2f}")
			else:
				labels.append(f"{value:.3f}")

		ax.bar_label(container, labels=labels, padding=2, fontsize=fontsize)


def add_min_max_yticks(ax: Axes, values: pd.Series) -> None:
	"""Set y-axis ticks to min and max plotted values only."""
	finite_values = pd.to_numeric(values, errors="coerce").dropna()
	if finite_values.empty:
		return

	min_val = float(finite_values.min())
	max_val = float(finite_values.max())

	if min_val == max_val:
		ax.set_yticks([min_val])
	else:
		ax.set_yticks([min_val, max_val])


def plot_kl_correlation_barplot(
	corr_df: pd.DataFrame,
	corr_column: str = "avg_group_corr",
	train_scope: str | None = "ALL_TRAIN_DATASETS",
	separate_by_kl_target: bool = True,
	figsize_per_panel: tuple[float, float] = (13, 5),
) -> None:
	required_columns = {"train_scope", "kl_target", "metric", "avg_group_corr", "pooled_corr"}
	missing = required_columns.difference(corr_df.columns)
	if missing:
		raise ValueError(f"Missing required columns for plotting: {sorted(missing)}")

	if corr_column not in {"avg_group_corr", "pooled_corr"}:
		raise ValueError("corr_column must be one of: 'avg_group_corr', 'pooled_corr'")

	plot_df = corr_df.copy()
	if train_scope is not None:
		plot_df = plot_df[plot_df["train_scope"] == train_scope].copy()

	if plot_df.empty:
		raise ValueError("No rows to plot after train_scope filtering")

	plot_df["metric"] = plot_df["metric"].astype(str)

	if separate_by_kl_target:
		kl_targets = plot_df["kl_target"].dropna().astype(str).unique().tolist()
		if not kl_targets:
			raise ValueError("No KL targets found to plot")

		fig, axes = plt.subplots(
			nrows=len(kl_targets),
			ncols=1,
			figsize=(figsize_per_panel[0], figsize_per_panel[1] * len(kl_targets)),
			squeeze=False,
		)
		axes = axes.flatten()

		for ax, kl_target in zip(axes, kl_targets):
			subset = plot_df[plot_df["kl_target"].astype(str) == kl_target].copy()
			subset = subset.sort_values(corr_column, ascending=False)

			sns.barplot(
				data=subset,
				x="metric",
				y=corr_column,
				hue="metric",
				dodge=False,
				palette="vlag",
				legend=False,
				ax=ax,
			)
			add_min_max_yticks(ax, subset[corr_column])
			ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
			ax.set_title(f"{kl_target} | All correlations ({corr_column})")
			ax.set_xlabel("metric")
			ax.set_ylabel(corr_column)
			ax.grid(axis="y", alpha=0.25)
			ax.tick_params(axis="x", rotation=90)

		plt.tight_layout()
		plt.show()
		return

	subset = plot_df.sort_values(corr_column, ascending=False).copy()
	fig, ax = plt.subplots(figsize=(max(figsize_per_panel[0], 0.25 * len(subset)), figsize_per_panel[1]))
	sns.barplot(
		data=subset,
		x="metric",
		y=corr_column,
		hue="kl_target",
		dodge=True,
		palette="tab10",
		ax=ax,
	)
	add_min_max_yticks(ax, subset[corr_column])
	ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
	ax.set_title(f"All correlations ({corr_column})")
	ax.set_xlabel("metric")
	ax.set_ylabel(corr_column)
	ax.grid(axis="y", alpha=0.25)
	ax.tick_params(axis="x", rotation=90)
	plt.tight_layout()
	ax.legend(bbox_to_anchor=(1, 1.5))
	plt.show()


def plot_metrics_correlation(
	dirty_df: pd.DataFrame,
	metric_columns: list[str],
	against_columns: list[str],
	hue_column: str = "model_name",
) -> None:
	if hue_column not in dirty_df.columns:
		raise ValueError(f"Missing hue column: {hue_column}")

	missing_against = [col for col in against_columns if col not in dirty_df.columns]
	if missing_against:
		raise ValueError(f"Missing against columns: {missing_against}")

	valid_metric_columns = [col for col in metric_columns if col in dirty_df.columns]
	if not valid_metric_columns:
		raise ValueError("No valid metric columns were found in dataframe")

	for metric in valid_metric_columns:
		subplot_count = len(against_columns)
		fig, axes = plt.subplots(1, subplot_count, figsize=(7 * subplot_count, 5), squeeze=False)
		axes = axes.flatten()

		for ax, against_col in zip(axes, against_columns):
			if metric == against_col:
				ax.text(0.5, 0.5, f"Skipping self-correlation: {metric}", ha="center", va="center")
				ax.set_axis_off()
				continue

			grouped = dirty_df.groupby(hue_column, dropna=False)
			group_corrs: dict[str, float] = {}

			unique_groups = list(grouped.groups.keys())
			palette = sns.color_palette("tab10", n_colors=max(1, len(unique_groups)))
			color_by_group = {
				group_name: palette[idx % len(palette)] for idx, group_name in enumerate(unique_groups)
			}

			for group_name, group_df in grouped:
				x = as_series(group_df[against_col])
				y = as_series(group_df[metric])

				valid = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
				if valid.empty:
					group_corrs[str(group_name)] = np.nan
					continue

				color = color_by_group[group_name]
				ax.scatter(valid["x"], valid["y"], s=28, alpha=0.75, color=color)
				group_corrs[str(group_name)] = safe_corr_xy(valid["x"], valid["y"])

			valid_corrs = [corr for corr in group_corrs.values() if pd.notna(corr)]
			avg_corr = float(np.mean(valid_corrs)) if valid_corrs else np.nan

			legend_handles = []
			for group_name in unique_groups:
				corr_val = group_corrs.get(str(group_name), np.nan)
				if pd.notna(corr_val):
					label = f"{group_name} (corr={corr_val:.2f})"
				else:
					label = f"{group_name} (corr=NA)"

				handle = Line2D(
					[0],
					[0],
					marker="o",
					color=color_by_group[group_name],
					markerfacecolor=color_by_group[group_name],
					markersize=6,
					linewidth=2,
					label=label,
				)
				legend_handles.append(handle)

			avg_label = f"Avg corr={avg_corr:.2f}" if pd.notna(avg_corr) else "Avg corr=NA"
			ax.set_title(f"{metric} vs {against_col}\n{avg_label}")
			ax.set_xlabel(against_col)
			ax.set_ylabel(metric)
			ax.grid(alpha=0.3)
			ax.legend(handles=legend_handles, title=hue_column, fontsize=8, title_fontsize=9)

		fig.suptitle(f"Metric: {metric}", y=1.02, fontsize=12, fontweight="bold")
		plt.tight_layout()
		plt.show()
