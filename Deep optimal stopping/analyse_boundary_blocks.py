from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLOCK_EDGES = np.linspace(0.0, 1.0, 11)
BLOCK_LABELS = [f"{BLOCK_EDGES[i]:.1f}-{BLOCK_EDGES[i + 1]:.1f}" for i in range(10)]
BLOCK_MIDPOINTS = (BLOCK_EDGES[:-1] + BLOCK_EDGES[1:]) / 2.0


def parse_configuration(path: Path) -> tuple[int, int]:
    match = re.search(r"N_(\d+)_M_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse N and M from {path.name}")
    return int(match.group(1)), int(match.group(2))


def root_mean_square(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def summarise_one(path: Path) -> tuple[pd.DataFrame, list[dict[str, float | int | str]]]:
    n_value, m_value = parse_configuration(path)
    study = "N sensitivity" if m_value == 8192 and n_value in {50, 100, 150} else "M sensitivity"

    data = pd.read_csv(path)
    required = {
        "time",
        "boundary_error",
        "absolute_boundary_error",
        "squared_boundary_error",
        "number_of_finite_seed_boundaries",
        "sample_sd_learned_boundary",
        "included_in_boundary_metrics",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    data = data.loc[(data["time"] > 0.0) & (data["time"] < 1.0)].copy()
    data["block"] = pd.cut(
        data["time"],
        bins=BLOCK_EDGES,
        labels=BLOCK_LABELS,
        right=True,
        include_lowest=False,
    )
    data["seed_coverage"] = data["number_of_finite_seed_boundaries"] / 5.0
    data["all_five_available"] = data["number_of_finite_seed_boundaries"] == 5
    data["current_valid"] = (
        data["included_in_boundary_metrics"].astype(bool)
        & data["boundary_error"].notna()
    )

    rows: list[dict[str, float | int | str]] = []
    total_current_sse = float(
        data.loc[data["current_valid"], "squared_boundary_error"].sum()
    )

    for idx, label in enumerate(BLOCK_LABELS):
        block = data.loc[data["block"] == label].copy()
        current = block.loc[block["current_valid"]]
        strict = block.loc[block["current_valid"] & block["all_five_available"]]
        sd_eligible = block.loc[block["number_of_finite_seed_boundaries"] >= 2]

        current_sse = float(current["squared_boundary_error"].sum())
        rows.append(
            {
                "study": study,
                "N": n_value,
                "M": m_value,
                "block": label,
                "block_start": float(BLOCK_EDGES[idx]),
                "block_end": float(BLOCK_EDGES[idx + 1]),
                "block_midpoint": float(BLOCK_MIDPOINTS[idx]),
                "number_of_exercise_dates": int(len(block)),
                "number_of_dates_current_metric": int(len(current)),
                "number_of_dates_all_five_available": int(len(strict)),
                "mean_seed_coverage": safe_mean(block["seed_coverage"]),
                "all_five_date_share": safe_mean(block["all_five_available"].astype(float)),
                "current_mean_error": safe_mean(current["boundary_error"]),
                "current_mae": safe_mean(current["absolute_boundary_error"]),
                "current_rmse": root_mean_square(current["boundary_error"]),
                "current_maximum_absolute_error": (
                    float(current["absolute_boundary_error"].max()) if len(current) else float("nan")
                ),
                "current_squared_error_share": (
                    current_sse / total_current_sse if total_current_sse > 0 else float("nan")
                ),
                "all_five_conditional_mae": safe_mean(strict["absolute_boundary_error"]),
                "all_five_conditional_rmse": root_mean_square(strict["boundary_error"]),
                "mean_cross_seed_sd_when_at_least_two_available": safe_mean(
                    sd_eligible["sample_sd_learned_boundary"]
                ),
            }
        )

    current_all = data.loc[data["current_valid"]]
    strict_all = data.loc[data["current_valid"] & data["all_five_available"]]
    early = current_all.loc[current_all["time"] <= 0.2]
    later = current_all.loc[current_all["time"] > 0.2]
    early_all_dates = data.loc[data["time"] <= 0.2]
    overall_sse = float(current_all["squared_boundary_error"].sum())
    max_row = current_all.loc[current_all["absolute_boundary_error"].idxmax()]

    summary: dict[str, float | int | str] = {
        "study": study,
        "N": n_value,
        "M": m_value,
        "overall_current_number_of_dates": int(len(current_all)),
        "overall_current_mae": safe_mean(current_all["absolute_boundary_error"]),
        "overall_current_rmse": root_mean_square(current_all["boundary_error"]),
        "early_current_mae_0_to_0_2": safe_mean(early["absolute_boundary_error"]),
        "later_current_mae_0_2_to_1": safe_mean(later["absolute_boundary_error"]),
        "early_to_later_mae_ratio": (
            safe_mean(early["absolute_boundary_error"])
            / safe_mean(later["absolute_boundary_error"])
        ),
        "first_block_share_of_total_squared_error": (
            float(early["squared_boundary_error"].sum()) / overall_sse
            if overall_sse > 0
            else float("nan")
        ),
        "maximum_absolute_error": float(max_row["absolute_boundary_error"]),
        "time_of_maximum_absolute_error": float(max_row["time"]),
        "finite_seeds_at_maximum_error": int(max_row["number_of_finite_seed_boundaries"]),
        "first_block_mean_seed_coverage": safe_mean(early_all_dates["seed_coverage"]),
        "first_block_all_five_date_share": safe_mean(
            early_all_dates["all_five_available"].astype(float)
        ),
        "overall_all_five_date_share": safe_mean(data["all_five_available"].astype(float)),
        "overall_all_five_conditional_number_of_dates": int(len(strict_all)),
        "overall_all_five_conditional_mae": safe_mean(strict_all["absolute_boundary_error"]),
        "overall_all_five_conditional_rmse": root_mean_square(strict_all["boundary_error"]),
    }
    blockwise = pd.DataFrame(rows)
    summaries = [summary]

    # N=100, M=8192 is the shared configuration in both sensitivity panels.
    if n_value == 100 and m_value == 8192:
        duplicate_blocks = blockwise.copy()
        duplicate_blocks["study"] = "M sensitivity"
        blockwise = pd.concat([blockwise, duplicate_blocks], ignore_index=True)
        duplicate_summary = summary.copy()
        duplicate_summary["study"] = "M sensitivity"
        summaries.append(duplicate_summary)

    return blockwise, summaries


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", labelsize=9)


def plot_error_curves(
    axis: plt.Axes,
    blockwise: pd.DataFrame,
    study: str,
    metric: str,
    legend_parameter: str,
    colours: list[str],
    title: str,
    y_axis_label: str,
) -> None:
    panel = blockwise.loc[blockwise["study"] == study]
    parameter = "N" if legend_parameter == "N" else "M"
    values = sorted(panel[parameter].unique())
    for colour, value in zip(colours, values):
        curve = panel.loc[panel[parameter] == value].sort_values("block_midpoint")
        axis.plot(
            curve["block_midpoint"],
            curve[metric],
            marker="o",
            markersize=4.2,
            linewidth=1.6,
            color=colour,
            label=f"{parameter} = {value:,}",
        )
    axis.set_title(title, fontsize=11.5, pad=10)
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks(np.linspace(0.0, 1.0, 11))
    axis.set_xlabel("Exercise time $t$", fontsize=10.5)
    axis.set_ylabel(y_axis_label, fontsize=10.5)
    axis.legend(frameon=False, fontsize=9)
    style_axis(axis)


def make_error_figures(blockwise: pd.DataFrame, output_dir: Path) -> None:
    colours_n = ["#0072B2", "#D55E00", "#009E73"]
    colours_m = ["#CC79A7", "#56B4E9", "#E69F00", "#009E73"]

    metric_specs = [
        (
            "current_mae",
            "mae",
            "Mean Absolute Error of the Learned Exercise Boundary over Time",
            "Mean absolute error (MAE)",
        ),
        (
            "current_rmse",
            "rmse",
            "Root Mean Squared Error of the Learned Exercise Boundary over Time",
            "Root mean squared error (RMSE)",
        ),
    ]
    panel_specs = [
        ("N sensitivity", "N", colours_n),
        ("M sensitivity", "M", colours_m),
    ]
    for metric, metric_tag, title, y_axis_label in metric_specs:
        for study, parameter, colours in panel_specs:
            filename = f"learned_exercise_boundary_{metric_tag}_{parameter}"
            fig, axis = plt.subplots(figsize=(6.6, 4.25))
            plot_error_curves(
                axis,
                blockwise,
                study,
                metric,
                parameter,
                colours,
                title,
                y_axis_label,
            )
            panel_max = blockwise.loc[blockwise["study"] == study, metric].max()
            axis.set_ylim(0.0, panel_max * 1.08)
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                fig.savefig(output_dir / f"{filename}.{suffix}", dpi=300)
            plt.close(fig)


def make_corresponding_tables(blockwise: pd.DataFrame, output_dir: Path) -> None:
    def display_interval(row: pd.Series, is_last: bool) -> str:
        left = f"{row['block_start']:.1f}"
        right = f"{row['block_end']:.1f}"
        return f"({left}, {right}{')' if is_last else ']'}"

    metric_specs = [
        ("current_mae", "mae", "Mean absolute error", "MAE"),
        ("current_rmse", "rmse", "Root mean squared error", "RMSE"),
    ]
    for metric, metric_tag, caption_metric, label_metric in metric_specs:
        n_data = (
            blockwise.loc[blockwise["study"] == "N sensitivity"]
            .pivot(index=["block_start", "block_end", "block_midpoint", "block"], columns="N", values=metric)
            .reset_index()
            .sort_values("block_start")
        )
        m_data = (
            blockwise.loc[blockwise["study"] == "M sensitivity"]
            .pivot(index=["block_start", "block_end", "block_midpoint", "block"], columns="M", values=metric)
            .reset_index()
            .sort_values("block_start")
        )

        n_table = pd.DataFrame({
            "Exercise-time interval": [display_interval(row, i == len(n_data) - 1) for i, (_, row) in enumerate(n_data.iterrows())],
            "N = 50": n_data[50],
            "N = 100": n_data[100],
            "N = 150": n_data[150],
        })
        m_table = pd.DataFrame({
            "Exercise-time interval": [display_interval(row, i == len(m_data) - 1) for i, (_, row) in enumerate(m_data.iterrows())],
            "M = 2,048": m_data[2048],
            "M = 4,096": m_data[4096],
            "M = 8,192": m_data[8192],
            "M = 16,384": m_data[16384],
        })

        n_table.to_csv(output_dir / f"learned_exercise_boundary_{metric_tag}_table_N.csv", index=False)
        m_table.to_csv(output_dir / f"learned_exercise_boundary_{metric_tag}_table_M.csv", index=False)

        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            rf"Exercise-time interval & $N=50$ & $N=100$ & $N=150$ \\",
            r"\midrule",
        ]
        for _, row in n_table.iterrows():
            lines.append(
                f"{row['Exercise-time interval']} & {row['N = 50']:.4f} & {row['N = 100']:.4f} & {row['N = 150']:.4f} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption_metric} of the learned exercise boundary across exercise-time intervals for different values of $N$ ($M=8192$).}}",
            rf"\label{{tab:learned_boundary_{metric_tag}_N}}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            rf"Exercise-time interval & $M=2048$ & $M=4096$ & $M=8192$ & $M=16384$ \\",
            r"\midrule",
        ])
        for _, row in m_table.iterrows():
            lines.append(
                f"{row['Exercise-time interval']} & {row['M = 2,048']:.4f} & {row['M = 4,096']:.4f} & {row['M = 8,192']:.4f} & {row['M = 16,384']:.4f} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption_metric} of the learned exercise boundary across exercise-time intervals for different values of $M$ ($N=100$).}}",
            rf"\label{{tab:learned_boundary_{metric_tag}_M}}",
            r"\end{table}",
        ])
        (output_dir / f"learned_exercise_boundary_{metric_tag}_tables.tex").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(args.input_dir.glob("boundary_error_by_date_N_*_M_*.csv"))
    if not source_files:
        raise FileNotFoundError(f"No boundary_error_by_date CSV files found in {args.input_dir}")

    all_blocks: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    for path in source_files:
        blockwise, configuration_summaries = summarise_one(path)
        all_blocks.append(blockwise)
        summaries.extend(configuration_summaries)

    blockwise_all = pd.concat(all_blocks, ignore_index=True)
    summary_all = pd.DataFrame(summaries)
    study_order = pd.CategoricalDtype(["N sensitivity", "M sensitivity"], ordered=True)
    blockwise_all["study"] = blockwise_all["study"].astype(study_order)
    summary_all["study"] = summary_all["study"].astype(study_order)
    blockwise_all = blockwise_all.sort_values(["study", "N", "M", "block_start"])
    summary_all = summary_all.sort_values(["study", "N", "M"])

    blockwise_all.to_csv(args.output_dir / "blockwise_boundary_diagnostics.csv", index=False)
    summary_all.to_csv(args.output_dir / "boundary_summary_reanalysis.csv", index=False)
    make_error_figures(blockwise_all, args.output_dir)
    make_corresponding_tables(blockwise_all, args.output_dir)

    print(f"Processed {len(source_files)} configurations")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
