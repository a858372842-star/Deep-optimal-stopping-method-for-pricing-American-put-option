from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing a required package. In the VS Code terminal run:\n"
        "  python -m pip install numpy pandas matplotlib openpyxl\n"
        "Then run this script again."
    ) from exc


DEFAULT_RESULTS_FOLDER = "outputs_4_1_single_case_sensitivity_five_seed"
OUTPUT_FOLDER = "postprocessing_4_3_local"
N_VALUES: Tuple[int, ...] = (50, 100, 150)
M_VALUES: Tuple[int, ...] = (2048, 4096, 8192, 16384)
BASELINE_M = 8192
K = 40.0
T = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(DEFAULT_RESULTS_FOLDER),
        help=(
            "Completed five-seed results folder. The default searches beside "
            "this script and in the current working directory."
        ),
    )
    return parser.parse_args()


def has_expected_structure(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "exercise_grid_sensitivity").is_dir()
        and (path / "training_path_sensitivity").is_dir()
        and (path / "shared_continuous_time_benchmark").is_dir()
    )


def resolve_results_root(requested: Path) -> Path:
    requested = requested.expanduser()
    script_parent = Path(__file__).resolve().parent
    candidates: List[Path] = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.extend(
            [
                Path.cwd() / requested,
                script_parent / requested,
                script_parent.parent / requested,
            ]
        )
    candidates.extend(
        [
            Path.cwd() / DEFAULT_RESULTS_FOLDER,
            script_parent / DEFAULT_RESULTS_FOLDER,
            script_parent.parent / DEFAULT_RESULTS_FOLDER,
        ]
    )

    checked: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        checked.append(key)
        if has_expected_structure(resolved):
            print("Found completed results:", resolved)
            return resolved

    raise FileNotFoundError(
        "Could not find the completed results folder. Checked:\n  - "
        + "\n  - ".join(checked)
        + "\nMove this script beside the results folder, or run:\n"
        + '  python local_boundary_learning_postprocess.py --results-root "C:\\path\\to\\'
        + DEFAULT_RESULTS_FOLDER
        + '"'
    )


def n_config_root(results_root: Path, N: int) -> Path:
    return results_root / "exercise_grid_sensitivity" / f"N_{N}_B_8192"


def m_config_root(results_root: Path, M: int) -> Path:
    if M == BASELINE_M:
        return n_config_root(results_root, 100)
    return results_root / "training_path_sensitivity" / "N_100" / f"B_{M}"


def seed_number(seed_dir: Path) -> int:
    return int(seed_dir.name.replace("seed_", ""))


def seed_dirs(config_root: Path, label: str) -> List[Path]:
    folders = sorted(
        (path for path in config_root.glob("seed_*") if path.is_dir()),
        key=seed_number,
    )
    if len(folders) != 5:
        raise RuntimeError(
            f"{label}: expected five seed folders under {config_root}, "
            f"but found {len(folders)}."
        )
    return folders


def find_one(folder: Path, filename: str) -> Path:
    matches = sorted(folder.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {filename} under {folder}, found {len(matches)}."
        )
    return matches[0]


def seed_learning_curve(seed_dir: Path) -> pd.DataFrame:
    path = find_one(seed_dir, "training_log.csv")
    frame = pd.read_csv(path)
    required = [
        "exercise_date_index",
        "step_within_date",
        "soft_objective",
        "hard_policy_batch_value",
        "mean_soft_stop_probability",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required)
    return (
        frame.groupby("step_within_date", as_index=False)
        .agg(
            soft_objective=("soft_objective", "mean"),
            hard_value=("hard_policy_batch_value", "mean"),
            stop_probability=("mean_soft_stop_probability", "mean"),
        )
        .sort_values("step_within_date")
    )


def aggregate_learning(folders: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for folder in folders:
        frame = seed_learning_curve(folder)
        frame["seed"] = seed_number(folder)
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    return (
        long.groupby("step_within_date", as_index=False)
        .agg(
            soft_mean=("soft_objective", "mean"),
            soft_sd=("soft_objective", "std"),
            hard_mean=("hard_value", "mean"),
            hard_sd=("hard_value", "std"),
            stop_mean=("stop_probability", "mean"),
            stop_sd=("stop_probability", "std"),
        )
        .sort_values("step_within_date")
    )


def plot_learning(
    summaries: Dict[str, pd.DataFrame],
    mean_column: str,
    sd_column: str,
    ylabel: str,
    stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for label, frame in summaries.items():
        x = frame["step_within_date"].to_numpy(float)
        mean = frame[mean_column].to_numpy(float)
        sd = frame[sd_column].fillna(0.0).to_numpy(float)
        ax.plot(x, mean, linewidth=1.7, label=label)
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.14)
    ax.set_xlabel("training step within each exercise date")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def create_learning_outputs(results_root: Path, output_root: Path) -> None:
    folder = output_root / "learning_curves"
    folder.mkdir(parents=True, exist_ok=True)

    n_summaries: Dict[str, pd.DataFrame] = {}
    for N in N_VALUES:
        summary = aggregate_learning(seed_dirs(n_config_root(results_root, N), f"N={N}"))
        summary.to_csv(folder / f"N_{N}_B_8192_five_seed_learning_curve.csv", index=False)
        n_summaries[f"N={N}"] = summary
    plot_learning(
        n_summaries,
        "soft_mean",
        "soft_sd",
        "mean training objective",
        folder / "learning_curve_compare_N_soft_objective",
    )

    m_summaries: Dict[str, pd.DataFrame] = {}
    for M in M_VALUES:
        summary = aggregate_learning(seed_dirs(m_config_root(results_root, M), f"M={M}"))
        summary.to_csv(folder / f"N_100_B_{M}_five_seed_learning_curve.csv", index=False)
        m_summaries[f"M={M}"] = summary
    plot_learning(
        m_summaries,
        "soft_mean",
        "soft_sd",
        "mean training objective",
        folder / "learning_curve_compare_batch_soft_objective",
    )
    print("Learning curves complete.")


def load_continuous_boundary(results_root: Path) -> pd.DataFrame:
    folder = results_root / "shared_continuous_time_benchmark"
    candidates = list(folder.rglob("boundary_M*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No Peskir boundary CSV found under {folder}.")

    def resolution(path: Path) -> int:
        try:
            return int(path.stem.split("boundary_M", 1)[1])
        except (IndexError, ValueError):
            return -1

    path = max(candidates, key=resolution)
    frame = pd.read_csv(path)
    if not {"time", "boundary"}.issubset(frame.columns):
        raise ValueError(f"Unexpected columns in {path}.")
    print("Using continuous boundary:", path)
    return frame


def boundary_metrics(
    frame: pd.DataFrame,
    *,
    study: str,
    N: int,
    M: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    required = [
        "time_index",
        "time",
        "theoretical_continuous_boundary",
        "mean_learned_boundary",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Boundary CSV is missing columns: {missing}")

    detail = frame.copy()
    for column in required:
        detail[column] = pd.to_numeric(detail[column], errors="coerce")
    interior = (detail["time"] > 1.0e-12) & (detail["time"] < T - 1.0e-12)
    finite = np.isfinite(detail["mean_learned_boundary"]) & np.isfinite(
        detail["theoretical_continuous_boundary"]
    )
    used = interior & finite
    error = (
        detail["mean_learned_boundary"]
        - detail["theoretical_continuous_boundary"]
    )
    detail["included_in_boundary_metrics"] = used
    detail["boundary_error"] = error
    detail["absolute_boundary_error"] = np.abs(error)
    detail["squared_boundary_error"] = error**2
    values = error.loc[used].to_numpy(float)
    if values.size == 0:
        raise ValueError(f"No finite interior learned boundaries for N={N}, M={M}.")

    summary: Dict[str, object] = {
        "study": study,
        "N": N,
        "M": M,
        "number_of_interior_exercise_dates": int(interior.sum()),
        "number_of_dates_used": int(values.size),
        "mean_error": float(values.mean()),
        "mae": float(np.abs(values).mean()),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "maximum_absolute_error": float(np.abs(values).max()),
        "fraction_learned_boundary_above_continuous": float(np.mean(values > 0.0)),
    }
    if "sample_sd_learned_boundary" in detail.columns:
        seed_sd = pd.to_numeric(
            detail.loc[used, "sample_sd_learned_boundary"], errors="coerce"
        ).dropna()
        summary["mean_across_seed_boundary_sd"] = (
            float(seed_sd.mean()) if not seed_sd.empty else math.nan
        )
    return summary, detail


def replot_boundary(
    config_root: Path,
    *,
    N: int,
    M: int,
    study: str,
    continuous: pd.DataFrame,
    output_folder: Path,
) -> Dict[str, object]:
    aggregate = config_root / "aggregate_5_seeds"
    average_path = aggregate / "average_boundary_comparison.csv"
    individual_path = aggregate / "individual_seed_boundaries.csv"
    if not average_path.exists():
        raise FileNotFoundError(average_path)
    if not individual_path.exists():
        raise FileNotFoundError(individual_path)

    average = pd.read_csv(average_path)
    individual = pd.read_csv(individual_path)
    summary, detail = boundary_metrics(average, study=study, N=N, M=M)
    detail.to_csv(output_folder / f"boundary_error_by_date_N_{N}_M_{M}.csv", index=False)

    time = pd.to_numeric(average["time"], errors="coerce").to_numpy(float)
    learned = pd.to_numeric(average["mean_learned_boundary"], errors="coerce").to_numpy(float)
    sd = pd.to_numeric(average["sample_sd_learned_boundary"], errors="coerce").to_numpy(float)
    counts = pd.to_numeric(
        average["number_of_finite_seed_boundaries"], errors="coerce"
    ).to_numpy(float)
    if "main_figure_display_mean_boundary" in average.columns:
        display = pd.to_numeric(
            average["main_figure_display_mean_boundary"], errors="coerce"
        ).to_numpy(float)
    else:
        display = learned.copy()

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    individual_time = pd.to_numeric(individual["time"], errors="coerce").to_numpy(float)
    for index, column in enumerate(c for c in individual.columns if c != "time"):
        values = pd.to_numeric(individual[column], errors="coerce").to_numpy(float)
        valid = np.isfinite(individual_time) & np.isfinite(values)
        ax.plot(
            individual_time[valid],
            values[valid],
            linewidth=0.9,
            alpha=0.35,
            color="0.45",
            label="seed-specific learned boundary" if index == 0 else None,
        )
    ax.plot(
        continuous["time"],
        continuous["boundary"],
        linewidth=2.5,
        color="tab:red",
        label="continuous-time Peskir-Shiryaev boundary",
    )
    valid_mean = np.isfinite(time) & np.isfinite(display)
    ax.plot(
        time[valid_mean],
        display[valid_mean],
        "--",
        linewidth=2.2,
        color="tab:blue",
        label="mean learned boundary (five seeds)",
    )
    valid_band = valid_mean & np.isfinite(sd) & (counts >= 2)
    if np.any(valid_band):
        ax.fill_between(
            time[valid_band],
            (learned - sd)[valid_band],
            (learned + sd)[valid_band],
            color="tab:blue",
            alpha=0.16,
            label="mean learned boundary ± 1 sample SD",
        )
    ax.axhline(K, color="black", linewidth=1.0, label="K=40")
    ax.set_xlabel("time")
    ax.set_ylabel("stock price")
    ax.set_title(f"Learned exercise boundary: N={N}, M={M}")
    ax.legend(frameon=False)
    fig.tight_layout()
    stem = output_folder / f"average_boundary_comparison_N_{N}_M_{M}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return summary


def create_boundary_outputs(results_root: Path, output_root: Path) -> pd.DataFrame:
    folder = output_root / "boundary_analysis"
    folder.mkdir(parents=True, exist_ok=True)
    continuous = load_continuous_boundary(results_root)
    rows: List[Dict[str, object]] = []

    for N in N_VALUES:
        rows.append(
            replot_boundary(
                n_config_root(results_root, N),
                N=N,
                M=BASELINE_M,
                study="exercise_grid_sensitivity",
                continuous=continuous,
                output_folder=folder,
            )
        )

    for M in M_VALUES:
        if M == BASELINE_M:
            baseline = next(
                row for row in rows if row["N"] == 100 and row["M"] == BASELINE_M
            )
            copied = dict(baseline)
            copied["study"] = "training_batch_sensitivity"
            rows.append(copied)
        else:
            rows.append(
                replot_boundary(
                    m_config_root(results_root, M),
                    N=100,
                    M=M,
                    study="training_batch_sensitivity",
                    continuous=continuous,
                    output_folder=folder,
                )
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "boundary_accuracy_summary.csv", index=False)
    summary.loc[summary["study"] == "exercise_grid_sensitivity"].to_csv(
        folder / "boundary_accuracy_N_panel.csv", index=False
    )
    summary.loc[summary["study"] == "training_batch_sensitivity"].to_csv(
        folder / "boundary_accuracy_M_panel.csv", index=False
    )
    try:
        summary.to_excel(folder / "boundary_accuracy_summary.xlsx", index=False)
    except ImportError:
        print("openpyxl is unavailable; Excel output skipped. CSV output is complete.")
    print("Boundary figures and accuracy metrics complete.")
    return summary


def write_readme(output_root: Path, results_root: Path) -> None:
    text = f"""LOCAL CPU-ONLY POST-PROCESSING

Source results: {results_root}

"""
    (output_root / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = resolve_results_root(args.results_root)
    output_root = results_root / OUTPUT_FOLDER
    output_root.mkdir(parents=True, exist_ok=True)
    create_learning_outputs(results_root, output_root)
    summary = create_boundary_outputs(results_root, output_root)
    write_readme(output_root, results_root)
    print("\nLOCAL POST-PROCESSING FINISHED")
    print("No training or TensorFlow was used.")
    print("Output folder:", output_root)
    print(
        "Please upload:",
        output_root / "boundary_analysis" / "boundary_accuracy_summary.csv",
    )
    print("Rows in boundary summary:", len(summary))


if __name__ == "__main__":
    main()
