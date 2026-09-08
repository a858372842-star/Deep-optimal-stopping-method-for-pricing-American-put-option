
from __future__ import annotations

import gc
import importlib.util
import json
import math
import shutil
import sys
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import ndtr

ROOT = Path("/root/autodl-tmp")
SOURCE = ROOT / "outputs_4_1_single_case_sensitivity_five_seed"
ENGINE_FILE = ROOT / "run_4_1_five_seed_sensitivity.py"

OUTPUT = ROOT / "seven_case_value_function_and_learning_curves"
ZIP_PATH = ROOT / "seven_case_value_function_and_learning_curves.zip"

S0 = 40.0
SIGMA = 0.20
BASELINE_BATCH = 8192

N_VALUES = (50, 100, 150)
BATCH_VALUES = (2048, 4096, 8192, 16384)

TARGET_TIME = 0.50

INTERNAL_STOCK_MIN = 0.01
INTERNAL_STOCK_MAX = 80.0
INTERNAL_STOCK_POINTS = 3001

PLOT_STOCK_MIN = 20.0
PLOT_STOCK_MAX = 55.0
PLOT_STOCK_POINTS = 351

ZOOM_STOCK_MIN = 32.0
ZOOM_STOCK_MAX = 42.0

GAUSS_HERMITE_NODES = 32


@dataclass(frozen=True)
class CaseSpec:
    key: str
    study: str
    label: str
    N: int
    batch_size: int
    seed_dirs: Tuple[Path, ...]


# Setup and source-directory helpers

def configure_tensorflow() -> None:
    tf.keras.backend.set_floatx("float32")

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    print("TensorFlow version:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))


def load_engine():
    if not ENGINE_FILE.exists():
        raise FileNotFoundError(f"Experiment engine not found: {ENGINE_FILE}")

    spec = importlib.util.spec_from_file_location(
        "run_4_1_five_seed_sensitivity_engine",
        ENGINE_FILE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import experiment engine: {ENGINE_FILE}")

    module = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


def seed_value(seed_dir: Path) -> int:
    name = seed_dir.name
    if not name.startswith("seed_"):
        raise ValueError(f"Unexpected seed-directory name: {name}")
    return int(name.replace("seed_", "", 1))


def case_directory(seed_dir: Path) -> Path:
    path = seed_dir / "cases" / "S0_40_sigma_0p20"
    if not path.exists():
        raise FileNotFoundError(f"Case directory not found: {path}")
    return path


def five_seed_directories(root: Path, description: str) -> Tuple[Path, ...]:
    seed_dirs = tuple(sorted(path for path in root.glob("seed_*") if path.is_dir()))

    if len(seed_dirs) != 5:
        raise RuntimeError(
            f"{description}: expected 5 seed folders, found {len(seed_dirs)} "
            f"under {root}"
        )

    for seed_dir in seed_dirs:
        result_file = case_directory(seed_dir) / "case_result.json"
        checkpoint_dir = case_directory(seed_dir) / "checkpoints"

        if not result_file.exists():
            raise FileNotFoundError(f"Completed result not found: {result_file}")
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    return seed_dirs


def exercise_grid_seed_dirs(N: int) -> Tuple[Path, ...]:
    root = (
        SOURCE
        / "exercise_grid_sensitivity"
        / f"N_{N}_B_{BASELINE_BATCH}"
    )
    return five_seed_directories(root, f"exercise-grid N={N}")


def training_batch_seed_dirs(batch_size: int) -> Tuple[Path, ...]:
    if batch_size == BASELINE_BATCH:
        # Shared baseline: do not create or use a duplicate result.
        return exercise_grid_seed_dirs(100)

    root = (
        SOURCE
        / "training_path_sensitivity"
        / "N_100"
        / f"B_{batch_size}"
    )
    return five_seed_directories(root, f"training-batch B={batch_size}")


def build_case_specs() -> Tuple[CaseSpec, ...]:
    return (
        CaseSpec(
            key="exercise_N50_B8192",
            study="exercise_grid",
            label="N=50",
            N=50,
            batch_size=8192,
            seed_dirs=exercise_grid_seed_dirs(50),
        ),
        CaseSpec(
            key="exercise_N100_B8192",
            study="exercise_grid",
            label="N=100",
            N=100,
            batch_size=8192,
            seed_dirs=exercise_grid_seed_dirs(100),
        ),
        CaseSpec(
            key="exercise_N150_B8192",
            study="exercise_grid",
            label="N=150",
            N=150,
            batch_size=8192,
            seed_dirs=exercise_grid_seed_dirs(150),
        ),
        CaseSpec(
            key="batch_N100_B2048",
            study="training_batch",
            label="B=2048",
            N=100,
            batch_size=2048,
            seed_dirs=training_batch_seed_dirs(2048),
        ),
        CaseSpec(
            key="batch_N100_B4096",
            study="training_batch",
            label="B=4096",
            N=100,
            batch_size=4096,
            seed_dirs=training_batch_seed_dirs(4096),
        ),
        CaseSpec(
            key="batch_N100_B8192",
            study="training_batch",
            label="B=8192",
            N=100,
            batch_size=8192,
            seed_dirs=training_batch_seed_dirs(8192),
        ),
        CaseSpec(
            key="batch_N100_B16384",
            study="training_batch",
            label="B=16384",
            N=100,
            batch_size=16384,
            seed_dirs=training_batch_seed_dirs(16384),
        ),
    )


# Five-seed learning curves

LEARNING_COLUMNS = (
    "exercise_date_index",
    "step_within_date",
    "soft_objective",
    "hard_policy_batch_value",
    "mean_soft_stop_probability",
)


def read_seed_learning_curve(seed_dir: Path) -> pd.DataFrame:
    path = case_directory(seed_dir) / "training_log.csv"
    if not path.exists():
        raise FileNotFoundError(f"Training log not found: {path}")

    frame = pd.read_csv(path)
    missing = set(LEARNING_COLUMNS).difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")

    for column in LEARNING_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=list(LEARNING_COLUMNS))

    # First average within one seed across exercise dates at each optimizer step.
    return (
        frame.groupby("step_within_date", as_index=False)
        .agg(
            soft_objective=("soft_objective", "mean"),
            hard_policy_value=("hard_policy_batch_value", "mean"),
            stop_probability=("mean_soft_stop_probability", "mean"),
            dates_available=("exercise_date_index", "nunique"),
        )
        .sort_values("step_within_date")
    )


def aggregate_five_seed_learning_curve(
    seed_dirs: Sequence[Path],
) -> pd.DataFrame:
    seed_frames: List[pd.DataFrame] = []

    for seed_dir in seed_dirs:
        seed_frame = read_seed_learning_curve(seed_dir)
        seed_frame["seed"] = seed_value(seed_dir)
        seed_frames.append(seed_frame)

    long_frame = pd.concat(seed_frames, ignore_index=True)

    # Then average the five seed-level curves.
    return (
        long_frame.groupby("step_within_date", as_index=False)
        .agg(
            soft_objective_mean=("soft_objective", "mean"),
            soft_objective_sd=("soft_objective", "std"),
            hard_policy_value_mean=("hard_policy_value", "mean"),
            hard_policy_value_sd=("hard_policy_value", "std"),
            stop_probability_mean=("stop_probability", "mean"),
            stop_probability_sd=("stop_probability", "std"),
            seeds_available=("seed", "nunique"),
        )
        .sort_values("step_within_date")
    )


def plot_learning_comparison(
    summaries: Mapping[str, pd.DataFrame],
    mean_column: str,
    sd_column: str,
    ylabel: str,
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    for label, frame in summaries.items():
        x = frame["step_within_date"].to_numpy(dtype=float)
        mean = frame[mean_column].to_numpy(dtype=float)
        sd = frame[sd_column].fillna(0.0).to_numpy(dtype=float)

        ax.plot(x, mean, linewidth=1.7, label=label)
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.14)

    ax.set_xlabel("training step within each exercise date")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_learning_curve_outputs() -> None:
    output_dir = OUTPUT / "learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    exercise_summaries: Dict[str, pd.DataFrame] = {}

    for N in N_VALUES:
        summary = aggregate_five_seed_learning_curve(
            exercise_grid_seed_dirs(N)
        )
        summary.to_csv(
            output_dir / f"exercise_grid_N_{N}_B_8192_five_seed_mean.csv",
            index=False,
        )
        exercise_summaries[f"N={N}"] = summary
        print(f"[Learning] exercise-grid N={N} complete")

    plot_learning_comparison(
        exercise_summaries,
        "soft_objective_mean",
        "soft_objective_sd",
        "five-seed mean soft training objective",
        output_dir / "exercise_grid_soft_objective_five_seed_mean",
    )
    plot_learning_comparison(
        exercise_summaries,
        "hard_policy_value_mean",
        "hard_policy_value_sd",
        "five-seed mean hard-policy batch value",
        output_dir / "exercise_grid_hard_value_five_seed_mean",
    )
    plot_learning_comparison(
        exercise_summaries,
        "stop_probability_mean",
        "stop_probability_sd",
        "five-seed mean soft stopping probability",
        output_dir / "exercise_grid_stop_probability_five_seed_mean",
    )

    batch_summaries: Dict[str, pd.DataFrame] = {}

    for batch_size in BATCH_VALUES:
        summary = aggregate_five_seed_learning_curve(
            training_batch_seed_dirs(batch_size)
        )
        summary.to_csv(
            output_dir
            / f"training_batch_N_100_B_{batch_size}_five_seed_mean.csv",
            index=False,
        )
        batch_summaries[f"B={batch_size}"] = summary
        print(f"[Learning] training-batch B={batch_size} complete")

    plot_learning_comparison(
        batch_summaries,
        "soft_objective_mean",
        "soft_objective_sd",
        "five-seed mean soft training objective",
        output_dir / "training_batch_soft_objective_five_seed_mean",
    )
    plot_learning_comparison(
        batch_summaries,
        "hard_policy_value_mean",
        "hard_policy_value_sd",
        "five-seed mean hard-policy batch value",
        output_dir / "training_batch_hard_value_five_seed_mean",
    )
    plot_learning_comparison(
        batch_summaries,
        "stop_probability_mean",
        "stop_probability_sd",
        "five-seed mean soft stopping probability",
        output_dir / "training_batch_stop_probability_five_seed_mean",
    )


# Restore saved stopping networks

def build_experiment_config(
    engine,
    N: int,
    batch_size: int,
    seed: int,
    seed_dir: Path,
):
    cfg = engine.ExperimentConfig()

    return replace(
        cfg,
        N=int(N),
        training_batch_size=int(batch_size),
        base_seed=int(seed),
        output_dir=str(seed_dir.parent.resolve()),
        require_gpu=False,
    )


def restore_models(
    engine,
    N: int,
    batch_size: int,
    seed_dir: Path,
):
    seed = seed_value(seed_dir)
    cfg = build_experiment_config(
        engine,
        N=N,
        batch_size=batch_size,
        seed=seed,
        seed_dir=seed_dir,
    )

    models, _ = engine.make_models_and_optimizers(cfg, S0)

    dummy_features = tf.zeros((1, 2), dtype=tf.float32)
    for n in range(1, N):
        if models[n] is None:
            raise RuntimeError(f"Stopping model is missing at date n={n}")
        _ = models[n](dummy_features, training=False)

    checkpoint = tf.train.Checkpoint(
        **{f"net_{n:02d}": models[n] for n in range(1, N)}
    )

    checkpoint_dir = case_directory(seed_dir) / "checkpoints"
    latest = tf.train.latest_checkpoint(str(checkpoint_dir))

    if latest is None:
        raise FileNotFoundError(f"No TensorFlow checkpoint under {checkpoint_dir}")

    status = checkpoint.restore(latest)
    status.expect_partial()

    print(
        f"[Restore] N={N}, B={batch_size}, seed={seed}: {latest}"
    )
    return cfg, models


# Learned-policy value-function reconstruction

def interpolate_value(
    query: np.ndarray,
    stock_grid: np.ndarray,
    values: np.ndarray,
    strike: float,
) -> np.ndarray:
    flat_query = query.reshape(-1)
    interpolated = np.interp(flat_query, stock_grid, values)

    below = flat_query < stock_grid[0]
    above = flat_query > stock_grid[-1]

    if np.any(below):
        interpolated[below] = np.maximum(strike - flat_query[below], 0.0)
    if np.any(above):
        interpolated[above] = 0.0

    return interpolated.reshape(query.shape)


def target_exercise_index(N: int) -> int:
    raw_index = TARGET_TIME * N
    index = int(round(raw_index))

    if not math.isclose(index / N, TARGET_TIME, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(
            f"TARGET_TIME={TARGET_TIME} is not exactly on the N={N} grid"
        )

    if index <= 0 or index >= N:
        raise ValueError(f"Invalid target exercise index n={index} for N={N}")

    return index


def reconstruct_value_at_target(
    cfg,
    models,
    sigma: float,
    stock_grid: np.ndarray,
    target_index: int,
) -> np.ndarray:

    N = int(cfg.N)
    strike = float(cfg.K)
    dt = float(cfg.T / N)

    gh_x, gh_w = np.polynomial.hermite.hermgauss(
        GAUSS_HERMITE_NODES
    )
    normal_nodes = np.sqrt(2.0) * gh_x
    normal_weights = gh_w / np.sqrt(np.pi)

    growth = np.exp(
        (cfg.r - cfg.q - 0.5 * sigma * sigma) * dt
        + sigma * math.sqrt(dt) * normal_nodes
    )

    immediate = np.maximum(strike - stock_grid, 0.0)
    value_next = immediate.copy()

    for n in range(N - 1, target_index - 1, -1):
        next_stock = stock_grid[:, None] * growth[None, :]
        next_value = interpolate_value(
            next_stock,
            stock_grid,
            value_next,
            strike,
        )

        continuation = (
            math.exp(-cfg.r * dt)
            * (next_value @ normal_weights)
        )

        discounted_immediate_feature = (
            math.exp(-cfg.r * n * dt) * immediate
        )
        features = np.column_stack(
            [stock_grid, discounted_immediate_feature]
        ).astype(np.float32)

        stop_probability = (
            models[n](
                tf.convert_to_tensor(features),
                training=False,
            )
            .numpy()
            .reshape(-1)
            .astype(np.float64)
        )

        hard_stop = stop_probability >= 0.5
        value_now = np.where(
            hard_stop,
            immediate,
            continuation,
        )

        value_next = value_now

    return value_next


# Continuous-time Peskir reference

def continuous_american_put_reference(
    spots: np.ndarray,
    t: float,
    cfg,
    sigma: float,
    peskir,
) -> np.ndarray:
    strike = float(cfg.K)
    rate = float(cfg.r)
    maturity = float(cfg.T)
    remaining_time = maturity - t

    if remaining_time <= 0.0:
        return np.maximum(strike - spots, 0.0)

    sqrt_tau = math.sqrt(remaining_time)

    d1 = (
        np.log(spots / strike)
        + (rate + 0.5 * sigma * sigma) * remaining_time
    ) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    european_value = (
        strike * math.exp(-rate * remaining_time) * ndtr(-d2)
        - spots * ndtr(-d1)
    )

    integration_step = float(peskir.times[1] - peskir.times[0])
    u = np.arange(
        0.0,
        remaining_time,
        integration_step,
        dtype=np.float64,
    )
    future_boundary = np.interp(
        t + u,
        peskir.times,
        peskir.boundary,
    )

    probability = np.empty(
        (len(spots), len(u)),
        dtype=np.float64,
    )

    boundary_now = float(
        np.interp(t, peskir.times, peskir.boundary)
    )

    probability[:, 0] = np.where(
        spots < boundary_now - 1.0e-12,
        1.0,
        np.where(
            np.abs(spots - boundary_now) <= 1.0e-12,
            0.5,
            0.0,
        ),
    )

    if len(u) > 1:
        positive_u = u[1:]
        argument = (
            np.log(future_boundary[1:][None, :] / spots[:, None])
            - (rate - 0.5 * sigma * sigma)
            * positive_u[None, :]
        ) / (
            sigma * np.sqrt(positive_u)[None, :]
        )

        probability[:, 1:] = ndtr(argument)

    early_exercise_premium = (
        rate
        * strike
        * integration_step
        * np.sum(
            np.exp(-rate * u)[None, :] * probability,
            axis=1,
        )
    )

    return european_value + early_exercise_premium


# Compute each unique configuration once

def unique_configuration_key(case: CaseSpec) -> Tuple[int, int]:
    return case.N, case.batch_size


def compute_unique_value_results(
    engine,
    cases: Sequence[CaseSpec],
) -> Dict[Tuple[int, int], pd.DataFrame]:
    internal_grid = np.linspace(
        INTERNAL_STOCK_MIN,
        INTERNAL_STOCK_MAX,
        INTERNAL_STOCK_POINTS,
        dtype=np.float64,
    )
    plot_grid = np.linspace(
        PLOT_STOCK_MIN,
        PLOT_STOCK_MAX,
        PLOT_STOCK_POINTS,
        dtype=np.float64,
    )

    base_cfg = engine.ExperimentConfig()
    peskir = engine.load_or_compute_peskir_boundary(
        base_cfg,
        SIGMA,
        SOURCE / "shared_continuous_time_benchmark",
        force=False,
    )

    reference = continuous_american_put_reference(
        plot_grid,
        TARGET_TIME,
        base_cfg,
        SIGMA,
        peskir,
    )
    intrinsic = np.maximum(
        float(base_cfg.K) - plot_grid,
        0.0,
    )

    unique_cases: Dict[Tuple[int, int], CaseSpec] = {}
    for case in cases:
        unique_cases.setdefault(unique_configuration_key(case), case)

    results: Dict[Tuple[int, int], pd.DataFrame] = {}

    for (N, batch_size), case in unique_cases.items():
        target_index = target_exercise_index(N)
        seed_curves: List[np.ndarray] = []
        seed_numbers: List[int] = []

        print()
        print("=" * 78)
        print(
            f"VALUE CONFIGURATION: N={N}, B={batch_size}, "
            f"target date n={target_index}, t={TARGET_TIME:.3f}"
        )
        print("=" * 78)

        for seed_dir in case.seed_dirs:
            tf.keras.backend.clear_session()
            gc.collect()

            cfg, models = restore_models(
                engine,
                N=N,
                batch_size=batch_size,
                seed_dir=seed_dir,
            )

            value_internal = reconstruct_value_at_target(
                cfg,
                models,
                SIGMA,
                internal_grid,
                target_index,
            )

            value_plot = np.interp(
                plot_grid,
                internal_grid,
                value_internal,
            )

            seed_curves.append(value_plot)
            seed_numbers.append(seed_value(seed_dir))

            del models, value_internal
            tf.keras.backend.clear_session()
            gc.collect()

            print(
                f"[Value complete] N={N}, B={batch_size}, "
                f"seed={seed_numbers[-1]}"
            )

        matrix = np.vstack(seed_curves)
        learned_mean = np.mean(matrix, axis=0)
        learned_sd = np.std(matrix, axis=0, ddof=1)

        frame = pd.DataFrame(
            {
                "stock_price": plot_grid,
                "learned_value_mean": learned_mean,
                "learned_value_sd": learned_sd,
                "continuous_peskir_value": reference,
                "intrinsic_payoff": intrinsic,
                "signed_error_vs_peskir": learned_mean - reference,
                "absolute_error_vs_peskir": np.abs(
                    learned_mean - reference
                ),
                "learned_continuation_premium": (
                    learned_mean - intrinsic
                ),
                "peskir_continuation_premium": (
                    reference - intrinsic
                ),
                "N": N,
                "training_batch_size": batch_size,
                "exercise_date_index": target_index,
                "actual_time": target_index / N,
                "number_of_seeds": matrix.shape[0],
            }
        )

        for row_index, seed in enumerate(seed_numbers):
            frame[f"seed_{seed}"] = matrix[row_index]

        results[(N, batch_size)] = frame

    return results


# Value-function figures and tables

def save_individual_value_figure(
    frame: pd.DataFrame,
    case: CaseSpec,
    output_stem: Path,
    zoom: bool,
) -> None:
    stock = frame["stock_price"].to_numpy(dtype=float)
    learned_mean = frame["learned_value_mean"].to_numpy(dtype=float)
    learned_sd = frame["learned_value_sd"].to_numpy(dtype=float)
    reference = frame["continuous_peskir_value"].to_numpy(dtype=float)
    intrinsic = frame["intrinsic_payoff"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    ax.plot(
        stock,
        learned_mean,
        linewidth=2.0,
        label=f"learned-policy mean ({case.label})",
    )
    ax.fill_between(
        stock,
        learned_mean - learned_sd,
        learned_mean + learned_sd,
        alpha=0.18,
        label="mean ± 1 SD across five seeds",
    )
    ax.plot(
        stock,
        reference,
        linestyle="--",
        linewidth=1.8,
        label="continuous-time Peskir reference",
    )
    ax.plot(
        stock,
        intrinsic,
        linestyle=":",
        linewidth=1.5,
        label="intrinsic payoff",
    )

    if zoom:
        ax.set_xlim(ZOOM_STOCK_MIN, ZOOM_STOCK_MAX)

    ax.set_xlabel("stock price")
    ax.set_ylabel("option value at t=0.5")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def save_value_comparison(
    case_frames: Sequence[Tuple[CaseSpec, pd.DataFrame]],
    output_stem: Path,
    zoom: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    reference_drawn = False

    for case, frame in case_frames:
        stock = frame["stock_price"].to_numpy(dtype=float)
        mean = frame["learned_value_mean"].to_numpy(dtype=float)

        ax.plot(
            stock,
            mean,
            linewidth=1.8,
            label=f"{case.label}, five-seed mean",
        )

        if not reference_drawn:
            ax.plot(
                stock,
                frame["continuous_peskir_value"].to_numpy(dtype=float),
                linestyle="--",
                linewidth=1.8,
                label="continuous-time Peskir reference",
            )
            reference_drawn = True

    if zoom:
        ax.set_xlim(ZOOM_STOCK_MIN, ZOOM_STOCK_MAX)

    ax.set_xlabel("stock price")
    ax.set_ylabel("option value at t=0.5")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_error_comparison(
    case_frames: Sequence[Tuple[CaseSpec, pd.DataFrame]],
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    for case, frame in case_frames:
        stock = frame["stock_price"].to_numpy(dtype=float)
        error = frame["signed_error_vs_peskir"].to_numpy(dtype=float)

        ax.plot(
            stock,
            error,
            linewidth=1.8,
            label=case.label,
        )

    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlim(ZOOM_STOCK_MIN, ZOOM_STOCK_MAX)
    ax.set_xlabel("stock price")
    ax.set_ylabel(
        "learned-policy value minus Peskir reference"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_premium_comparison(
    case_frames: Sequence[Tuple[CaseSpec, pd.DataFrame]],
    output_stem: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    reference_drawn = False

    for case, frame in case_frames:
        stock = frame["stock_price"].to_numpy(dtype=float)
        premium = frame[
            "learned_continuation_premium"
        ].to_numpy(dtype=float)

        ax.plot(
            stock,
            premium,
            linewidth=1.8,
            label=case.label,
        )

        if not reference_drawn:
            ax.plot(
                stock,
                frame[
                    "peskir_continuation_premium"
                ].to_numpy(dtype=float),
                linestyle="--",
                linewidth=1.8,
                label="Peskir continuation premium",
            )
            reference_drawn = True

    ax.set_xlim(ZOOM_STOCK_MIN, ZOOM_STOCK_MAX)
    ax.set_xlabel("stock price")
    ax.set_ylabel("value minus intrinsic payoff")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def generate_value_outputs(
    cases: Sequence[CaseSpec],
    unique_results: Mapping[Tuple[int, int], pd.DataFrame],
) -> None:
    value_root = OUTPUT / "value_functions"
    individual_root = value_root / "individual_cases"
    comparison_root = value_root / "comparisons"

    individual_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    for case in cases:
        frame = unique_results[(case.N, case.batch_size)].copy()

        case_dir = individual_root / case.study / case.key
        case_dir.mkdir(parents=True, exist_ok=True)

        csv_path = case_dir / f"{case.key}_five_seed_value_function.csv"
        frame.to_csv(csv_path, index=False)

        save_individual_value_figure(
            frame,
            case,
            case_dir / f"{case.key}_value_full",
            zoom=False,
        )
        save_individual_value_figure(
            frame,
            case,
            case_dir / f"{case.key}_value_zoom",
            zoom=True,
        )

        max_error_row = frame.loc[
            frame["absolute_error_vs_peskir"].idxmax()
        ]

        error_rows.append(
            {
                "case_key": case.key,
                "study": case.study,
                "case_label": case.label,
                "N": case.N,
                "training_batch_size": case.batch_size,
                "target_time": TARGET_TIME,
                "max_absolute_error": float(
                    frame["absolute_error_vs_peskir"].max()
                ),
                "mean_absolute_error": float(
                    frame["absolute_error_vs_peskir"].mean()
                ),
                "most_negative_error": float(
                    frame["signed_error_vs_peskir"].min()
                ),
                "most_positive_error": float(
                    frame["signed_error_vs_peskir"].max()
                ),
                "stock_at_max_absolute_error": float(
                    max_error_row["stock_price"]
                ),
                "maximum_seed_sd": float(
                    frame["learned_value_sd"].max()
                ),
            }
        )

        index_rows.append(
            {
                "case_key": case.key,
                "study": case.study,
                "case_label": case.label,
                "N": case.N,
                "training_batch_size": case.batch_size,
                "target_time": TARGET_TIME,
                "source_seed_folders": "; ".join(
                    str(path) for path in case.seed_dirs
                ),
                "csv_file": str(csv_path),
                "full_png": str(
                    case_dir / f"{case.key}_value_full.png"
                ),
                "zoom_png": str(
                    case_dir / f"{case.key}_value_zoom.png"
                ),
            }
        )

    exercise_cases = [
        (case, unique_results[(case.N, case.batch_size)])
        for case in cases
        if case.study == "exercise_grid"
    ]
    batch_cases = [
        (case, unique_results[(case.N, case.batch_size)])
        for case in cases
        if case.study == "training_batch"
    ]

    save_value_comparison(
        exercise_cases,
        comparison_root
        / "exercise_grid_value_comparison_t_0p5",
        zoom=True,
    )
    save_error_comparison(
        exercise_cases,
        comparison_root
        / "exercise_grid_value_error_vs_peskir_t_0p5",
    )
    save_premium_comparison(
        exercise_cases,
        comparison_root
        / "exercise_grid_continuation_premium_t_0p5",
    )

    save_value_comparison(
        batch_cases,
        comparison_root
        / "training_batch_value_comparison_t_0p5",
        zoom=True,
    )
    save_error_comparison(
        batch_cases,
        comparison_root
        / "training_batch_value_error_vs_peskir_t_0p5",
    )
    save_premium_comparison(
        batch_cases,
        comparison_root
        / "training_batch_continuation_premium_t_0p5",
    )

    pd.DataFrame(index_rows).to_csv(
        value_root / "seven_case_value_function_index.csv",
        index=False,
    )
    pd.DataFrame(error_rows).to_csv(
        value_root / "seven_case_value_function_error_summary.csv",
        index=False,
    )

def save_run_manifest(cases: Sequence[CaseSpec]) -> None:
    manifest = {
        "source_root": str(SOURCE),
        "engine_file": str(ENGINE_FILE),
        "output_root": str(OUTPUT),
        "target_time": TARGET_TIME,
        "sigma": SIGMA,
        "stock_plot_range": [
            PLOT_STOCK_MIN,
            PLOT_STOCK_MAX,
        ],
        "stock_zoom_range": [
            ZOOM_STOCK_MIN,
            ZOOM_STOCK_MAX,
        ],
        "gauss_hermite_nodes": GAUSS_HERMITE_NODES,
        "reported_cases": [
            {
                "key": case.key,
                "study": case.study,
                "label": case.label,
                "N": case.N,
                "training_batch_size": case.batch_size,
                "seeds": [
                    seed_value(path) for path in case.seed_dirs
                ],
            }
            for case in cases
        ],
    }

    with (OUTPUT / "run_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)


def make_zip_archive() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    files = sorted(
        path for path in OUTPUT.rglob("*")
        if path.is_file()
    )

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        for index, path in enumerate(files, start=1):
            archive.write(
                path,
                path.relative_to(ROOT),
            )

            if index % 100 == 0 or index == len(files):
                print(f"[ZIP] {index}/{len(files)} files")

    print("ZIP created:", ZIP_PATH)
    print(
        "ZIP size:",
        f"{ZIP_PATH.stat().st_size / 1024**2:.2f} MB",
    )


# Main

def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Completed five-seed output not found: {SOURCE}"
        )

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    OUTPUT.mkdir(parents=True)

    configure_tensorflow()
    engine = load_engine()
    cases = build_case_specs()

    print()
    print("Generating five-seed mean learning curves...")
    generate_learning_curve_outputs()

    print()
    print("Reconstructing six unique configurations for seven reported cases...")
    unique_results = compute_unique_value_results(
        engine,
        cases,
    )

    print()
    print("Saving seven-case value-function outputs...")
    generate_value_outputs(
        cases,
        unique_results,
    )

    write_readme(cases)
    save_run_manifest(cases)
    make_zip_archive()

    print()
    print("=" * 78)
    print("ALL SEVEN-CASE POST-PROCESSING FINISHED")
    print("Output folder:", OUTPUT)
    print("Download file:", ZIP_PATH)
    print("=" * 78)


if __name__ == "__main__":
    main()
