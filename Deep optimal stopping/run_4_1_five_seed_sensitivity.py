from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

try:
    import tensorflow as tf
except ImportError as exc:  
    raise SystemExit(
        "TensorFlow is not installed. In the VS Code WSL terminal run:\n"
        "  python -m pip install 'tensorflow[and-cuda]' numpy scipy pandas matplotlib h5py openpyxl\n"
        "Then verify with:\n"
        "  python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
    ) from exc


UPPER_ESTIMATOR_VERSION = "streamed_date_major_v2"
UPPER_INNER_CHUNK_ENV = "BECKER_UPPER_INNER_CHUNK_SIZE"
DEFAULT_UPPER_INNER_CHUNK_SIZE = 4_096



# Configuration
@dataclass(frozen=True)
class ExperimentConfig:
    algorithm_version: str = "becker_datewise_frozen_future_v1"

    K: float = 40.0
    r: float = 0.06
    q: float = 0.0
    T: float = 1.0
    N: int = 50
    dimension: int = 1

    hidden_width: int = 41  # d + 40 = 41
    training_steps: int = 3001
    training_batch_size: int = 8192
    learning_rate: float = 1.0e-3
    batch_norm_momentum: float = 0.99
    training_log_every: int = 50
    checkpoint_every: int = 100

    K_L: int = 4_096_000
    lower_chunk_size: int = 128_000  
    K_U: int = 1_024
    J: int = 16_384
    upper_outer_batch_size: int = 8  
    f0_pilot_paths: int = 262_144


    peskir_time_steps: int = 4000
    peskir_coarse_time_steps: int = 2000
    peskir_terminal_z_nodes: int = 4096
    peskir_root_xtol: float = 1.0e-11
    run_peskir_coarse_check: bool = True


    boundary_stock_grid_size: int = 4001
    boundary_stock_min_multiple_beta: float = 0.55
    boundary_stock_max_multiple_K: float = 1.05
    sample_path_steps: int = 1200
    sample_path_search_paths: int = 10_000


    base_seed: int = 20260711
    output_dir: str = "outputs_4_1_becker_recursive_gpu"
    require_gpu: bool = True
    confidence_level: float = 0.95


CASES: Tuple[Tuple[float, float], ...] = (
    (38.0, 0.20),
    (40.0, 0.20),
    (42.0, 0.20),
    (38.0, 0.40),
    (40.0, 0.40),
    (42.0, 0.40),
)


LITERATURE_LS: Dict[Tuple[float, float], float] = {
    (38.0, 0.20): 3.244,
    (40.0, 0.20): 2.313,
    (42.0, 0.20): 1.617,
    (38.0, 0.40): 6.139,
    (40.0, 0.40): 5.308,
    (42.0, 0.40): 4.588,
}

# General utilities

def _engine_parse_args_unused() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use tiny sample sizes in a separate output folder for a pipeline test.",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=None,
        help="Run one case only, indexed 0,...,5. Default: run all six cases.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Delete selected case folders and recompute them from scratch.",
    )
    parser.add_argument(
        "--force-peskir",
        action="store_true",
        help="Ignore cached Peskir boundary CSV files and recompute them.",
    )
    parser.add_argument(
        "--skip-upper",
        action="store_true",
        help="Debug option: stop after training/lower bound/figures. Not a full result.",
    )
    return parser.parse_args()


def _engine_quick_config_unused(cfg: ExperimentConfig) -> ExperimentConfig:
    return replace(
        cfg,
        training_steps=4,
        training_batch_size=256,
        training_log_every=1,
        checkpoint_every=1,
        K_L=4096,
        lower_chunk_size=1024,
        K_U=8,
        J=64,
        upper_outer_batch_size=2,
        f0_pilot_paths=2048,
        peskir_time_steps=240,
        peskir_coarse_time_steps=120,
        peskir_terminal_z_nodes=512,
        boundary_stock_grid_size=801,
        sample_path_steps=300,
        sample_path_search_paths=1000,
        output_dir="outputs_4_1_becker_recursive_gpu_QUICK_TEST",
        require_gpu=False,
    )


def ensure_divisibility(cfg: ExperimentConfig) -> None:
    if cfg.K_L % cfg.lower_chunk_size != 0:
        raise ValueError("lower_chunk_size must divide K_L exactly in the strict run.")
    if cfg.K_U % cfg.upper_outer_batch_size != 0:
        raise ValueError("upper_outer_batch_size must divide K_U exactly.")
    if cfg.dimension != 1:
        raise ValueError("This Section 4.1 script is intentionally one-dimensional.")
    if abs(cfg.q) > 1.0e-15:
        raise ValueError(
            "Peskir--Shiryaev Section 25.2 used here assumes q=0. "
            "Do not use this benchmark solver with a non-zero dividend yield."
        )


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_tensorflow(require_gpu: bool) -> List[Any]:
    tf.keras.backend.set_floatx("float32")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            # This can only be changed before GPUs are initialized.
            pass

    print("\nTensorFlow version:", tf.__version__)
    print("Physical GPU devices:", gpus)
    if require_gpu and not gpus:
        raise RuntimeError(
            "No TensorFlow GPU is visible. The full run is configured to require a GPU.\n"
            "Inside WSL, verify NVIDIA access with `nvidia-smi`, then run:\n"
            "  python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
        )
    return gpus


def case_tag(S0: float, sigma: float) -> str:
    return f"S0_{int(round(S0))}_sigma_{sigma:.2f}".replace(".", "p")


def sigma_tag(sigma: float) -> str:
    return f"sigma_{sigma:.2f}".replace(".", "p")


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=float)
    tmp.replace(path)


def json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def update_timing(timing_path: Path, key: str, seconds: float) -> Dict[str, float]:
    timing = json_load(timing_path, {})
    timing[key] = float(timing.get(key, 0.0) + seconds)
    json_dump(timing_path, timing)
    return timing


def config_signature(cfg: ExperimentConfig, S0: float, sigma: float) -> Dict[str, Any]:
    # Include all items that affect trained weights or numerical estimators.
    payload = asdict(cfg)
    payload.update({"S0": S0, "sigma": sigma})
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "parameters": payload}


def check_or_write_signature(case_dir: Path, signature: Dict[str, Any]) -> None:
    path = case_dir / "configuration_signature.json"
    if path.exists():
        old = json_load(path, {})
        if old.get("sha256") != signature.get("sha256"):
            raise RuntimeError(
                f"Configuration mismatch in {case_dir}. Use --force-retrain to create "
                "a clean case folder for the new settings."
            )
    else:
        json_dump(path, signature)


def seed_pair(base_seed: int, stream: int, index: int) -> np.ndarray:
    # TensorFlow stateless random seeds are two int32 values.
    a = np.int32((base_seed + 104_729 * stream) % 2_147_483_647)
    b = np.int32((index + 1_000_003 * stream + 97) % 2_147_483_647)
    return np.asarray([a, b], dtype=np.int32)


def normal_quantile_975(confidence_level: float) -> float:
    # For the requested 95% interval this is 1.959963984540054.
    alpha = 1.0 - confidence_level
    return float(math.sqrt(2.0) * _erfinv(1.0 - alpha))


def _erfinv(x: float) -> float:
    from scipy.special import erfinv
    return float(erfinv(x))

@dataclass
class PeskirBoundaryResult:
    times: np.ndarray
    boundary: np.ndarray
    beta: float
    h: float
    sigma: float
    terminal_integral_max_abs_error: float
    coarse_fine_value_differences: Dict[str, float]
    computation_seconds: float


def european_put_closed_form(S: float, K: float, r: float, sigma: float, tau: float) -> float:
    if tau <= 0.0:
        return max(K - S, 0.0)
    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return float(K * math.exp(-r * tau) * ndtr(-d2) - S * ndtr(-d1))


def perpetual_put_boundary_beta(K: float, r: float, sigma: float) -> float:
    # Peskir--Shiryaev (25.1.16), D=sigma^2/2:
    # beta = K / (1 + D/r) = 2rK/(2r+sigma^2).
    D = 0.5 * sigma * sigma
    return float(K / (1.0 + D / r))


def discounted_terminal_put_riemann(
    x: float,
    tau: float,
    K: float,
    r: float,
    sigma: float,
    y_midpoints: np.ndarray,
    dy: float,
) -> float:

    if tau <= 0.0:
        return max(K - x, 0.0)
    gamma = r - 0.5 * sigma * sigma
    arg = (np.log(y_midpoints / x) - gamma * tau) / (sigma * math.sqrt(tau))
    return float(math.exp(-r * tau) * dy * np.sum(ndtr(arg), dtype=np.float64))


def solve_peskir_boundary_once(
    cfg: ExperimentConfig,
    sigma: float,
    time_steps: int,
) -> Tuple[np.ndarray, np.ndarray, float, float]:

    K, r, T = cfg.K, cfg.r, cfg.T
    M = int(time_steps)
    h = T / M
    beta = perpetual_put_boundary_beta(K, r, sigma)
    gamma = r - 0.5 * sigma * sigma

    # B[i] = b(T-i*h): B[0]=b(T)=K and B[M]=b(0).
    B = np.empty(M + 1, dtype=np.float64)
    B[0] = K

    z_nodes = int(cfg.peskir_terminal_z_nodes)
    dy = K / z_nodes
    y_midpoints = (np.arange(z_nodes, dtype=np.float64) + 0.5) * dy

    for i in range(1, M + 1):
        tau = i * h
        if i > 1:
            j = np.arange(1, i, dtype=np.int64)
            u = j.astype(np.float64) * h
            future_B = B[i - j]
            exp_discount = np.exp(-r * u)
            sqrt_u = np.sqrt(u)
        else:
            u = np.empty(0, dtype=np.float64)
            future_B = np.empty(0, dtype=np.float64)
            exp_discount = np.empty(0, dtype=np.float64)
            sqrt_u = np.empty(0, dtype=np.float64)

        def residual(x: float) -> float:
            terminal = discounted_terminal_put_riemann(
                x=x,
                tau=tau,
                K=K,
                r=r,
                sigma=sigma,
                y_midpoints=y_midpoints,
                dy=dy,
            )
            probability_sum = 0.5
            if i > 1:
                arg = (np.log(future_B / x) - gamma * u) / (sigma * sqrt_u)
                probability_sum += float(
                    np.sum(exp_discount * ndtr(arg), dtype=np.float64)
                )
            premium = r * K * h * probability_sum
            return float(K - x - terminal - premium)

        lo = beta * (1.0 + 1.0e-12)
        hi = K * (1.0 - 1.0e-12)
        f_lo, f_hi = residual(lo), residual(hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            raise FloatingPointError(
                f"Non-finite boundary residual at time-to-maturity index {i}."
            )
        if f_lo * f_hi > 0.0:
            raise RuntimeError(
                "Could not bracket the Peskir boundary root at "
                f"tau={tau:.8f}; residual(beta)={f_lo:.6e}, "
                f"residual(K)={f_hi:.6e}."
            )

        B[i] = brentq(
            residual,
            lo,
            hi,
            xtol=cfg.peskir_root_xtol,
            rtol=4.0 * np.finfo(np.float64).eps,
            maxiter=200,
        )

    times = np.linspace(0.0, T, M + 1, dtype=np.float64)
    boundary = B[::-1].copy()
    if np.any(np.diff(boundary) < -5.0e-8):
        min_diff = float(np.min(np.diff(boundary)))
        raise RuntimeError(
            f"Computed Peskir boundary is not increasing; min diff={min_diff}."
        )
    return times, boundary, beta, h


def peskir_value_from_boundary(
    S0: float,
    cfg: ExperimentConfig,
    sigma: float,
    times: np.ndarray,
    boundary: np.ndarray,
) -> float:
    """Evaluate (25.2.36) at t=0 with a left Riemann sum."""
    K, r, T = cfg.K, cfg.r, cfg.T
    M = len(times) - 1
    h = T / M
    gamma = r - 0.5 * sigma * sigma

    z_nodes = int(cfg.peskir_terminal_z_nodes)
    dy = K / z_nodes
    y_midpoints = (np.arange(z_nodes, dtype=np.float64) + 0.5) * dy
    terminal = discounted_terminal_put_riemann(
        x=S0,
        tau=T,
        K=K,
        r=r,
        sigma=sigma,
        y_midpoints=y_midpoints,
        dy=dy,
    )

    if S0 < boundary[0] - 1.0e-12:
        initial_probability = 1.0
    elif abs(S0 - boundary[0]) <= 1.0e-12:
        initial_probability = 0.5
    else:
        initial_probability = 0.0

    if M > 1:
        u = np.arange(1, M, dtype=np.float64) * h
        arg = (np.log(boundary[1:M] / S0) - gamma * u) / (sigma * np.sqrt(u))
        probability_sum = initial_probability + float(
            np.sum(np.exp(-r * u) * ndtr(arg), dtype=np.float64)
        )
    else:
        probability_sum = initial_probability
    premium = r * K * h * probability_sum
    return float(terminal + premium)


def terminal_riemann_validation_error(
    cfg: ExperimentConfig,
    sigma: float,
    test_spots: Sequence[float],
) -> float:
    z_nodes = int(cfg.peskir_terminal_z_nodes)
    dy = cfg.K / z_nodes
    y_midpoints = (np.arange(z_nodes, dtype=np.float64) + 0.5) * dy
    errors = []
    for x in test_spots:
        riemann = discounted_terminal_put_riemann(
            x=x,
            tau=cfg.T,
            K=cfg.K,
            r=cfg.r,
            sigma=sigma,
            y_midpoints=y_midpoints,
            dy=dy,
        )
        closed = european_put_closed_form(x, cfg.K, cfg.r, sigma, cfg.T)
        errors.append(abs(riemann - closed))
    return float(max(errors))


def load_or_compute_peskir_boundary(
    cfg: ExperimentConfig,
    sigma: float,
    root: Path,
    force: bool,
) -> PeskirBoundaryResult:
    folder = root / "peskir_boundaries" / sigma_tag(sigma)
    folder.mkdir(parents=True, exist_ok=True)
    fine_csv = folder / f"boundary_M{cfg.peskir_time_steps}.csv"
    meta_json = folder / f"boundary_M{cfg.peskir_time_steps}_metadata.json"

    start = time.perf_counter()
    if fine_csv.exists() and meta_json.exists() and not force:
        frame = pd.read_csv(fine_csv)
        meta = json_load(meta_json, {})
        return PeskirBoundaryResult(
            times=frame["time"].to_numpy(dtype=np.float64),
            boundary=frame["boundary"].to_numpy(dtype=np.float64),
            beta=float(meta["beta"]),
            h=float(meta["h"]),
            sigma=sigma,
            terminal_integral_max_abs_error=float(meta["terminal_integral_max_abs_error"]),
            coarse_fine_value_differences=dict(meta.get("coarse_fine_value_differences", {})),
            computation_seconds=float(meta.get("computation_seconds", 0.0)),
        )

    print(f"\n[Peskir] Solving sigma={sigma:.2f}, M={cfg.peskir_time_steps} ...")
    times, boundary, beta, h = solve_peskir_boundary_once(cfg, sigma, cfg.peskir_time_steps)
    pd.DataFrame({"time": times, "boundary": boundary}).to_csv(fine_csv, index=False)

    coarse_fine: Dict[str, float] = {}
    if cfg.run_peskir_coarse_check:
        coarse_times, coarse_boundary, _, _ = solve_peskir_boundary_once(
            cfg, sigma, cfg.peskir_coarse_time_steps
        )
        coarse_csv = folder / f"boundary_M{cfg.peskir_coarse_time_steps}.csv"
        pd.DataFrame({"time": coarse_times, "boundary": coarse_boundary}).to_csv(
            coarse_csv, index=False
        )
        for S0 in sorted({case[0] for case in CASES}):
            fine_v = peskir_value_from_boundary(S0, cfg, sigma, times, boundary)
            coarse_v = peskir_value_from_boundary(
                S0, cfg, sigma, coarse_times, coarse_boundary
            )
            coarse_fine[f"S0_{int(S0)}"] = float(fine_v - coarse_v)

    error = terminal_riemann_validation_error(
        cfg,
        sigma,
        [32.0, 38.0, 40.0, 42.0, 48.0],
    )
    elapsed = time.perf_counter() - start
    meta = {
        "source_equations": ["25.2.36", "25.2.37", "25.2.38", "25.2.39"],
        "riemann_convention": {
            "terminal_integral": "midpoint Riemann sum on [0,K]",
            "early_exercise_premium": "left Riemann sum on [0,T-t), with boundary-start limit 1/2",
        },
        "K": cfg.K,
        "r": cfg.r,
        "q": cfg.q,
        "T": cfg.T,
        "sigma": sigma,
        "M": cfg.peskir_time_steps,
        "terminal_z_nodes": cfg.peskir_terminal_z_nodes,
        "beta": beta,
        "h": h,
        "boundary_at_zero": float(boundary[0]),
        "boundary_at_maturity": float(boundary[-1]),
        "terminal_integral_max_abs_error": error,
        "coarse_fine_value_differences": coarse_fine,
        "computation_seconds": elapsed,
    }
    json_dump(meta_json, meta)
    print(
        f"[Peskir] sigma={sigma:.2f}: b(0)={boundary[0]:.8f}, "
        f"beta={beta:.8f}, elapsed={elapsed:.1f}s"
    )
    return PeskirBoundaryResult(
        times=times,
        boundary=boundary,
        beta=beta,
        h=h,
        sigma=sigma,
        terminal_integral_max_abs_error=error,
        coarse_fine_value_differences=coarse_fine,
        computation_seconds=elapsed,
    )

# Becker datewise backward-recursive stopping networks and training

class StoppingNetwork(tf.keras.Model):
    """Two hidden layers for Y_n=(X_n,g_n), Xavier + BN + ReLU + sigmoid."""

    def __init__(self, width: int, bn_momentum: float, name: str):
        super().__init__(name=name)
        initializer1 = tf.keras.initializers.GlorotUniform()
        initializer2 = tf.keras.initializers.GlorotUniform()
        initializer3 = tf.keras.initializers.GlorotUniform()
        self.dense1 = tf.keras.layers.Dense(
            width,
            kernel_initializer=initializer1,
            bias_initializer="zeros",
            name="dense1",
        )
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum, name="bn1")
        self.dense2 = tf.keras.layers.Dense(
            width,
            kernel_initializer=initializer2,
            bias_initializer="zeros",
            name="dense2",
        )
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=bn_momentum, name="bn2")
        self.output_layer = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer=initializer3,
            bias_initializer="zeros",
            name="stop_probability",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        return self.output_layer(x)


def make_models_and_optimizers(
    cfg: ExperimentConfig,
    S0: float,
) -> Tuple[List[Optional[StoppingNetwork]], List[Optional[tf.keras.optimizers.Optimizer]]]:
    models: List[Optional[StoppingNetwork]] = [None] * (cfg.N + 1)
    optimizers: List[Optional[tf.keras.optimizers.Optimizer]] = [None] * (cfg.N + 1)
    dummy = tf.constant([[S0, max(cfg.K - S0, 0.0)]], dtype=tf.float32)

    for n in range(1, cfg.N):
        model = StoppingNetwork(
            width=cfg.hidden_width,
            bn_momentum=cfg.batch_norm_momentum,
            name=f"stopping_net_{n:02d}",
        )
        _ = model(dummy, training=False)  # build variables
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        try:
            optimizer.build(model.trainable_variables)
        except (AttributeError, TypeError):
            zero_grads = [tf.zeros_like(v) for v in model.trainable_variables]
            optimizer.apply_gradients(zip(zero_grads, model.trainable_variables))
        models[n] = model
        optimizers[n] = optimizer
    return models, optimizers


def make_checkpoint(
    cfg: ExperimentConfig,
    case_dir: Path,
    models: Sequence[Optional[StoppingNetwork]],
    optimizers: Sequence[Optional[tf.keras.optimizers.Optimizer]],
    rng: tf.random.Generator,
) -> Tuple[
    tf.train.Checkpoint,
    tf.train.CheckpointManager,
    tf.Variable,
    tf.Variable,
    tf.Variable,
]:
    current_date = tf.Variable(
        cfg.N - 1, dtype=tf.int64, trainable=False, name="current_training_date"
    )
    date_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="date_training_step")
    total_updates = tf.Variable(0, dtype=tf.int64, trainable=False, name="total_updates")

    objects: Dict[str, Any] = {
        "current_date": current_date,
        "date_step": date_step,
        "total_updates": total_updates,
        "rng": rng,
    }
    for n in range(1, len(models) - 1):
        objects[f"net_{n:02d}"] = models[n]
        objects[f"opt_{n:02d}"] = optimizers[n]

    checkpoint = tf.train.Checkpoint(**objects)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=str(case_dir / "checkpoints"),
        max_to_keep=3,
    )
    if manager.latest_checkpoint:
        status = checkpoint.restore(manager.latest_checkpoint)
        status.expect_partial()
        print(
            f"[Resume] Restored {manager.latest_checkpoint}; "
            f"date={int(current_date.numpy())}, "
            f"date_step={int(date_step.numpy())}, "
            f"total_updates={int(total_updates.numpy())}"
        )
    return checkpoint, manager, current_date, date_step, total_updates


def make_date_training_step(
    cfg: ExperimentConfig,
    S0: float,
    sigma: float,
    date_n: int,
    models: Sequence[Optional[StoppingNetwork]],
    optimizer: tf.keras.optimizers.Optimizer,
    rng: tf.random.Generator,
) -> Callable[[], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    N = cfg.N
    if not 1 <= date_n < N:
        raise ValueError(f"date_n must lie in 1,...,{N-1}; got {date_n}.")

    dt = np.float32(cfg.T / N)
    mu_dt = np.float32((cfg.r - cfg.q - 0.5 * sigma * sigma) * dt)
    vol_sqrt_dt = np.float32(sigma * math.sqrt(float(dt)))
    discounts = tf.constant(
        np.exp(-cfg.r * np.linspace(0.0, cfg.T, N + 1)).astype(np.float32)
    )
    K_tf = tf.constant(cfg.K, dtype=tf.float32)
    S0_tf = tf.constant(S0, dtype=tf.float32)
    batch_size = cfg.training_batch_size
    current_model = models[date_n]
    assert current_model is not None

    # The Python loop bounds are fixed when tf.function is traced for date_n.
    @tf.function(reduce_retracing=True)
    def train_step() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        eps = rng.normal(shape=(batch_size, N), dtype=tf.float32)
        log_increments = mu_dt + vol_sqrt_dt * eps
        log_paths = tf.math.log(S0_tf) + tf.cumsum(log_increments, axis=1)
        paths = tf.concat(
            [tf.fill((batch_size, 1), S0_tf), tf.exp(log_paths)], axis=1
        )

        # Realized reward under the already-trained future hard policy.
        future_reward = discounts[N] * tf.nn.relu(K_tf - paths[:, N])
        for m in range(N - 1, date_n, -1):
            future_model = models[m]
            assert future_model is not None
            immediate_m = discounts[m] * tf.nn.relu(K_tf - paths[:, m])
            state_m = tf.stack([paths[:, m], immediate_m], axis=1)
            probability_m = tf.reshape(
                future_model(state_m, training=False), (-1,)
            )
            hard_stop_m = probability_m >= 0.5
            future_reward = tf.where(hard_stop_m, immediate_m, future_reward)
        future_reward = tf.stop_gradient(future_reward)

        immediate_n = discounts[date_n] * tf.nn.relu(K_tf - paths[:, date_n])
        state_n = tf.stack([paths[:, date_n], immediate_n], axis=1)

        with tf.GradientTape() as tape:
            soft_stop = tf.reshape(current_model(state_n, training=True), (-1,))
            soft_value = immediate_n * soft_stop + future_reward * (1.0 - soft_stop)
            objective = tf.reduce_mean(soft_value)
            loss = -objective

        gradients = tape.gradient(loss, current_model.trainable_variables)
        grads_and_vars = [
            (g, v)
            for g, v in zip(gradients, current_model.trainable_variables)
            if g is not None
        ]
        if not grads_and_vars:
            raise RuntimeError(f"No gradients were produced for exercise date {date_n}.")
        optimizer.apply_gradients(grads_and_vars)

        hard_probability_n = tf.reshape(
            current_model(state_n, training=False), (-1,)
        )
        hard_value = tf.where(
            hard_probability_n >= 0.5, immediate_n, future_reward
        )
        return objective, tf.reduce_mean(hard_value), tf.reduce_mean(soft_stop)

    return train_step

def train_or_resume_policy(
    cfg: ExperimentConfig,
    S0: float,
    sigma: float,
    case_dir: Path,
    models: Sequence[Optional[StoppingNetwork]],
    optimizers: Sequence[Optional[tf.keras.optimizers.Optimizer]],
    rng: tf.random.Generator,
    manager: tf.train.CheckpointManager,
    current_date_var: tf.Variable,
    date_step_var: tf.Variable,
    total_updates_var: tf.Variable,
    timing_path: Path,
) -> None:
    current_date = int(current_date_var.numpy())
    if current_date <= 0:
        print("[Training] Becker datewise recursion already complete.")
        return
    if current_date >= cfg.N:
        raise RuntimeError(f"Invalid checkpoint current_date={current_date}.")

    log_path = case_dir / "training_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    "exercise_date_index",
                    "step_within_date",
                    "total_update",
                    "soft_objective",
                    "hard_policy_batch_value",
                    "mean_soft_stop_probability",
                    "wall_time",
                ]
            )

    date_summary_path = case_dir / "datewise_training_summary.csv"
    if not date_summary_path.exists():
        with date_summary_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    "exercise_date_index",
                    "exercise_time",
                    "steps_completed",
                    "total_updates_after_date",
                    "completion_time",
                ]
            )

    total_required = (cfg.N - 1) * cfg.training_steps
    print(
        f"[Training] Becker datewise backward recursion for S0={S0:g}, "
        f"sigma={sigma:.2f}: dates {current_date},...,1; "
        f"{cfg.training_steps} steps/date, batch={cfg.training_batch_size}, "
        f"total required updates={total_required:,}."
    )

    # Networks later than current_date are completed future policies.
    for m in range(current_date + 1, cfg.N):
        model_m = models[m]
        if model_m is not None:
            model_m.trainable = False

    while int(current_date_var.numpy()) >= 1:
        n = int(current_date_var.numpy())
        model_n = models[n]
        optimizer_n = optimizers[n]
        assert model_n is not None and optimizer_n is not None
        model_n.trainable = True

        # All future networks are frozen and used only through hard decisions.
        for m in range(n + 1, cfg.N):
            model_m = models[m]
            if model_m is not None:
                model_m.trainable = False

        start_step = int(date_step_var.numpy())
        if not 0 <= start_step <= cfg.training_steps:
            raise RuntimeError(
                f"Invalid checkpoint date_step={start_step} for exercise date {n}."
            )

        print(
            f"[Training date {n:02d}/{cfg.N-1:02d}] "
            f"time={n * cfg.T / cfg.N:.6f}; "
            f"steps {start_step + 1}...{cfg.training_steps}"
        )
        train_step = make_date_training_step(
            cfg, S0, sigma, n, models, optimizer_n, rng
        )

        segment_start = time.perf_counter()
        for step in range(start_step, cfg.training_steps):
            soft_objective, hard_value, mean_soft_stop = train_step()
            date_step_var.assign(step + 1)
            total_updates_var.assign_add(1)

            should_log = (
                (step + 1) % cfg.training_log_every == 0
                or step == start_step
                or step + 1 == cfg.training_steps
            )
            should_checkpoint = (
                (step + 1) % cfg.checkpoint_every == 0
                or step + 1 == cfg.training_steps
            )

            if should_log:
                now = time.perf_counter()
                update_timing(timing_path, "training_seconds", now - segment_start)
                segment_start = now
                with log_path.open("a", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerow(
                        [
                            n,
                            step + 1,
                            int(total_updates_var.numpy()),
                            float(soft_objective.numpy()),
                            float(hard_value.numpy()),
                            float(mean_soft_stop.numpy()),
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                        ]
                    )
                print(
                    f"  date {n:02d}, step {step + 1:4d}/{cfg.training_steps}: "
                    f"objective={float(soft_objective.numpy()):.6f}, "
                    f"hard value={float(hard_value.numpy()):.6f}, "
                    f"mean stop={float(mean_soft_stop.numpy()):.4f}"
                )

            if should_checkpoint:
                manager.save(checkpoint_number=int(total_updates_var.numpy()))

        now = time.perf_counter()
        if now > segment_start:
            update_timing(timing_path, "training_seconds", now - segment_start)

        model_n.trainable = False
        with date_summary_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    n,
                    n * cfg.T / cfg.N,
                    cfg.training_steps,
                    int(total_updates_var.numpy()),
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )
        current_date_var.assign(n - 1)
        date_step_var.assign(0)
        manager.save(checkpoint_number=int(total_updates_var.numpy()))
        print(f"[Training date {n:02d}] complete and frozen.")
        gc.collect()

    print(
        f"[Training] Becker datewise recursion complete: "
        f"{int(total_updates_var.numpy()):,}/{total_required:,} updates."
    )

# Policy evaluation and lower bound

def make_policy_payoff_function(
    cfg: ExperimentConfig,
    S0: float,
    sigma: float,
    models: Sequence[Optional[StoppingNetwork]],
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    N = cfg.N
    dt = np.float32(cfg.T / N)
    mu_dt = tf.constant((cfg.r - cfg.q - 0.5 * sigma * sigma) * dt, tf.float32)
    vol_sqrt_dt = tf.constant(sigma * math.sqrt(float(dt)), tf.float32)
    discounts = tf.constant(
        np.exp(-cfg.r * np.linspace(0.0, cfg.T, N + 1)).astype(np.float32)
    )
    K_tf = tf.constant(cfg.K, tf.float32)
    S0_tf = tf.constant(S0, tf.float32)

    @tf.function(reduce_retracing=True)
    def simulate_and_evaluate(seed: tf.Tensor, n_paths: tf.Tensor) -> tf.Tensor:
        shape = tf.stack([tf.cast(n_paths, tf.int32), tf.constant(N, tf.int32)])
        eps = tf.random.stateless_normal(shape, seed=seed, dtype=tf.float32)
        log_paths = tf.math.log(S0_tf) + tf.cumsum(mu_dt + vol_sqrt_dt * eps, axis=1)
        n_paths_i = tf.cast(n_paths, tf.int32)
        start_shape = tf.stack([n_paths_i, tf.constant(1, tf.int32)])
        vector_shape = tf.stack([n_paths_i])
        paths = tf.concat(
            [tf.fill(start_shape, S0_tf), tf.exp(log_paths)], axis=1
        )

        alive = tf.ones(vector_shape, dtype=tf.bool)
        payoff = tf.zeros(vector_shape, dtype=tf.float32)

        for n in range(1, N):
            model = models[n]
            assert model is not None
            immediate = discounts[n] * tf.nn.relu(K_tf - paths[:, n])
            state_n = tf.stack([paths[:, n], immediate], axis=1)
            probability = tf.reshape(model(state_n, training=False), (-1,))
            stop = tf.logical_and(alive, probability >= 0.5)
            payoff = tf.where(stop, immediate, payoff)
            alive = tf.logical_and(alive, tf.logical_not(stop))

        maturity = discounts[N] * tf.nn.relu(K_tf - paths[:, N])
        payoff = tf.where(alive, maturity, payoff)
        return payoff

    return simulate_and_evaluate


def determine_f0(
    cfg: ExperimentConfig,
    S0: float,
    case_seed: int,
    policy_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    case_dir: Path,
) -> Tuple[int, float, float]:
    path = case_dir / "f0_decision.json"
    if path.exists():
        saved = json_load(path, {})
        return int(saved["f0"]), float(saved["immediate_payoff"]), float(
            saved["estimated_continuation_value"]
        )

    seed = tf.constant(seed_pair(case_seed, stream=31, index=0), dtype=tf.int32)
    payoffs = policy_fn(seed, tf.constant(cfg.f0_pilot_paths, dtype=tf.int32))
    continuation = float(tf.reduce_mean(payoffs).numpy())
    immediate = max(cfg.K - S0, 0.0)
    f0 = int(immediate >= continuation)
    json_dump(
        path,
        {
            "f0": f0,
            "rule": "stop at t0 iff immediate payoff >= independent MC continuation estimate",
            "pilot_paths": cfg.f0_pilot_paths,
            "immediate_payoff": immediate,
            "estimated_continuation_value": continuation,
        },
    )
    return f0, immediate, continuation


def open_or_create_nan_memmap(path: Path, shape: Tuple[int, ...], dtype: np.dtype) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        arr = np.lib.format.open_memmap(path, mode="r+")
        if tuple(arr.shape) != shape or arr.dtype != np.dtype(dtype):
            raise RuntimeError(f"Existing memmap {path} has incompatible shape/dtype.")
        return arr
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    arr[...] = np.nan
    arr.flush()
    return arr


def compute_lower_bound(
    cfg: ExperimentConfig,
    S0: float,
    case_seed: int,
    f0: int,
    immediate: float,
    policy_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    case_dir: Path,
    timing_path: Path,
) -> Tuple[float, float, float]:
    payoff_path = case_dir / "lower_bound_payoffs.npy"
    payoffs = open_or_create_nan_memmap(payoff_path, (cfg.K_L,), np.float32)

    if f0 == 1:
        payoffs[:] = np.float32(immediate)
        payoffs.flush()
    else:
        n_chunks = cfg.K_L // cfg.lower_chunk_size
        print(f"[Lower] K_L={cfg.K_L:,}, chunks={n_chunks}")
        for chunk_id in range(n_chunks):
            start = chunk_id * cfg.lower_chunk_size
            end = start + cfg.lower_chunk_size
            if np.all(np.isfinite(payoffs[start:end])):
                continue
            tick = time.perf_counter()
            seed = tf.constant(seed_pair(case_seed, stream=41, index=chunk_id), tf.int32)
            values = policy_fn(seed, tf.constant(cfg.lower_chunk_size, tf.int32)).numpy()
            payoffs[start:end] = values.astype(np.float32, copy=False)
            payoffs.flush()
            update_timing(timing_path, "lower_seconds", time.perf_counter() - tick)
            print(f"  lower chunk {chunk_id + 1:3d}/{n_chunks} complete")

    if not np.all(np.isfinite(payoffs)):
        raise RuntimeError("Lower-bound payoff file contains unfinished/invalid entries.")
    values64 = np.asarray(payoffs, dtype=np.float64)
    lower = float(np.mean(values64))
    lower_sd = float(np.std(values64, ddof=1)) if cfg.K_L > 1 else 0.0
    lower_se = lower_sd / math.sqrt(cfg.K_L)
    return lower, lower_sd, lower_se


# Becker dual upper bound, equations (17)--(18)

def make_outer_path_function(
    cfg: ExperimentConfig,
    S0: float,
    sigma: float,
) -> Callable[[tf.Tensor], tf.Tensor]:
    N = cfg.N
    B = cfg.upper_outer_batch_size
    dt = np.float32(cfg.T / N)
    mu_dt = tf.constant((cfg.r - cfg.q - 0.5 * sigma * sigma) * dt, tf.float32)
    vol_sqrt_dt = tf.constant(sigma * math.sqrt(float(dt)), tf.float32)
    S0_tf = tf.constant(S0, tf.float32)

    @tf.function(reduce_retracing=True)
    def simulate(seed: tf.Tensor) -> tf.Tensor:
        eps = tf.random.stateless_normal((B, N), seed=seed, dtype=tf.float32)
        log_paths = tf.math.log(S0_tf) + tf.cumsum(mu_dt + vol_sqrt_dt * eps, axis=1)
        return tf.concat([tf.fill((B, 1), S0_tf), tf.exp(log_paths)], axis=1)

    return simulate


def resolve_upper_inner_chunk_size(cfg: ExperimentConfig) -> int:
    raw = os.environ.get(UPPER_INNER_CHUNK_ENV, str(DEFAULT_UPPER_INNER_CHUNK_SIZE))
    try:
        requested = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{UPPER_INNER_CHUNK_ENV} must be a positive integer; got {raw!r}."
        ) from exc
    if requested <= 0:
        raise ValueError(f"{UPPER_INNER_CHUNK_ENV} must be positive.")
    chunk = min(requested, int(cfg.J))
    if cfg.J % chunk != 0:
        raise ValueError(
            f"J={cfg.J} must be divisible by the upper inner chunk size {chunk}. "
            f"Set {UPPER_INNER_CHUNK_ENV} to a divisor of J."
        )
    return chunk


def check_or_write_upper_stream_metadata(
    cfg: ExperimentConfig,
    case_dir: Path,
    inner_chunk_size: int,
) -> bool:
    metadata_path = case_dir / "upper_streamed_v2_metadata.json"
    metadata = {
        "implementation": UPPER_ESTIMATOR_VERSION,
        "seed_scheme": (
            "stateless nested normal; stream=61; "
            "index=outer_batch*n_inner_chunks+inner_chunk; reused across dates"
        ),
        "K_U": int(cfg.K_U),
        "J": int(cfg.J),
        "N": int(cfg.N),
        "upper_outer_batch_size": int(cfg.upper_outer_batch_size),
        "upper_inner_chunk_size": int(inner_chunk_size),
        "continuation_storage_dtype": "float32",
        "dual_storage_dtype": "float64",
    }
    if metadata_path.exists():
        old = json_load(metadata_path, {})
        if old != metadata:
            raise RuntimeError(
                "The saved streamed upper-bound workspace was created with different "
                "implementation settings. Keep the same "
                f"{UPPER_INNER_CHUNK_ENV}, or delete only these files and restart the "
                "upper calculation:\n"
                "  upper_streamed_v2_metadata.json\n"
                "  upper_outer_paths_streamed_v2.npy\n"
                "  upper_continuation_values_streamed_v2.npy\n"
                "  upper_bound_dual_payoffs_streamed_v2.npy"
            )
        return False
    json_dump(metadata_path, metadata)
    return True


def make_continuation_sum_function(
    cfg: ExperimentConfig,
    sigma: float,
    models: Sequence[Optional[StoppingNetwork]],
    start_n: int,
    inner_chunk_size: int,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    N = int(cfg.N)
    B = int(cfg.upper_outer_batch_size)
    H = int(inner_chunk_size)
    if not 0 <= start_n < N:
        raise ValueError(f"start_n must lie in 0,...,{N - 1}; got {start_n}.")

    dt = np.float32(cfg.T / N)
    mu_dt = tf.constant((cfg.r - cfg.q - 0.5 * sigma * sigma) * dt, tf.float32)
    vol_sqrt_dt = tf.constant(sigma * math.sqrt(float(dt)), tf.float32)
    discounts = tf.constant(
        np.exp(-cfg.r * np.linspace(0.0, cfg.T, N + 1)).astype(np.float32)
    )
    K_tf = tf.constant(cfg.K, tf.float32)
    matrix_shape = (B, H)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(B,), dtype=tf.float32),
            tf.TensorSpec(shape=(B, H, N), dtype=tf.float32),
        ],
        reduce_retracing=True,
    )
    def continuation_sum(z_n: tf.Tensor, nested_eps: tf.Tensor) -> tf.Tensor:
        s = tf.broadcast_to(z_n[:, None], matrix_shape)
        alive = tf.ones(matrix_shape, dtype=tf.bool)
        payoff = tf.zeros(matrix_shape, dtype=tf.float32)

        for m in range(start_n + 1, N + 1):
            s = s * tf.exp(mu_dt + vol_sqrt_dt * nested_eps[:, :, m - 1])
            reward = discounts[m] * tf.nn.relu(K_tf - s)
            if m < N:
                model = models[m]
                assert model is not None
                flat_s = tf.reshape(s, (-1,))
                flat_reward = tf.reshape(reward, (-1,))
                features = tf.stack([flat_s, flat_reward], axis=1)
                probability = tf.reshape(
                    model(features, training=False), matrix_shape
                )
                stop = tf.logical_and(alive, probability >= 0.5)
            else:
                stop = alive
            payoff = tf.where(stop, reward, payoff)
            alive = tf.logical_and(alive, tf.logical_not(stop))

        return tf.reduce_sum(payoff, axis=1)

    return continuation_sum


def evaluate_hard_policy_on_outer_paths(
    cfg: ExperimentConfig,
    outer_paths: tf.Tensor,
    models: Sequence[Optional[StoppingNetwork]],
) -> np.ndarray:
    B, N = cfg.upper_outer_batch_size, cfg.N
    decisions = np.zeros((B, N + 1), dtype=np.float64)
    for n in range(1, N):
        model = models[n]
        assert model is not None
        discount_n = tf.constant(math.exp(-cfg.r * (n * cfg.T / cfg.N)), tf.float32)
        immediate = discount_n * tf.nn.relu(
            tf.constant(cfg.K, tf.float32) - outer_paths[:, n]
        )
        features = tf.stack([outer_paths[:, n], immediate], axis=1)
        p = tf.reshape(model(features, training=False), (-1,)).numpy()
        decisions[:, n] = (p >= 0.5).astype(np.float64)
    decisions[:, N] = 1.0
    return decisions


def ensure_streamed_outer_paths(
    cfg: ExperimentConfig,
    S0: float,
    case_seed: int,
    outer_fn: Callable[[tf.Tensor], tf.Tensor],
    case_dir: Path,
) -> np.memmap:
    path = case_dir / "upper_outer_paths_streamed_v2.npy"
    outer = open_or_create_nan_memmap(
        path, (cfg.K_U, cfg.N + 1), np.float32
    )
    n_batches = cfg.K_U // cfg.upper_outer_batch_size
    for batch_id in range(n_batches):
        start = batch_id * cfg.upper_outer_batch_size
        end = start + cfg.upper_outer_batch_size
        if np.all(np.isfinite(outer[start:end, :])):
            continue
        seed = tf.constant(seed_pair(case_seed, stream=51, index=batch_id), tf.int32)
        values = outer_fn(seed).numpy().astype(np.float32, copy=False)
        outer[start:end, :] = values
        outer.flush()
    return outer


def compute_streamed_continuation_values(
    cfg: ExperimentConfig,
    sigma: float,
    case_seed: int,
    models: Sequence[Optional[StoppingNetwork]],
    outer_paths: np.memmap,
    case_dir: Path,
    timing_path: Path,
    inner_chunk_size: int,
) -> np.memmap:
    continuation_path = case_dir / "upper_continuation_values_streamed_v2.npy"
    continuation = open_or_create_nan_memmap(
        continuation_path, (cfg.K_U, cfg.N), np.float32
    )
    B = int(cfg.upper_outer_batch_size)
    N = int(cfg.N)
    n_batches = cfg.K_U // B
    n_inner_chunks = cfg.J // inner_chunk_size
    report_every = max(1, n_batches // 8)

    for start_n in range(N):
        if np.all(np.isfinite(continuation[:, start_n])):
            continue

        print(
            f"[Upper:C] date {start_n + 1:3d}/{N}, "
            f"inner chunks={n_inner_chunks}, chunk size={inner_chunk_size:,}"
        )
        continuation_sum_fn = make_continuation_sum_function(
            cfg, sigma, models, start_n, inner_chunk_size
        )

        for batch_id in range(n_batches):
            start = batch_id * B
            end = start + B
            if np.all(np.isfinite(continuation[start:end, start_n])):
                continue

            tick = time.perf_counter()
            z_n = tf.convert_to_tensor(
                np.asarray(outer_paths[start:end, start_n], dtype=np.float32),
                dtype=tf.float32,
            )
            payoff_sum = np.zeros(B, dtype=np.float64)

            for inner_id in range(n_inner_chunks):
                seed_index = batch_id * n_inner_chunks + inner_id
                nested_seed = tf.constant(
                    seed_pair(case_seed, stream=61, index=seed_index), tf.int32
                )
                nested_eps = tf.random.stateless_normal(
                    (B, inner_chunk_size, N),
                    seed=nested_seed,
                    dtype=tf.float32,
                )
                chunk_sum = continuation_sum_fn(z_n, nested_eps).numpy()
                payoff_sum += chunk_sum.astype(np.float64, copy=False)
                del nested_eps, chunk_sum

            continuation[start:end, start_n] = (
                payoff_sum / float(cfg.J)
            ).astype(np.float32)
            continuation.flush()
            update_timing(
                timing_path,
                "upper_continuation_seconds",
                time.perf_counter() - tick,
            )

            if (
                batch_id == 0
                or (batch_id + 1) % report_every == 0
                or batch_id + 1 == n_batches
            ):
                print(
                    f"  continuation date {start_n + 1:3d}/{N}, "
                    f"outer batch {batch_id + 1:3d}/{n_batches} complete"
                )

        # Release the large fixed-date graph before tracing the next date.
        del continuation_sum_fn
        gc.collect()
        print(f"[Upper:C] date {start_n + 1:3d}/{N} complete")

    if not np.all(np.isfinite(continuation)):
        raise RuntimeError(
            "Streamed continuation-value file contains unfinished/invalid entries."
        )
    return continuation


def compute_dual_from_streamed_values(
    cfg: ExperimentConfig,
    models: Sequence[Optional[StoppingNetwork]],
    outer_paths: np.memmap,
    continuation: np.memmap,
    case_dir: Path,
    timing_path: Path,
) -> np.memmap:
    dual_path = case_dir / "upper_bound_dual_payoffs_streamed_v2.npy"
    dual = open_or_create_nan_memmap(dual_path, (cfg.K_U,), np.float64)
    B, N = cfg.upper_outer_batch_size, cfg.N
    n_batches = cfg.K_U // B
    times = np.linspace(0.0, cfg.T, N + 1)
    discounts = np.exp(-cfg.r * times)

    print(f"[Upper:Dual] assembling {n_batches} outer batches")
    for batch_id in range(n_batches):
        start = batch_id * B
        end = start + B
        if np.all(np.isfinite(dual[start:end])):
            continue

        tick = time.perf_counter()
        outer_np = np.asarray(outer_paths[start:end, :], dtype=np.float64)
        outer_tf = tf.convert_to_tensor(
            np.asarray(outer_paths[start:end, :], dtype=np.float32),
            dtype=tf.float32,
        )
        C = np.asarray(continuation[start:end, :], dtype=np.float64)
        g = discounts[None, :] * np.maximum(cfg.K - outer_np, 0.0)
        f = evaluate_hard_policy_on_outer_paths(cfg, outer_tf, models)

        delta_M = np.empty((B, N), dtype=np.float64)
        for n in range(1, N):
            delta_M[:, n - 1] = (
                f[:, n] * g[:, n]
                + (1.0 - f[:, n]) * C[:, n]
                - C[:, n - 1]
            )
        delta_M[:, N - 1] = g[:, N] - C[:, N - 1]

        M = np.zeros((B, N + 1), dtype=np.float64)
        M[:, 1:] = np.cumsum(delta_M, axis=1)
        dual[start:end] = np.max(g - M, axis=1)
        dual.flush()
        update_timing(timing_path, "upper_dual_seconds", time.perf_counter() - tick)
        print(f"  upper dual batch {batch_id + 1:3d}/{n_batches} complete")

        del outer_tf, outer_np, C, g, f, delta_M, M
        gc.collect()

    if not np.all(np.isfinite(dual)):
        raise RuntimeError("Upper-bound dual payoff file contains unfinished/invalid entries.")
    return dual


def compute_upper_bound(
    cfg: ExperimentConfig,
    S0: float,
    sigma: float,
    case_seed: int,
    models: Sequence[Optional[StoppingNetwork]],
    case_dir: Path,
    timing_path: Path,
) -> Tuple[float, float, float]:
    inner_chunk_size = resolve_upper_inner_chunk_size(cfg)
    new_workspace = check_or_write_upper_stream_metadata(
        cfg, case_dir, inner_chunk_size
    )
    if new_workspace:
        # Do not mix wall time from an earlier failed legacy upper-bound attempt
        # with the new streamed implementation. Training/lower timings are retained.
        timing = json_load(timing_path, {})
        for key in (
            "upper_seconds",
            "upper_continuation_seconds",
            "upper_dual_seconds",
        ):
            timing.pop(key, None)
        json_dump(timing_path, timing)

    n_batches = cfg.K_U // cfg.upper_outer_batch_size
    print(
        f"[Upper] K_U={cfg.K_U:,}, J={cfg.J:,}, outer batches={n_batches}, "
        f"inner chunk={inner_chunk_size:,}."
    )
    print(
        "[Upper] Streamed date-major implementation: one continuation graph and "
        "one inner chunk are held in memory at a time."
    )

    outer_fn = make_outer_path_function(cfg, S0, sigma)
    outer_paths = ensure_streamed_outer_paths(
        cfg, S0, case_seed, outer_fn, case_dir
    )
    del outer_fn
    gc.collect()

    tick = time.perf_counter()
    continuation = compute_streamed_continuation_values(
        cfg,
        sigma,
        case_seed,
        models,
        outer_paths,
        case_dir,
        timing_path,
        inner_chunk_size,
    )
    dual = compute_dual_from_streamed_values(
        cfg,
        models,
        outer_paths,
        continuation,
        case_dir,
        timing_path,
    )
    update_timing(timing_path, "upper_seconds", time.perf_counter() - tick)

    dual64 = np.asarray(dual, dtype=np.float64)
    upper = float(np.mean(dual64))
    upper_sd = float(np.std(dual64, ddof=1)) if cfg.K_U > 1 else 0.0
    upper_se = upper_sd / math.sqrt(cfg.K_U)
    return upper, upper_sd, upper_se


# Learned boundary, stopping-region diagnostics and figures

def threshold_projection_boundary(stock_grid: np.ndarray, hard: np.ndarray) -> Tuple[float, float]:
    hard_i = hard.astype(np.int64)
    n = hard_i.size
    # Candidate k=-1 means no stopping; candidate k=n-1 means stop everywhere.
    prefix_false = np.concatenate([[0], np.cumsum(1 - hard_i)])
    suffix_true = np.concatenate([np.cumsum(hard_i[::-1])[::-1], [0]])
    mismatches = np.empty(n + 1, dtype=np.int64)
    for count_stopped in range(n + 1):
        mismatches[count_stopped] = prefix_false[count_stopped] + suffix_true[count_stopped]
    best_count = int(np.argmin(mismatches))
    if best_count == 0:
        threshold = float("nan")
    else:
        threshold = float(stock_grid[best_count - 1])
    change_rate = float(mismatches[best_count] / n)
    return threshold, change_rate




def main_figure_boundary_display(
    raw_boundary: np.ndarray,
    beta: float,
    K: float,
) -> np.ndarray:
    displayed = np.asarray(raw_boundary, dtype=np.float64).copy()
    invalid = (~np.isfinite(displayed)) | (displayed < beta) | (displayed > K + 1.0e-10)
    displayed[invalid] = np.nan
    return displayed


def extract_learned_boundary_and_region(
    cfg: ExperimentConfig,
    sigma: float,
    peskir: PeskirBoundaryResult,
    models: Sequence[Optional[StoppingNetwork]],
    case_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta = peskir.beta
    stock_grid = np.linspace(
        max(1.0e-6, cfg.boundary_stock_min_multiple_beta * beta),
        cfg.boundary_stock_max_multiple_K * cfg.K,
        cfg.boundary_stock_grid_size,
        dtype=np.float32,
    )
    times = np.linspace(0.0, cfg.T, cfg.N + 1)
    probabilities = np.full((cfg.N + 1, stock_grid.size), np.nan, dtype=np.float32)
    hard_region = np.zeros((cfg.N + 1, stock_grid.size), dtype=np.uint8)
    learned_boundary = np.full(cfg.N + 1, np.nan, dtype=np.float64)
    projection_change = np.full(cfg.N + 1, np.nan, dtype=np.float64)

    stock_tf = tf.constant(stock_grid, dtype=tf.float32)
    itm_mask = stock_grid <= cfg.K
    for n in range(1, cfg.N):
        model = models[n]
        assert model is not None
        discount_n = np.float32(math.exp(-cfg.r * times[n]))
        reward_tf = discount_n * tf.nn.relu(tf.constant(cfg.K, tf.float32) - stock_tf)
        features = tf.stack([stock_tf, reward_tf], axis=1)
        p = tf.reshape(model(features, training=False), (-1,)).numpy().astype(np.float32)
        hard = (p >= 0.5) & itm_mask
        probabilities[n] = p
        hard_region[n] = hard.astype(np.uint8)
        threshold, change_rate = threshold_projection_boundary(stock_grid[itm_mask], hard[itm_mask])
        learned_boundary[n] = threshold
        projection_change[n] = change_rate

    # At maturity, exercise for S<=K.
    probabilities[cfg.N] = (stock_grid <= cfg.K).astype(np.float32)
    hard_region[cfg.N] = (stock_grid <= cfg.K).astype(np.uint8)
    learned_boundary[cfg.N] = cfg.K
    projection_change[cfg.N] = 0.0

    display_boundary = main_figure_boundary_display(
        learned_boundary, peskir.beta, cfg.K
    )
    theo_at_exercise = np.interp(times, peskir.times, peskir.boundary)
    frame = pd.DataFrame(
        {
            "time_index": np.arange(cfg.N + 1),
            "time": times,
            "theoretical_continuous_boundary": theo_at_exercise,
            "learned_threshold_projected_boundary": learned_boundary,
            "main_figure_display_boundary": display_boundary,
            "projection_change_rate": projection_change,
        }
    )
    frame.to_csv(case_dir / "boundary_comparison.csv", index=False)
    np.save(case_dir / "learned_stop_probabilities.npy", probabilities)
    np.save(case_dir / "learned_hard_stopping_region.npy", hard_region)
    np.save(case_dir / "stopping_region_stock_grid.npy", stock_grid)
    return times, learned_boundary, display_boundary, stock_grid.astype(np.float64), hard_region


def select_hitting_sample_path(
    cfg: ExperimentConfig,
    S0: float,
    sigma: float,
    peskir: PeskirBoundaryResult,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    steps = cfg.sample_path_steps
    t = np.linspace(0.0, cfg.T, steps + 1)
    b = np.interp(t, peskir.times, peskir.boundary)
    dt = cfg.T / steps
    rng = np.random.default_rng(seed)
    batch = 250
    best_path: Optional[np.ndarray] = None
    best_gap = float("inf")
    best_index = steps

    searched = 0
    while searched < cfg.sample_path_search_paths:
        current_batch = min(batch, cfg.sample_path_search_paths - searched)
        eps = rng.standard_normal((current_batch, steps))
        log_inc = (cfg.r - cfg.q - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * eps
        log_s = math.log(S0) + np.cumsum(log_inc, axis=1)
        paths = np.concatenate(
            [np.full((current_batch, 1), S0), np.exp(log_s)], axis=1
        )
        hits = paths <= b[None, :]
        for row in range(current_batch):
            idxs = np.flatnonzero(hits[row])
            if idxs.size:
                idx = int(idxs[0])
                frac = idx / steps
                if 0.12 <= frac <= 0.88:
                    return t, paths[row], idx
                if best_path is None:
                    best_path, best_index = paths[row].copy(), idx
            gap = np.min(paths[row] - b)
            if gap < best_gap:
                best_gap = float(gap)
                best_path = paths[row].copy()
                best_index = int(np.argmin(paths[row] - b))
        searched += current_batch

    assert best_path is not None
    genuine = np.flatnonzero(best_path <= b)
    if genuine.size:
        best_index = int(genuine[0])
    return t, best_path, best_index


def plot_peskir_style_comparison(
    cfg: ExperimentConfig,
    S0: float,
    sigma: float,
    peskir: PeskirBoundaryResult,
    exercise_times: np.ndarray,
    learned_boundary: np.ndarray,
    case_dir: Path,
    seed: int,
) -> None:
    path_t, path_s, hit_idx = select_hitting_sample_path(cfg, S0, sigma, peskir, seed)
    theoretical_on_path = np.interp(path_t, peskir.times, peskir.boundary)
    genuine_hit = bool(path_s[hit_idx] <= theoretical_on_path[hit_idx])
    tau = float(path_t[hit_idx])

    beta, K, T = peskir.beta, cfg.K, cfg.T
    spread = K - beta
    axis_y = beta - 0.13 * spread
    y_min = axis_y - 0.05 * spread
    displayed_path = path_s[: hit_idx + 1]
    y_max = max(K + 0.16 * spread, float(np.max(displayed_path)) + 0.05 * spread)

    fig, ax = plt.subplots(figsize=(12.8, 8.6))
    ax.set_xlim(-0.035 * T, 1.085 * T)
    ax.set_ylim(y_min, y_max)

    # Horizontal K and beta levels.
    ax.plot([0.0, T], [K, K], linewidth=1.2, color="black")
    ax.plot([0.0, T], [beta, beta], linewidth=1.0, color="black")

    # Continuous theoretical boundary and discrete learned boundary.
    ax.plot(
        peskir.times,
        peskir.boundary,
        linewidth=3.2,
        color="black",
        label=r"Peskir--Shiryaev $b(t)$",
        zorder=4,
    )
    valid = np.isfinite(learned_boundary)
    ax.plot(
        exercise_times[valid],
        learned_boundary[valid],
        linestyle="--",
        linewidth=2.0,
        color="tab:blue",
        label=r"learned boundary $\widehat b_n$",
        zorder=5,
    )

    # Sample path stops at the first theoretical-boundary hit, as in Figure VII.1.
    ax.plot(path_t[: hit_idx + 1], path_s[: hit_idx + 1], linewidth=1.0, color="black")
    ax.plot([tau, tau], [axis_y, theoretical_on_path[hit_idx]], "--", linewidth=1.0, color="black")
    ax.plot([T, T], [axis_y, K], "--", linewidth=1.0, color="black")

    # Arrow axes.
    ax.annotate(
        "",
        xy=(1.075 * T, axis_y),
        xytext=(-0.035 * T, axis_y),
        arrowprops=dict(arrowstyle="-|>", mutation_scale=22, linewidth=1.0, color="black"),
    )
    ax.annotate(
        "",
        xy=(0.0, y_max),
        xytext=(0.0, y_min),
        arrowprops=dict(arrowstyle="-|>", mutation_scale=22, linewidth=1.0, color="black"),
    )

    # Book-style labels.
    ax.text(-0.012 * T, K, r"$K$", fontsize=22, ha="right", va="center")
    ax.text(-0.012 * T, beta, r"$\beta$", fontsize=22, ha="right", va="center")
    ax.text(-0.012 * T, S0, r"$x$", fontsize=20, ha="right", va="center")
    ax.text(T, axis_y - 0.035 * spread, r"$T$", fontsize=20, ha="center", va="top")
    tau_label = r"$\tau_b$" if genuine_hit else r"$t_{\mathrm{closest}}$"
    ax.text(tau, axis_y - 0.035 * spread, tau_label, fontsize=20, ha="center", va="top")

    path_label_index = max(1, int(0.42 * (hit_idx + 1)))
    ax.text(
        path_t[path_label_index],
        path_s[path_label_index] - 0.10 * spread,
        r"$t\mapsto X_t$",
        fontsize=19,
        ha="center",
    )
    theory_label_t = 0.70 * T
    theory_label_y = float(np.interp(theory_label_t, peskir.times, peskir.boundary)) + 0.08 * spread
    ax.text(theory_label_t, theory_label_y, r"$t\mapsto b(t)$", fontsize=19, ha="center")

    ax.legend(loc="upper right", frameon=False, fontsize=12)
    ax.set_title(
        f"American put boundary comparison: $S_0={S0:g}$, $\\sigma={sigma:.2f}$",
        fontsize=15,
        pad=16,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(case_dir / "peskir_style_boundary_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(case_dir / "peskir_style_boundary_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_raw_stopping_region(
    cfg: ExperimentConfig,
    sigma: float,
    peskir: PeskirBoundaryResult,
    exercise_times: np.ndarray,
    learned_boundary: np.ndarray,
    stock_grid: np.ndarray,
    hard_region: np.ndarray,
    case_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    # Transpose so stock is vertical. Values 1/0 are stop/continue.
    mesh = ax.pcolormesh(
        exercise_times,
        stock_grid,
        hard_region.T,
        shading="nearest",
        cmap="Greys",
        vmin=0,
        vmax=1,
    )
    ax.plot(
        peskir.times,
        peskir.boundary,
        linewidth=2.4,
        color="tab:red",
        label="continuous theoretical boundary",
    )
    valid = np.isfinite(learned_boundary)
    ax.plot(
        exercise_times[valid],
        learned_boundary[valid],
        linestyle="--",
        linewidth=1.8,
        color="tab:blue",
        label="threshold projection of network region",
    )
    ax.axhline(cfg.K, linewidth=1.0, color="black", label="$K$")
    ax.set_xlabel("time")
    ax.set_ylabel("stock price")
    ax.set_title("Raw neural-network stopping region (dark = stop)")
    ax.legend(frameon=False)
    colorbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["continue", "stop"])
    fig.tight_layout()
    fig.savefig(case_dir / "raw_learned_stopping_region.png", dpi=300, bbox_inches="tight")
    fig.savefig(case_dir / "raw_learned_stopping_region.pdf", bbox_inches="tight")
    plt.close(fig)

# One case and main orchestration

def run_case(
    cfg: ExperimentConfig,
    case_index: int,
    S0: float,
    sigma: float,
    peskir: PeskirBoundaryResult,
    root: Path,
    force_retrain: bool,
    skip_upper: bool,
) -> Dict[str, Any]:
    tag = case_tag(S0, sigma)
    case_dir = root / "cases" / tag
    if force_retrain and case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    signature = config_signature(cfg, S0, sigma)
    check_or_write_signature(case_dir, signature)
    timing_path = case_dir / "timing_state.json"
    case_seed = cfg.base_seed + 10_000 * (case_index + 1)

    print("\n" + "=" * 90)
    print(f"CASE {case_index}: S0={S0:g}, sigma={sigma:.2f} ({tag})")
    print("=" * 90)

    tf.keras.backend.clear_session()
    gc.collect()
    set_global_seeds(case_seed)

    models, optimizers = make_models_and_optimizers(cfg, S0)
    train_rng = tf.random.Generator.from_seed(case_seed + 701)
    (
        _,
        manager,
        current_date_var,
        date_step_var,
        total_updates_var,
    ) = make_checkpoint(cfg, case_dir, models, optimizers, train_rng)
    train_or_resume_policy(
        cfg,
        S0,
        sigma,
        case_dir,
        models,
        optimizers,
        train_rng,
        manager,
        current_date_var,
        date_step_var,
        total_updates_var,
        timing_path,
    )

    policy_fn = make_policy_payoff_function(cfg, S0, sigma, models)
    f0, immediate, f0_continuation = determine_f0(
        cfg, S0, case_seed, policy_fn, case_dir
    )
    print(
        f"[f0] immediate={immediate:.8f}, continuation={f0_continuation:.8f}, f0={f0}"
    )

    lower, lower_sd, lower_se = compute_lower_bound(
        cfg,
        S0,
        case_seed,
        f0,
        immediate,
        policy_fn,
        case_dir,
        timing_path,
    )
    print(f"[Lower] L={lower:.8f}, SE={lower_se:.8f}")

    (
        exercise_times,
        learned_boundary,
        display_boundary,
        stock_grid,
        hard_region,
    ) = extract_learned_boundary_and_region(cfg, sigma, peskir, models, case_dir)
    plot_peskir_style_comparison(
        cfg,
        S0,
        sigma,
        peskir,
        exercise_times,
        display_boundary,
        case_dir,
        seed=case_seed + 909,
    )
    plot_raw_stopping_region(
        cfg,
        sigma,
        peskir,
        exercise_times,
        learned_boundary,
        stock_grid,
        hard_region,
        case_dir,
    )

    true_value = peskir_value_from_boundary(
        S0, cfg, sigma, peskir.times, peskir.boundary
    )

    timing = json_load(timing_path, {})
    partial_result: Dict[str, Any] = {
        "case_index": case_index,
        "S0": S0,
        "sigma": sigma,
        "K": cfg.K,
        "r": cfg.r,
        "q": cfg.q,
        "T": cfg.T,
        "N": cfg.N,
        "independent_policy_training": True,
        "training_steps": cfg.training_steps,
        "training_algorithm": "Becker datewise backward recursion with frozen future hard policies",
        "training_steps_per_date": cfg.training_steps,
        "total_gradient_updates": (cfg.N - 1) * cfg.training_steps,
        "training_batch_size": cfg.training_batch_size,
        "K_L": cfg.K_L,
        "K_U": cfg.K_U,
        "J": cfg.J,
        "upper_estimator_implementation": UPPER_ESTIMATOR_VERSION,
        "upper_inner_chunk_size": resolve_upper_inner_chunk_size(cfg),
        "upper_outer_batch_size": cfg.upper_outer_batch_size,
        "f0": f0,
        "f0_immediate": immediate,
        "f0_continuation_estimate": f0_continuation,
        "lower_bound": lower,
        "lower_sample_sd": lower_sd,
        "lower_standard_error": lower_se,
        "peskir_true_value": true_value,
        "peskir_beta": peskir.beta,
        "peskir_boundary_at_zero": float(peskir.boundary[0]),
        "peskir_terminal_integral_validation_error": peskir.terminal_integral_max_abs_error,
        "peskir_coarse_fine_value_difference": float(
            peskir.coarse_fine_value_differences.get(f"S0_{int(S0)}", float("nan"))
        ),
        "literature_LS": LITERATURE_LS[(S0, sigma)],
        "training_seconds": float(timing.get("training_seconds", 0.0)),
        "lower_seconds": float(timing.get("lower_seconds", 0.0)),
        "upper_seconds": float(timing.get("upper_seconds", 0.0)),
        "status": "lower_and_figures_complete",
    }
    json_dump(case_dir / "case_result_partial.json", partial_result)

    if skip_upper:
        print("[Upper] Skipped by --skip-upper; this is not a complete Section 4.1 result.")
        return partial_result

    upper, upper_sd, upper_se = compute_upper_bound(
        cfg,
        S0,
        sigma,
        case_seed,
        models,
        case_dir,
        timing_path,
    )
    print(f"[Upper] U={upper:.8f}, SE={upper_se:.8f}")

    z = normal_quantile_975(cfg.confidence_level)
    ci_low = lower - z * lower_se
    ci_high = upper + z * upper_se
    point = 0.5 * (lower + upper)
    timing = json_load(timing_path, {})

    result = dict(partial_result)
    result.update(
        {
            "upper_bound": upper,
            "upper_sample_sd": upper_sd,
            "upper_standard_error": upper_se,
            "point_estimate": point,
            "gap": upper - lower,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "point_minus_peskir_true": point - true_value,
            "lower_minus_peskir_true": lower - true_value,
            "upper_minus_peskir_true": upper - true_value,
            "training_seconds": float(timing.get("training_seconds", 0.0)),
            "lower_seconds": float(timing.get("lower_seconds", 0.0)),
            "upper_seconds": float(timing.get("upper_seconds", 0.0)),
            "t_L_training_plus_lower": float(
                timing.get("training_seconds", 0.0) + timing.get("lower_seconds", 0.0)
            ),
            "status": "complete",
        }
    )
    json_dump(case_dir / "case_result.json", result)
    print(
        f"[Result] point={point:.8f}, gap={upper-lower:.8f}, "
        f"95% CI=[{ci_low:.8f}, {ci_high:.8f}], "
        f"Peskir true={true_value:.8f}"
    )

    del models, optimizers, policy_fn
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def write_master_results(root: Path, results: Sequence[Dict[str, Any]]) -> None:
    if not results:
        return
    frame = pd.DataFrame(results).sort_values(["sigma", "S0"])
    preferred = [
        "case_index",
        "S0",
        "sigma",
        "K",
        "r",
        "q",
        "T",
        "N",
        "lower_bound",
        "upper_bound",
        "point_estimate",
        "gap",
        "ci_low",
        "ci_high",
        "peskir_true_value",
        "point_minus_peskir_true",
        "literature_LS",
        "f0",
        "training_seconds",
        "lower_seconds",
        "upper_seconds",
        "t_L_training_plus_lower",
        "status",
    ]
    columns = [c for c in preferred if c in frame.columns] + [
        c for c in frame.columns if c not in preferred
    ]
    frame = frame[columns]
    frame.to_csv(root / "section_4_1_results.csv", index=False)
    try:
        frame.to_excel(root / "section_4_1_results.xlsx", index=False)
    except ImportError:
        print("[Output] openpyxl is not installed; CSV was written but XLSX was skipped.")


def _engine_main_unused() -> None:
    args = _engine_parse_args_unused()
    cfg = ExperimentConfig()
    if args.quick_test:
        cfg = _engine_quick_config_unused(cfg)
    ensure_divisibility(cfg)
    set_global_seeds(cfg.base_seed)
    configure_tensorflow(cfg.require_gpu)

    root = Path(cfg.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_dump(root / "experiment_configuration.json", asdict(cfg))

    if args.case_index is None:
        selected = list(enumerate(CASES))
    else:
        if not 0 <= args.case_index < len(CASES):
            raise SystemExit("--case-index must be one of 0,1,2,3,4,5.")
        selected = [(args.case_index, CASES[args.case_index])]

    needed_sigmas = sorted({sigma for _, (_, sigma) in selected})
    peskir_results: Dict[float, PeskirBoundaryResult] = {}
    for sigma in needed_sigmas:
        peskir_results[sigma] = load_or_compute_peskir_boundary(
            cfg, sigma, root, force=args.force_peskir
        )

    results: List[Dict[str, Any]] = []
    for case_index, (S0, sigma) in selected:
        result = run_case(
            cfg,
            case_index,
            S0,
            sigma,
            peskir_results[sigma],
            root,
            force_retrain=args.force_retrain,
            skip_upper=args.skip_upper,
        )
        results.append(result)
        # Include already completed non-selected cases in the master table below.
        write_master_results(root, results)

    # Rebuild the master table from every completed case folder, not just this invocation.
    all_results: List[Dict[str, Any]] = []
    for S0, sigma in CASES:
        folder = root / "cases" / case_tag(S0, sigma)
        complete = folder / "case_result.json"
        partial = folder / "case_result_partial.json"
        if complete.exists():
            all_results.append(json_load(complete, {}))
        elif partial.exists():
            all_results.append(json_load(partial, {}))
    write_master_results(root, all_results)

    print("\nFinished.")
    print("Output root:", root)
    print("Master CSV:", root / "section_4_1_results.csv")
    print("Each case folder contains checkpoints, payoff arrays, boundary CSVs and figures.")

# FIVE-SEED SINGLE-CASE SENSITIVITY DRIVER

from types import ModuleType

DEFAULT_OUTPUT_ROOT = "outputs_4_1_single_case_sensitivity_five_seed"
DEFAULT_N_VALUES: Tuple[int, ...] = (50, 100, 150)
DEFAULT_BATCH_VALUES: Tuple[int, ...] = (2048, 4096, 8192, 16384)
DEFAULT_REPEAT_SEEDS: Tuple[int, ...] = (
    20260711,
    20261711,
    20262711,
    20263711,
    20264711,
)
BASELINE_BATCH = 8192
SINGLE_CASE_INDEX = 1  # In the validated engine: (S0=40, sigma=0.20)
EXPECTED_CASE = (40.0, 0.20)
AGGREGATE_FOLDER = "aggregate_5_seeds"
DRIVER_VERSION = "single_file_single_case_N_batch_five_seed_sensitivity_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=("all", "n-grid", "batch"),
        default="all",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--n-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_N_VALUES),
    )
    parser.add_argument(
        "--batch-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_VALUES),
    )
    parser.add_argument(
        "--repeat-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_REPEAT_SEEDS),

    )
    parser.add_argument(
        "--quick-test",
        action="store_true",

    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",

    )
    parser.add_argument(
        "--force-peskir",
        action="store_true",
    )
    parser.add_argument(
        "--skip-upper",
        action="store_true",
    )
    return parser.parse_args()


def validate_repeat_seeds(values: Sequence[int]) -> Tuple[int, ...]:
    seeds = tuple(int(x) for x in values)
    if len(seeds) != 5:
        raise ValueError(
            f"Exactly five repeat seeds are required; received {len(seeds)}: {seeds}."
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"The five repeat seeds must be distinct; received {seeds}.")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Every repeat seed must be non-negative.")
    return seeds


def full_config(
    engine: ModuleType,
    *,
    N: int,
    batch_size: int,
    output_dir: Path,
    base_seed: Optional[int] = None,
) -> Any:
    cfg = engine.ExperimentConfig()
    cfg = replace(
        cfg,
        algorithm_version=DRIVER_VERSION,
        N=int(N),
        training_batch_size=int(batch_size),
        base_seed=int(cfg.base_seed if base_seed is None else base_seed),
        output_dir=str(output_dir.resolve()),
    )
    engine.ensure_divisibility(cfg)
    return cfg


def quick_config(cfg: Any, batch_label: int) -> Any:
    quick_batch = max(32, int(batch_label) // 32)
    return replace(
        cfg,
        training_steps=2,
        training_batch_size=quick_batch,
        training_log_every=1,
        checkpoint_every=1,
        K_L=1024,
        lower_chunk_size=256,
        K_U=4,
        J=32,
        upper_outer_batch_size=1,
        f0_pilot_paths=512,
        peskir_time_steps=160,
        peskir_coarse_time_steps=80,
        peskir_terminal_z_nodes=256,
        boundary_stock_grid_size=401,
        sample_path_steps=200,
        sample_path_search_paths=500,
        require_gpu=False,
    )


def case_item(engine: ModuleType) -> Tuple[int, Tuple[float, float]]:
    return SINGLE_CASE_INDEX, engine.CASES[SINGLE_CASE_INDEX]


def case_directory(engine: ModuleType, root: Path) -> Path:
    _, (S0, sigma) = case_item(engine)
    return root / "cases" / engine.case_tag(S0, sigma)


def read_saved_result(
    engine: ModuleType,
    root: Path,
    *,
    require_complete: bool,
) -> Optional[Dict[str, Any]]:
    case_dir = case_directory(engine, root)
    complete = case_dir / "case_result.json"
    partial = case_dir / "case_result_partial.json"
    if complete.exists():
        return engine.json_load(complete, {})
    if not require_complete and partial.exists():
        return engine.json_load(partial, {})
    return None


def seed_boundary_artifacts_exist(engine: ModuleType, seed_root: Path) -> bool:
    case_dir = case_directory(engine, seed_root)
    required = (
        case_dir / "boundary_comparison.csv",
        case_dir / "learned_stop_probabilities.npy",
        case_dir / "learned_hard_stopping_region.npy",
        case_dir / "stopping_region_stock_grid.npy",
    )
    return all(path.exists() for path in required)


def safe_to_excel(frame: pd.DataFrame, path: Path) -> None:
    try:
        frame.to_excel(path, index=False)
    except ImportError:
        print(f"[Output] openpyxl not installed; skipped {path.name}.")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        return
    frame.to_csv(path, index=False)
    safe_to_excel(frame, path.with_suffix(".xlsx"))


def shared_peskir_result(
    engine: ModuleType,
    cfg: Any,
    shared_root: Path,
    force: bool,
) -> Dict[float, Any]:
    shared_root.mkdir(parents=True, exist_ok=True)
    engine.json_dump(shared_root / "peskir_configuration.json", asdict(cfg))
    sigma = EXPECTED_CASE[1]
    return {
        sigma: engine.load_or_compute_peskir_boundary(
            cfg, sigma, shared_root, force=force
        )
    }


def run_one_configuration(
    engine: ModuleType,
    cfg: Any,
    root: Path,
    peskir_results: Dict[float, Any],
    *,
    force_retrain: bool,
    skip_upper: bool,
    experiment: str,
    requested_batch_size: int,
) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    engine.ensure_divisibility(cfg)
    engine.json_dump(root / "experiment_configuration.json", asdict(cfg))

    idx, (S0, sigma) = case_item(engine)
    start = time.perf_counter()
    result = engine.run_case(
        cfg,
        idx,
        S0,
        sigma,
        peskir_results[float(sigma)],
        root,
        force_retrain=force_retrain,
        skip_upper=skip_upper,
    )
    wall_seconds = time.perf_counter() - start

    engine.write_master_results(root, [result])
    row = dict(result)
    row.update(
        {
            "experiment": experiment,
            "requested_training_batch_size": int(requested_batch_size),
            "driver_wall_seconds_this_call": float(wall_seconds),
            "result_root": str(root.resolve()),
        }
    )
    print(
        f"[Completed seed run] N={cfg.N}, requested B={requested_batch_size}, "
        f"base seed={cfg.base_seed}; this call took {wall_seconds / 3600.0:.3f} hours."
    )
    return row


def nan_column_mean_sd(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values64 = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values64)
    counts = np.sum(valid, axis=0).astype(np.int64)

    sums = np.sum(np.where(valid, values64, 0.0), axis=0, dtype=np.float64)
    means = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)

    centered = np.where(valid, values64 - means[None, :], 0.0)
    squared = np.sum(centered * centered, axis=0, dtype=np.float64)
    sds = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(squared, counts - 1, out=sds, where=counts > 1)
    sds = np.sqrt(sds)
    sds[counts == 1] = 0.0
    return means, sds, counts


def summarise_repeat_results(
    frame: pd.DataFrame,
    *,
    experiment: str,
    requested_batch_size: int,
    repeat_seeds: Sequence[int],
    config_root: Path,
) -> Dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot summarise an empty repeated-results table.")

    summary: Dict[str, Any] = {
        "experiment": experiment,
        "n_repeats": int(len(frame)),
        "repeat_seeds": ",".join(str(int(seed)) for seed in repeat_seeds),
        "requested_training_batch_size": int(requested_batch_size),
        "configuration_root": str(config_root.resolve()),
    }

    fixed_columns = (
        "case_index",
        "S0",
        "sigma",
        "K",
        "r",
        "q",
        "T",
        "N",
        "independent_policy_training",
        "training_steps",
        "training_algorithm",
        "training_steps_per_date",
        "total_gradient_updates",
        "training_batch_size",
        "K_L",
        "K_U",
        "J",
        "upper_estimator_implementation",
        "upper_inner_chunk_size",
        "upper_outer_batch_size",
        "peskir_true_value",
        "peskir_beta",
        "peskir_boundary_at_zero",
        "peskir_terminal_integral_validation_error",
        "peskir_coarse_fine_value_difference",
        "literature_LS",
    )
    for column in fixed_columns:
        if column in frame.columns:
            summary[column] = frame.iloc[0][column]

    excluded_numeric = {
        "case_index",
        "repeat_index",
        "repeat_base_seed",
        "effective_case_seed",
        "requested_training_batch_size",
    }
    fixed_numeric = {
        col
        for col in fixed_columns
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col])
    }

    for column in frame.columns:
        if column in excluded_numeric or column in fixed_numeric:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        summary[f"{column}_mean"] = float(numeric.mean())
        summary[f"{column}_std"] = (
            float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
        )

    for metric in ("lower_bound", "upper_bound", "point_estimate"):
        if metric not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if len(numeric) < 2:
            continue
        sample_sd = float(numeric.std(ddof=1))
        sem = sample_sd / math.sqrt(len(numeric))
        from scipy.stats import t as student_t

        critical = float(student_t.ppf(0.975, df=len(numeric) - 1))
        mean = float(numeric.mean())
        summary[f"{metric}_across_seed_sem"] = sem
        summary[f"{metric}_across_seed_ci95_low"] = mean - critical * sem
        summary[f"{metric}_across_seed_ci95_high"] = mean + critical * sem

    summary["driver_wall_seconds_this_call_sum"] = float(
        pd.to_numeric(
            frame.get("driver_wall_seconds_this_call", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
    )
    summary["n_new_seed_runs_this_call"] = int(
        (frame.get("result_source", pd.Series(dtype=str)) == "new_run").sum()
    )
    summary["n_reused_seed_runs_this_call"] = int(
        (frame.get("result_source", pd.Series(dtype=str)) == "reused_saved_result").sum()
    )
    return summary


def numerical_summary_long(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        rows.append(
            {
                "metric": column,
                "n": int(len(numeric)),
                "mean": float(numeric.mean()),
                "sample_std": float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
                "minimum": float(numeric.min()),
                "maximum": float(numeric.max()),
            }
        )
    return pd.DataFrame(rows)


def build_average_stopping_region_outputs(
    engine: ModuleType,
    cfg: Any,
    config_root: Path,
    peskir: Any,
    repeat_seeds: Sequence[int],
) -> Path:
    _, (S0, sigma) = case_item(engine)
    tag = engine.case_tag(S0, sigma)

    boundaries: List[np.ndarray] = []
    hard_regions: List[np.ndarray] = []
    stop_probabilities: List[np.ndarray] = []
    seed_labels: List[str] = []
    reference_times: Optional[np.ndarray] = None
    reference_stock_grid: Optional[np.ndarray] = None
    theoretical_at_dates: Optional[np.ndarray] = None

    for seed in repeat_seeds:
        case_dir = config_root / f"seed_{int(seed)}" / "cases" / tag
        boundary_path = case_dir / "boundary_comparison.csv"
        hard_path = case_dir / "learned_hard_stopping_region.npy"
        probability_path = case_dir / "learned_stop_probabilities.npy"
        stock_path = case_dir / "stopping_region_stock_grid.npy"
        for required in (boundary_path, hard_path, probability_path, stock_path):
            if not required.exists():
                raise FileNotFoundError(
                    f"Cannot aggregate five seeds because {required} is missing."
                )

        boundary_frame = pd.read_csv(boundary_path)
        times = boundary_frame["time"].to_numpy(dtype=np.float64)
        boundary = boundary_frame[
            "learned_threshold_projected_boundary"
        ].to_numpy(dtype=np.float64)
        theoretical = boundary_frame[
            "theoretical_continuous_boundary"
        ].to_numpy(dtype=np.float64)
        stock_grid = np.load(stock_path).astype(np.float64)
        hard = np.load(hard_path).astype(np.float64)
        probabilities = np.load(probability_path).astype(np.float64)

        if reference_times is None:
            reference_times = times
            reference_stock_grid = stock_grid
            theoretical_at_dates = theoretical
        else:
            if not np.array_equal(times, reference_times):
                raise RuntimeError("Seed-level exercise-time grids do not match.")
            if not np.array_equal(stock_grid, reference_stock_grid):
                raise RuntimeError("Seed-level stopping-region stock grids do not match.")
            if hard.shape != hard_regions[0].shape:
                raise RuntimeError("Seed-level hard stopping-region shapes do not match.")
            if probabilities.shape != stop_probabilities[0].shape:
                raise RuntimeError("Seed-level stopping-probability shapes do not match.")

        boundaries.append(boundary)
        hard_regions.append(hard)
        stop_probabilities.append(probabilities)
        seed_labels.append(f"seed_{int(seed)}")

    assert reference_times is not None
    assert reference_stock_grid is not None
    assert theoretical_at_dates is not None

    boundary_matrix = np.vstack(boundaries)
    mean_boundary, sd_boundary, boundary_counts = nan_column_mean_sd(boundary_matrix)
    display_mean_boundary = engine.main_figure_boundary_display(
        mean_boundary, peskir.beta, cfg.K
    )

    hard_stack = np.stack(hard_regions, axis=0)
    mean_hard_region = np.mean(hard_stack, axis=0, dtype=np.float64)

    probability_stack = np.stack(stop_probabilities, axis=0)
    probability_valid = np.isfinite(probability_stack)
    probability_counts = np.sum(probability_valid, axis=0)
    probability_sums = np.sum(
        np.where(probability_valid, probability_stack, 0.0),
        axis=0,
        dtype=np.float64,
    )
    mean_stop_probability = np.full(probability_counts.shape, np.nan, dtype=np.float64)
    np.divide(
        probability_sums,
        probability_counts,
        out=mean_stop_probability,
        where=probability_counts > 0,
    )

    aggregate_dir = config_root / AGGREGATE_FOLDER
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    boundary_output = pd.DataFrame(
        {
            "time_index": np.arange(len(reference_times)),
            "time": reference_times,
            "theoretical_continuous_boundary": theoretical_at_dates,
            "mean_learned_boundary": mean_boundary,
            "sample_sd_learned_boundary": sd_boundary,
            "number_of_finite_seed_boundaries": boundary_counts,
            "main_figure_display_mean_boundary": display_mean_boundary,
        }
    )
    boundary_output.to_csv(
        aggregate_dir / "average_boundary_comparison.csv", index=False
    )

    individual_boundary_output = pd.DataFrame(
        {"time": reference_times, **{
            label: boundary_matrix[index]
            for index, label in enumerate(seed_labels)
        }}
    )
    individual_boundary_output.to_csv(
        aggregate_dir / "individual_seed_boundaries.csv", index=False
    )

    np.save(aggregate_dir / "average_learned_boundary.npy", mean_boundary)
    np.save(aggregate_dir / "sample_sd_learned_boundary.npy", sd_boundary)
    np.save(aggregate_dir / "average_hard_stopping_region.npy", mean_hard_region)
    np.save(aggregate_dir / "average_stop_probability.npy", mean_stop_probability)
    np.save(aggregate_dir / "stopping_region_stock_grid.npy", reference_stock_grid)

    # Mean learned boundary, individual seed boundaries, and theoretical boundary.
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    for index, label in enumerate(seed_labels):
        valid_seed = np.isfinite(boundary_matrix[index])
        ax.plot(
            reference_times[valid_seed],
            boundary_matrix[index, valid_seed],
            linewidth=0.9,
            alpha=0.35,
            color="0.45",
            label="individual seed boundary" if index == 0 else None,
        )
    ax.plot(
        peskir.times,
        peskir.boundary,
        linewidth=2.5,
        color="tab:red",
        label="continuous theoretical boundary",
    )
    valid_mean = np.isfinite(display_mean_boundary)
    ax.plot(
        reference_times[valid_mean],
        display_mean_boundary[valid_mean],
        linestyle="--",
        linewidth=2.2,
        color="tab:blue",
        label="mean learned boundary (5 seeds)",
    )
    valid_band = (
        np.isfinite(mean_boundary)
        & np.isfinite(sd_boundary)
        & (boundary_counts >= 2)
    )
    if np.any(valid_band):
        ax.fill_between(
            reference_times[valid_band],
            (mean_boundary - sd_boundary)[valid_band],
            (mean_boundary + sd_boundary)[valid_band],
            color="tab:blue",
            alpha=0.16,
            label=r"mean learned boundary $\pm 1$ sample SD",
        )
    ax.axhline(cfg.K, linewidth=1.0, color="black", label="$K$")
    ax.set_xlabel("time")
    ax.set_ylabel("stock price")
    ax.set_title(
        f"Mean learned exercise boundary over five seeds: "
        f"$N={cfg.N}$, requested $B={cfg.training_batch_size}$"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        aggregate_dir / "average_boundary_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        aggregate_dir / "average_boundary_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # The heatmap value is the fraction of the five hard policies choosing stop.
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    mesh = ax.pcolormesh(
        reference_times,
        reference_stock_grid,
        mean_hard_region.T,
        shading="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.plot(
        peskir.times,
        peskir.boundary,
        linewidth=2.3,
        color="tab:red",
        label="continuous theoretical boundary",
    )
    ax.plot(
        reference_times[valid_mean],
        display_mean_boundary[valid_mean],
        linestyle="--",
        linewidth=2.0,
        color="tab:blue",
        label="mean learned boundary",
    )
    ax.axhline(cfg.K, linewidth=1.0, color="black", label="$K$")
    ax.set_xlabel("time")
    ax.set_ylabel("stock price")
    ax.set_title("Average hard stopping region over five random seeds")
    ax.legend(frameon=False)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("fraction of seeds choosing stop")
    fig.tight_layout()
    fig.savefig(
        aggregate_dir / "average_stopping_region_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        aggregate_dir / "average_stopping_region_heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Also retain the direct mean of the five network stop probabilities.
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    probability_mesh = ax.pcolormesh(
        reference_times,
        reference_stock_grid,
        mean_stop_probability.T,
        shading="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.plot(
        peskir.times,
        peskir.boundary,
        linewidth=2.3,
        color="tab:red",
        label="continuous theoretical boundary",
    )
    ax.plot(
        reference_times[valid_mean],
        display_mean_boundary[valid_mean],
        linestyle="--",
        linewidth=2.0,
        color="tab:blue",
        label="mean learned boundary",
    )
    ax.axhline(cfg.K, linewidth=1.0, color="black", label="$K$")
    ax.set_xlabel("time")
    ax.set_ylabel("stock price")
    ax.set_title("Mean neural-network stop probability over five random seeds")
    ax.legend(frameon=False)
    probability_colorbar = fig.colorbar(probability_mesh, ax=ax)
    probability_colorbar.set_label("mean network stop probability")
    fig.tight_layout()
    fig.savefig(
        aggregate_dir / "average_stop_probability_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        aggregate_dir / "average_stop_probability_heatmap.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    engine.json_dump(
        aggregate_dir / "aggregation_manifest.json",
        {
            "aggregation_version": DRIVER_VERSION,
            "number_of_seeds": len(repeat_seeds),
            "repeat_seeds": [int(seed) for seed in repeat_seeds],
            "mean_boundary_definition": (
                "At every exercise date, arithmetic mean of the finite threshold-"
                "projected learned boundaries extracted independently from the five seeds."
            ),
            "boundary_sd_definition": (
                "Sample standard deviation across finite seed-level learned boundaries."
            ),
            "average_hard_region_definition": (
                "At every (time, stock) grid point, fraction of the five hard policies "
                "whose stop decision equals one."
            ),
            "average_stop_probability_definition": (
                "At every (time, stock) grid point, arithmetic mean of the finite raw "
                "network stop probabilities."
            ),
        },
    )
    return aggregate_dir


def run_repeated_configuration(
    engine: ModuleType,
    base_cfg: Any,
    config_root: Path,
    peskir_results: Dict[float, Any],
    *,
    force_retrain: bool,
    skip_upper: bool,
    experiment: str,
    requested_batch_size: int,
    repeat_seeds: Sequence[int],
) -> Dict[str, Any]:
    config_root.mkdir(parents=True, exist_ok=True)
    engine.json_dump(
        config_root / "five_seed_configuration_template.json",
        {
            "driver_version": DRIVER_VERSION,
            "repeat_seeds": [int(seed) for seed in repeat_seeds],
            "base_configuration": asdict(base_cfg),
        },
    )

    rows: List[Dict[str, Any]] = []
    require_complete = not skip_upper
    for repeat_index, repeat_seed in enumerate(repeat_seeds, start=1):
        seed_root = config_root / f"seed_{int(repeat_seed)}"
        seed_cfg = replace(
            base_cfg,
            base_seed=int(repeat_seed),
            output_dir=str(seed_root.resolve()),
        )
        engine.ensure_divisibility(seed_cfg)

        saved = None
        if not force_retrain and seed_boundary_artifacts_exist(engine, seed_root):
            saved = read_saved_result(
                engine,
                seed_root,
                require_complete=require_complete,
            )

        print("\n" + "*" * 100)
        print(
            f"REPEAT {repeat_index}/5: N={seed_cfg.N}, requested B={requested_batch_size}, "
            f"actual B={seed_cfg.training_batch_size}, base seed={repeat_seed}"
        )
        print("*" * 100)

        if saved is not None:
            row = dict(saved)
            row.update(
                {
                    "experiment": experiment,
                    "requested_training_batch_size": int(requested_batch_size),
                    "driver_wall_seconds_this_call": 0.0,
                    "result_root": str(seed_root.resolve()),
                    "result_source": "reused_saved_result",
                }
            )
            print(
                f"[Reuse] Complete seed output already exists for base seed {repeat_seed}."
            )
        else:
            row = run_one_configuration(
                engine,
                seed_cfg,
                seed_root,
                peskir_results,
                force_retrain=force_retrain,
                skip_upper=skip_upper,
                experiment=experiment,
                requested_batch_size=requested_batch_size,
            )
            row["result_source"] = "new_run"

        effective_case_seed = int(
            repeat_seed + 10_000 * (SINGLE_CASE_INDEX + 1)
        )
        row.update(
            {
                "repeat_index": int(repeat_index),
                "repeat_base_seed": int(repeat_seed),
                "effective_case_seed": effective_case_seed,
                "configuration_root": str(config_root.resolve()),
            }
        )
        rows.append(row)
        engine.json_dump(seed_root / "repeat_driver_result.json", row)
        write_table(
            pd.DataFrame(rows).sort_values("repeat_index"),
            config_root / "all_5_seed_results.csv",
        )

    repeat_frame = pd.DataFrame(rows).sort_values("repeat_index").reset_index(drop=True)
    write_table(repeat_frame, config_root / "all_5_seed_results.csv")

    long_summary = numerical_summary_long(repeat_frame)
    write_table(long_summary, config_root / "five_seed_numerical_summary_long.csv")

    aggregate_dir = build_average_stopping_region_outputs(
        engine=engine,
        cfg=base_cfg,
        config_root=config_root,
        peskir=peskir_results[EXPECTED_CASE[1]],
        repeat_seeds=repeat_seeds,
    )

    summary_row = summarise_repeat_results(
        repeat_frame,
        experiment=experiment,
        requested_batch_size=requested_batch_size,
        repeat_seeds=repeat_seeds,
        config_root=config_root,
    )
    summary_row["aggregate_output_root"] = str(aggregate_dir.resolve())
    summary_frame = pd.DataFrame([summary_row])
    write_table(summary_frame, config_root / "five_seed_summary_wide.csv")
    engine.json_dump(config_root / "five_seed_summary.json", summary_row)

    print(
        f"[Five-seed aggregation complete] N={base_cfg.N}, "
        f"requested B={requested_batch_size}; summary: "
        f"{config_root / 'five_seed_summary_wide.csv'}"
    )
    return summary_row


def get_or_run_n100_baseline(
    engine: ModuleType,
    args: argparse.Namespace,
    output_root: Path,
    peskir_results: Dict[float, Any],
    repeat_seeds: Sequence[int],
    baseline_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    cache_key = "N100_B8192_five_seed"
    if cache_key in baseline_cache:
        print(
            "[Reuse within invocation] The five-seed N=100, B=8192 baseline "
            "has already been processed."
        )
        return dict(baseline_cache[cache_key])

    root = output_root / "exercise_grid_sensitivity" / "N_100_B_8192"
    cfg = full_config(
        engine,
        N=100,
        batch_size=BASELINE_BATCH,
        output_dir=root,
        base_seed=repeat_seeds[0],
    )
    if args.quick_test:
        cfg = replace(
            quick_config(cfg, BASELINE_BATCH),
            output_dir=str(root.resolve()),
        )

    row = run_repeated_configuration(
        engine,
        cfg,
        root,
        peskir_results,
        force_retrain=args.force_retrain,
        skip_upper=args.skip_upper,
        experiment="shared_N100_B8192_baseline",
        requested_batch_size=BASELINE_BATCH,
        repeat_seeds=repeat_seeds,
    )
    row["configuration_result_source"] = "shared_five_seed_baseline"
    baseline_cache[cache_key] = dict(row)
    return row


def run_n_grid_study(
    engine: ModuleType,
    args: argparse.Namespace,
    output_root: Path,
    peskir_results: Dict[float, Any],
    repeat_seeds: Sequence[int],
    baseline_cache: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n_values = sorted(set(int(x) for x in args.n_values))
    if any(N <= 1 for N in n_values):
        raise ValueError("Every N must be greater than 1.")

    for N in n_values:
        if N == 100:
            row = get_or_run_n100_baseline(
                engine,
                args,
                output_root,
                peskir_results,
                repeat_seeds,
                baseline_cache,
            )
            row = dict(row)
            row["experiment"] = "exercise_grid_sensitivity"
            rows.append(row)
            continue

        root = output_root / "exercise_grid_sensitivity" / f"N_{N}_B_8192"
        cfg = full_config(
            engine,
            N=N,
            batch_size=BASELINE_BATCH,
            output_dir=root,
            base_seed=repeat_seeds[0],
        )
        if args.quick_test:
            cfg = replace(
                quick_config(cfg, BASELINE_BATCH),
                output_dir=str(root.resolve()),
            )

        print("\n" + "#" * 100)
        print(
            f"FIVE-SEED EXERCISE-GRID STUDY: S0=40, sigma=0.20, N={N}, "
            f"requested B={BASELINE_BATCH}, actual B={cfg.training_batch_size}"
        )
        print("#" * 100)
        row = run_repeated_configuration(
            engine,
            cfg,
            root,
            peskir_results,
            force_retrain=args.force_retrain,
            skip_upper=args.skip_upper,
            experiment="exercise_grid_sensitivity",
            requested_batch_size=BASELINE_BATCH,
            repeat_seeds=repeat_seeds,
        )
        row["configuration_result_source"] = "five_seed_configuration"
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty and "N" in frame.columns:
        frame = frame.sort_values("N").reset_index(drop=True)
    write_table(
        frame,
        output_root
        / "exercise_grid_sensitivity"
        / "N_50_100_150_five_seed_summary.csv",
    )
    return frame


def run_batch_study(
    engine: ModuleType,
    args: argparse.Namespace,
    output_root: Path,
    peskir_results: Dict[float, Any],
    repeat_seeds: Sequence[int],
    baseline_cache: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    batch_values = sorted(set(int(x) for x in args.batch_values))
    if any(B <= 0 for B in batch_values):
        raise ValueError("Every training path count must be positive.")
    if BASELINE_BATCH not in batch_values:
        batch_values.append(BASELINE_BATCH)
        batch_values.sort()

    for batch_size in batch_values:
        if batch_size == BASELINE_BATCH:
            row = get_or_run_n100_baseline(
                engine,
                args,
                output_root,
                peskir_results,
                repeat_seeds,
                baseline_cache,
            )
            row = dict(row)
            row["experiment"] = "training_path_sensitivity"
            row["requested_training_batch_size"] = BASELINE_BATCH
            rows.append(row)
            continue

        root = (
            output_root
            / "training_path_sensitivity"
            / "N_100"
            / f"B_{batch_size}"
        )
        cfg = full_config(
            engine,
            N=100,
            batch_size=batch_size,
            output_dir=root,
            base_seed=repeat_seeds[0],
        )
        if args.quick_test:
            cfg = replace(
                quick_config(cfg, batch_size),
                output_dir=str(root.resolve()),
            )

        print("\n" + "#" * 100)
        print(
            f"FIVE-SEED TRAINING-PATH STUDY: S0=40, sigma=0.20, N=100, "
            f"requested B={batch_size}, actual B={cfg.training_batch_size}"
        )
        print("#" * 100)
        row = run_repeated_configuration(
            engine,
            cfg,
            root,
            peskir_results,
            force_retrain=args.force_retrain,
            skip_upper=args.skip_upper,
            experiment="training_path_sensitivity",
            requested_batch_size=batch_size,
            repeat_seeds=repeat_seeds,
        )
        row["configuration_result_source"] = "five_seed_configuration"
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("requested_training_batch_size").reset_index(
            drop=True
        )
    write_table(
        frame,
        output_root
        / "training_path_sensitivity"
        / "N100_B_2048_4096_8192_16384_five_seed_summary.csv",
    )
    return frame


def collect_all_seed_results(output_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(output_root.rglob("all_5_seed_results.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "source_table", str(path.resolve()))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    write_table(
        combined,
        output_root / "all_configurations_all_seed_results.csv",
    )
    return combined


def write_design_manifest(
    engine: ModuleType,
    args: argparse.Namespace,
    output_root: Path,
    repeat_seeds: Sequence[int],
) -> None:
    cfg = engine.ExperimentConfig()
    manifest = {
        "driver_version": DRIVER_VERSION,
        "single_file": str(Path(__file__).resolve()),
        "case": {
            "case_index": SINGLE_CASE_INDEX,
            "S0": EXPECTED_CASE[0],
            "sigma": EXPECTED_CASE[1],
            "dimension": int(cfg.dimension),
            "K": float(cfg.K),
            "r": float(cfg.r),
            "q": float(cfg.q),
            "T": float(cfg.T),
        },
        "repeat_design": {
            "number_of_independent_runs_per_configuration": 5,
            "base_seeds": [int(seed) for seed in repeat_seeds],
            "complete_seed_folders_retained": True,
            "numerical_summary": "mean and sample standard deviation across five seeds",
            "mean_boundary": (
                "pointwise arithmetic mean of the five threshold-projected learned boundaries"
            ),
            "mean_hard_stopping_region": (
                "pointwise fraction of the five hard policies choosing stop"
            ),
            "mean_stop_probability": (
                "pointwise arithmetic mean of the five raw network stop probabilities"
            ),
        },
        "exercise_grid_design": {
            "N_values": [int(x) for x in args.n_values],
            "training_batch_size": BASELINE_BATCH,
        },
        "training_path_design": {
            "N": 100,
            "training_batch_sizes": [int(x) for x in args.batch_values],
        },
        "fixed_full_run_settings": {
            "training_steps_per_date": int(cfg.training_steps),
            "K_L": int(cfg.K_L),
            "K_U": int(cfg.K_U),
            "J": int(cfg.J),
        },
        "N100_B8192_five_seed_configuration_reused_between_studies": True,
        "quick_test": bool(args.quick_test),
        "skip_upper": bool(args.skip_upper),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "experiment_design_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    args = parse_args()
    engine = sys.modules[__name__]
    repeat_seeds = validate_repeat_seeds(args.repeat_seeds)

    output_root = args.output_root.expanduser().resolve()
    if args.quick_test:
        output_root = output_root.with_name(output_root.name + "_QUICK_TEST")
    output_root.mkdir(parents=True, exist_ok=True)
    write_design_manifest(engine, args, output_root, repeat_seeds)

    base_cfg = full_config(
        engine,
        N=100,
        batch_size=BASELINE_BATCH,
        output_dir=output_root,
        base_seed=repeat_seeds[0],
    )
    if args.quick_test:
        base_cfg = replace(
            quick_config(base_cfg, BASELINE_BATCH),
            output_dir=str(output_root.resolve()),
        )

    engine.set_global_seeds(repeat_seeds[0])
    engine.configure_tensorflow(base_cfg.require_gpu)

    if args.quick_test:
        print(
            "\nQUICK TEST MODE: all five seeds are retained, but training steps, "
            "batch sizes, K_L, K_U and J are tiny. Do not use these prices in the dissertation."
        )
    else:
        print(
            "\nFULL FIVE-SEED MODE: each unique parameter configuration is trained "
            "five times independently. With the default design there are six unique "
            "configurations and thirty complete seed-level runs."
        )
    print("Repeat base seeds:", repeat_seeds)

    peskir_results = shared_peskir_result(
        engine,
        base_cfg,
        output_root / "shared_continuous_time_benchmark",
        force=args.force_peskir,
    )

    total_start = time.perf_counter()
    n_frame = pd.DataFrame()
    batch_frame = pd.DataFrame()
    baseline_cache: Dict[str, Dict[str, Any]] = {}

    if args.experiment in ("all", "n-grid"):
        n_frame = run_n_grid_study(
            engine,
            args,
            output_root,
            peskir_results,
            repeat_seeds,
            baseline_cache,
        )

    if args.experiment in ("all", "batch"):
        batch_frame = run_batch_study(
            engine,
            args,
            output_root,
            peskir_results,
            repeat_seeds,
            baseline_cache,
        )

    combined_parts: List[pd.DataFrame] = []
    if not n_frame.empty:
        combined_parts.append(n_frame)
    if not batch_frame.empty:
        combined_parts.append(batch_frame)
    if combined_parts:
        combined = pd.concat(combined_parts, ignore_index=True, sort=False)
        write_table(
            combined,
            output_root / "all_five_seed_sensitivity_summary.csv",
        )

    all_seed_frame = collect_all_seed_results(output_root)
    elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 100)
    print("ALL REQUESTED FIVE-SEED SINGLE-CASE EXPERIMENTS FINISHED")
    print("=" * 100)
    print("Output root:", output_root)
    print(f"Total wall time in this invocation: {elapsed / 3600.0:.3f} hours")
    print(f"Seed-level result rows collected: {len(all_seed_frame)}")
    if not n_frame.empty:
        print(
            "N-study five-seed summary:",
            output_root
            / "exercise_grid_sensitivity"
            / "N_50_100_150_five_seed_summary.csv",
        )
    if not batch_frame.empty:
        print(
            "Path-study five-seed summary:",
            output_root
            / "training_path_sensitivity"
            / "N100_B_2048_4096_8192_16384_five_seed_summary.csv",
        )
    print(
        "The N=100, B=8192 five-seed configuration is stored once under "
        "exercise_grid_sensitivity/N_100_B_8192 and reused by both study tables."
    )
    print(
        "Every configuration folder contains five complete seed folders, raw and "
        "summary tables, and aggregate_5_seeds boundary/region figures."
    )


if __name__ == "__main__":
    main()
