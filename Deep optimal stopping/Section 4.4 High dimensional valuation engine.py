from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import erfinv, ndtr

try:
    import tensorflow as tf
except ImportError as exc: 
    raise SystemExit(
        "TensorFlow is unavailable. Use the TensorFlow already installed in the "
        "AutoDL image; do not reinstall it blindly."
    ) from exc


ALGORITHM_VERSION = "high_dim_basket_becker_frozen_future_v1"
UPPER_VERSION = "streamed_date_major_inner_resume_v1"
TRAINING_SEEDS: Tuple[int, ...] = (202601, 202602, 202603)
DIMENSIONS: Tuple[int, ...] = (1, 2, 3, 5, 10, 20, 30, 50)


@dataclass(frozen=True)
class ExperimentConfig:
    S0: float = 40.0
    K: float = 40.0
    r: float = 0.06
    q: float = 0.0
    sigma: float = 0.20
    rho: float = 0.0
    T: float = 1.0
    N: int = 50

    training_batch_size: int = 8_192
    learning_rate: float = 1.0e-3
    batch_norm_momentum: float = 0.99
    training_log_every: int = 50
    checkpoint_every: int = 100
    training_steps_override: Optional[int] = None

    K_L: int = 4_096_000
    K_U: int = 1_024
    J: int = 16_384
    f0_pilot_paths: int = 262_144

    evaluation_seed: int = 20261201
    max_common_dimension: int = 50
    confidence_level: float = 0.95

    output_dir: str = "/root/autodl-tmp/autodl_high_dim_becker/outputs"
    require_gpu: bool = True

    peskir_time_steps: int = 4_000
    peskir_coarse_time_steps: int = 2_000
    peskir_terminal_z_nodes: int = 4_096
    peskir_root_xtol: float = 1.0e-11
    run_peskir_coarse_check: bool = True

    def hidden_width(self, dimension: int) -> int:
        return int(dimension + 40)

    def training_steps(self, dimension: int) -> int:
        if self.training_steps_override is not None:
            return int(self.training_steps_override)
        return int(3_000 + dimension)


def quick_config(cfg: ExperimentConfig) -> ExperimentConfig:
    return replace(
        cfg,
        N=5,
        training_batch_size=128,
        training_log_every=1,
        checkpoint_every=1,
        training_steps_override=3,
        K_L=1_024,
        K_U=4,
        J=16,
        f0_pilot_paths=512,
        peskir_time_steps=160,
        peskir_coarse_time_steps=80,
        peskir_terminal_z_nodes=256,
        run_peskir_coarse_check=False,
        output_dir=str(Path(cfg.output_dir).with_name("outputs_QUICK_TEST")),
        require_gpu=False,
    )


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=_json_default)
    os.replace(tmp, path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
            pass
    print(f"[Environment] TensorFlow {tf.__version__}")
    print(f"[Environment] GPUs: {gpus}")
    if require_gpu and not gpus:
        raise RuntimeError("No TensorFlow GPU detected; refusing to start the full run.")
    return gpus


def update_timing(path: Path, key: str, seconds: float) -> Dict[str, float]:
    timing = json_load(path, {})
    timing[key] = float(timing.get(key, 0.0) + seconds)
    json_dump(path, timing)
    return timing


def case_dir(root: Path, dimension: int, training_seed: int) -> Path:
    return root / f"d_{dimension:03d}" / f"seed_{training_seed}"


def config_signature(cfg: ExperimentConfig, dimension: int, training_seed: int) -> Dict[str, Any]:
    payload = asdict(cfg)
    payload.update(
        {
            "algorithm_version": ALGORITHM_VERSION,
            "upper_version": UPPER_VERSION,
            "dimension": int(dimension),
            "training_seed": int(training_seed),
            "hidden_width": cfg.hidden_width(dimension),
            "training_steps": cfg.training_steps(dimension),
            "payoff": "discounted equal-weight arithmetic-average basket put",
            "network_input": "(S_1/K,...,S_d/K,g/K)",
        }
    )
    return payload


def check_or_write_signature(folder: Path, signature: Dict[str, Any]) -> None:
    path = folder / "configuration_signature.json"
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    wrapped = {"sha256": digest, "configuration": signature}
    if path.exists():
        old = json_load(path, {})
        if old != wrapped:
            raise RuntimeError(
                f"Configuration mismatch in {folder}. Use a new output folder or "
                "delete this case deliberately before rerunning."
            )
    else:
        json_dump(path, wrapped)


def seed_pair(base_seed: int, stream: int, block: int, asset: int = 0) -> np.ndarray:
    modulus = 2_147_483_647
    a = np.int32((base_seed + 104_729 * stream + 15_485_863 * asset) % modulus)
    b = np.int32((block + 1_000_003 * stream + 32_452_843 * asset + 97) % modulus)
    return np.asarray([a, b], dtype=np.int32)


def evaluation_seed_matrix(
    cfg: ExperimentConfig, stream: int, block: int, dimension: int
) -> np.ndarray:
    return np.stack(
        [seed_pair(cfg.evaluation_seed, stream, block, asset=i) for i in range(dimension)],
        axis=0,
    )


def confidence_z(cfg: ExperimentConfig) -> float:
    alpha = 1.0 - cfg.confidence_level
    return float(math.sqrt(2.0) * erfinv(1.0 - alpha))


def resolve_lower_chunk_size(cfg: ExperimentConfig, dimension: int) -> int:
    override = os.environ.get("BECKER_LOWER_CHUNK_SIZE")
    if override:
        chunk = int(override)
    elif cfg.max_common_dimension <= 30:
        chunk = 16_000
    else:
        chunk = 8_000
    if chunk <= 0:
        raise ValueError("Lower chunk must be positive.")
    if cfg.K_L % chunk != 0:
        candidates = [8_000, 4_000, 2_000, 1_000, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        valid = [c for c in candidates if c <= cfg.K_L and cfg.K_L % c == 0]
        if not valid:
            raise ValueError(f"No valid lower chunk divisor found for K_L={cfg.K_L}.")
        chunk = valid[0]
    return chunk


def _gpu_memory_bytes() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        return int(float(out)) * 1024 * 1024
    except Exception:
        return None


def resolve_upper_workspace(cfg: ExperimentConfig, dimension: int) -> Tuple[int, int, Dict[str, Any]]:
    b_override = os.environ.get("BECKER_UPPER_OUTER_BATCH_SIZE")
    h_override = os.environ.get("BECKER_UPPER_INNER_CHUNK_SIZE")
    gpu_bytes = _gpu_memory_bytes()
    target = min(2 * 1024**3, int(0.05 * gpu_bytes)) if gpu_bytes else 768 * 1024**2

    outer_candidates = [8, 4, 2, 1]
    inner_candidates = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    if b_override:
        outer_candidates = [int(b_override)]
    if h_override:
        inner_candidates = [int(h_override)]

    selected: Optional[Tuple[int, int, int]] = None
    for b in outer_candidates:
        if b <= 0 or cfg.K_U % b != 0:
            continue
        for h in inner_candidates:
            if h <= 0 or cfg.J % h != 0:
                continue
            estimated = int(b * h * cfg.N * cfg.max_common_dimension * 4 * 3.0)
            if estimated <= target:
                selected = (b, h, estimated)
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError(
            "Could not choose a valid upper workspace. Set environment variables "
            "BECKER_UPPER_OUTER_BATCH_SIZE and BECKER_UPPER_INNER_CHUNK_SIZE "
            "to divisors of K_U and J."
        )
    b, h, estimated = selected
    meta = {
        "gpu_memory_bytes_detected": gpu_bytes,
        "target_workspace_bytes": target,
        "estimated_peak_workspace_bytes": estimated,
        "outer_batch_size": b,
        "inner_chunk_size": h,
        "scientific_note": "workspace partition only; K_U and J are unchanged",
    }
    return b, h, meta


# Peskir--Shiryaev benchmark for d=1

@dataclass
class PeskirBoundaryResult:
    times: np.ndarray
    boundary: np.ndarray
    beta: float
    h: float
    value_at_S0: float
    terminal_integral_max_abs_error: float
    coarse_fine_value_difference: Optional[float]
    computation_seconds: float


def european_put_closed_form(S: float, K: float, r: float, sigma: float, tau: float) -> float:
    if tau <= 0.0:
        return max(K - S, 0.0)
    st = math.sqrt(tau)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * st)
    d2 = d1 - sigma * st
    return float(K * math.exp(-r * tau) * ndtr(-d2) - S * ndtr(-d1))


def perpetual_put_boundary_beta(K: float, r: float, sigma: float) -> float:
    return float(K / (1.0 + 0.5 * sigma * sigma / r))


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
    cfg: ExperimentConfig, time_steps: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    K, r, T, sigma = cfg.K, cfg.r, cfg.T, cfg.sigma
    M = int(time_steps)
    h = T / M
    beta = perpetual_put_boundary_beta(K, r, sigma)
    gamma = r - 0.5 * sigma * sigma
    B = np.empty(M + 1, dtype=np.float64)
    B[0] = K
    dy = K / int(cfg.peskir_terminal_z_nodes)
    y_mid = (np.arange(int(cfg.peskir_terminal_z_nodes), dtype=np.float64) + 0.5) * dy

    for i in range(1, M + 1):
        tau = i * h
        if i > 1:
            j = np.arange(1, i, dtype=np.int64)
            u = j.astype(np.float64) * h
            future_B = B[i - j]
            disc = np.exp(-r * u)
            sqrt_u = np.sqrt(u)
        else:
            u = future_B = disc = sqrt_u = np.empty(0, dtype=np.float64)

        def residual(x: float) -> float:
            terminal = discounted_terminal_put_riemann(x, tau, K, r, sigma, y_mid, dy)
            probability_sum = 0.5
            if i > 1:
                arg = (np.log(future_B / x) - gamma * u) / (sigma * sqrt_u)
                probability_sum += float(np.sum(disc * ndtr(arg), dtype=np.float64))
            return float(K - x - terminal - r * K * h * probability_sum)

        lo = beta * (1.0 + 1.0e-12)
        hi = K * (1.0 - 1.0e-12)
        if residual(lo) * residual(hi) > 0.0:
            raise RuntimeError(f"Could not bracket Peskir boundary root at index {i}.")
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
    return times, boundary, beta, h


def peskir_value_from_boundary(
    cfg: ExperimentConfig, times: np.ndarray, boundary: np.ndarray
) -> float:
    S0, K, r, T, sigma = cfg.S0, cfg.K, cfg.r, cfg.T, cfg.sigma
    M = len(times) - 1
    h = T / M
    gamma = r - 0.5 * sigma * sigma
    dy = K / int(cfg.peskir_terminal_z_nodes)
    y_mid = (np.arange(int(cfg.peskir_terminal_z_nodes), dtype=np.float64) + 0.5) * dy
    terminal = discounted_terminal_put_riemann(S0, T, K, r, sigma, y_mid, dy)
    if S0 < boundary[0] - 1e-12:
        p0 = 1.0
    elif abs(S0 - boundary[0]) <= 1e-12:
        p0 = 0.5
    else:
        p0 = 0.0
    if M > 1:
        u = np.arange(1, M, dtype=np.float64) * h
        arg = (np.log(boundary[1:M] / S0) - gamma * u) / (sigma * np.sqrt(u))
        psum = p0 + float(np.sum(np.exp(-r * u) * ndtr(arg), dtype=np.float64))
    else:
        psum = p0
    return float(terminal + r * K * h * psum)


def load_or_compute_peskir_reference(
    cfg: ExperimentConfig, root: Path, force: bool = False
) -> PeskirBoundaryResult:
    folder = root / "peskir_d1_reference"
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"boundary_M{cfg.peskir_time_steps}.csv"
    json_path = folder / f"reference_M{cfg.peskir_time_steps}.json"
    if csv_path.exists() and json_path.exists() and not force:
        frame = pd.read_csv(csv_path)
        meta = json_load(json_path, {})
        return PeskirBoundaryResult(
            times=frame["time"].to_numpy(np.float64),
            boundary=frame["boundary"].to_numpy(np.float64),
            beta=float(meta["beta"]),
            h=float(meta["h"]),
            value_at_S0=float(meta["value_at_S0"]),
            terminal_integral_max_abs_error=float(meta["terminal_integral_max_abs_error"]),
            coarse_fine_value_difference=meta.get("coarse_fine_value_difference"),
            computation_seconds=float(meta.get("computation_seconds", 0.0)),
        )

    start = time.perf_counter()
    times, boundary, beta, h = solve_peskir_boundary_once(cfg, cfg.peskir_time_steps)
    value = peskir_value_from_boundary(cfg, times, boundary)
    coarse_diff: Optional[float] = None
    if cfg.run_peskir_coarse_check:
        ct, cb, _, _ = solve_peskir_boundary_once(cfg, cfg.peskir_coarse_time_steps)
        coarse_diff = float(value - peskir_value_from_boundary(cfg, ct, cb))
    dy = cfg.K / int(cfg.peskir_terminal_z_nodes)
    y_mid = (np.arange(int(cfg.peskir_terminal_z_nodes), dtype=np.float64) + 0.5) * dy
    riemann = discounted_terminal_put_riemann(
        cfg.S0, cfg.T, cfg.K, cfg.r, cfg.sigma, y_mid, dy
    )
    terminal_error = abs(
        riemann - european_put_closed_form(cfg.S0, cfg.K, cfg.r, cfg.sigma, cfg.T)
    )
    elapsed = time.perf_counter() - start
    pd.DataFrame({"time": times, "boundary": boundary}).to_csv(csv_path, index=False)
    json_dump(
        json_path,
        {
            "value_at_S0": value,
            "S0": cfg.S0,
            "K": cfg.K,
            "r": cfg.r,
            "sigma": cfg.sigma,
            "T": cfg.T,
            "beta": beta,
            "h": h,
            "terminal_integral_max_abs_error": terminal_error,
            "coarse_fine_value_difference": coarse_diff,
            "computation_seconds": elapsed,
            "scope": "continuous-time one-dimensional American put reference",
        },
    )
    print(f"[Peskir] d=1 continuous reference={value:.10f}")
    return PeskirBoundaryResult(
        times, boundary, beta, h, value, terminal_error, coarse_diff, elapsed
    )

# Networks and datewise Becker training

class StoppingNetwork(tf.keras.Model):
    def __init__(self, width: int, bn_momentum: float, name: str):
        super().__init__(name=name)
        self.dense1 = tf.keras.layers.Dense(
            width,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer="zeros",
            name="dense1",
        )
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=bn_momentum, name="bn1")
        self.dense2 = tf.keras.layers.Dense(
            width,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer="zeros",
            name="dense2",
        )
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=bn_momentum, name="bn2")
        self.out = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
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
        return self.out(x)


def normalized_features(states: tf.Tensor, rewards: tf.Tensor, K: float) -> tf.Tensor:
    return tf.concat(
        [states / tf.constant(K, tf.float32), rewards[..., None] / tf.constant(K, tf.float32)],
        axis=-1,
    )


def basket_reward(states: tf.Tensor, discount: tf.Tensor, K: float) -> tf.Tensor:
    mean_state = tf.reduce_mean(states, axis=-1)
    return discount * tf.nn.relu(tf.constant(K, tf.float32) - mean_state)


def make_models_and_optimizers(
    cfg: ExperimentConfig, dimension: int
) -> Tuple[List[Optional[StoppingNetwork]], List[Optional[tf.keras.optimizers.Optimizer]]]:
    models: List[Optional[StoppingNetwork]] = [None] * (cfg.N + 1)
    optimizers: List[Optional[tf.keras.optimizers.Optimizer]] = [None] * (cfg.N + 1)
    dummy = tf.constant(
        [[1.0] * dimension + [max(cfg.K - cfg.S0, 0.0) / cfg.K]],
        dtype=tf.float32,
    )
    for n in range(1, cfg.N):
        model = StoppingNetwork(
            cfg.hidden_width(dimension), cfg.batch_norm_momentum, f"stopping_net_{n:02d}"
        )
        _ = model(dummy, training=False)
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        try:
            opt.build(model.trainable_variables)
        except (AttributeError, TypeError):
            opt.apply_gradients(
                zip([tf.zeros_like(v) for v in model.trainable_variables], model.trainable_variables)
            )
        models[n] = model
        optimizers[n] = opt
    return models, optimizers


def make_checkpoint(
    cfg: ExperimentConfig,
    folder: Path,
    models: Sequence[Optional[StoppingNetwork]],
    optimizers: Sequence[Optional[tf.keras.optimizers.Optimizer]],
    rng: tf.random.Generator,
) -> Tuple[tf.train.CheckpointManager, tf.Variable, tf.Variable, tf.Variable]:
    current_date = tf.Variable(cfg.N - 1, dtype=tf.int64, trainable=False, name="current_date")
    date_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="date_step")
    total_updates = tf.Variable(0, dtype=tf.int64, trainable=False, name="total_updates")
    objects: Dict[str, Any] = {
        "current_date": current_date,
        "date_step": date_step,
        "total_updates": total_updates,
        "rng": rng,
    }
    for n in range(1, cfg.N):
        objects[f"net_{n:02d}"] = models[n]
        objects[f"opt_{n:02d}"] = optimizers[n]
    checkpoint = tf.train.Checkpoint(**objects)
    manager = tf.train.CheckpointManager(checkpoint, str(folder / "checkpoints"), max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(
            f"[Resume] {manager.latest_checkpoint}; date={int(current_date.numpy())}, "
            f"step={int(date_step.numpy())}, total={int(total_updates.numpy())}"
        )
    return manager, current_date, date_step, total_updates


def make_training_step(
    cfg: ExperimentConfig,
    dimension: int,
    date_n: int,
    models: Sequence[Optional[StoppingNetwork]],
    optimizer: tf.keras.optimizers.Optimizer,
    rng: tf.random.Generator,
) -> Callable[[], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    N = cfg.N
    batch = cfg.training_batch_size
    dt = np.float32(cfg.T / N)
    mu = tf.constant((cfg.r - cfg.q - 0.5 * cfg.sigma**2) * dt, tf.float32)
    vol = tf.constant(cfg.sigma * math.sqrt(float(dt)), tf.float32)
    discounts = tf.constant(np.exp(-cfg.r * np.linspace(0.0, cfg.T, N + 1)).astype(np.float32))
    s0 = tf.constant(cfg.S0, tf.float32)
    current = models[date_n]
    assert current is not None

    @tf.function(reduce_retracing=True)
    def train_step() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        eps = rng.normal((batch, N, dimension), dtype=tf.float32)
        log_inc = mu + vol * eps
        log_paths = tf.math.log(s0) + tf.cumsum(log_inc, axis=1)
        start = tf.fill((batch, 1, dimension), s0)
        paths = tf.concat([start, tf.exp(log_paths)], axis=1)

        future_reward = basket_reward(paths[:, N, :], discounts[N], cfg.K)
        for m in range(N - 1, date_n, -1):
            model = models[m]
            assert model is not None
            immediate = basket_reward(paths[:, m, :], discounts[m], cfg.K)
            p = tf.reshape(model(normalized_features(paths[:, m, :], immediate, cfg.K), training=False), (-1,))
            future_reward = tf.where(p >= 0.5, immediate, future_reward)
        future_reward = tf.stop_gradient(future_reward)

        immediate_n = basket_reward(paths[:, date_n, :], discounts[date_n], cfg.K)
        features_n = normalized_features(paths[:, date_n, :], immediate_n, cfg.K)
        with tf.GradientTape() as tape:
            soft = tf.reshape(current(features_n, training=True), (-1,))
            value = immediate_n * soft + future_reward * (1.0 - soft)
            objective = tf.reduce_mean(value)
            loss = -objective
        grads = tape.gradient(loss, current.trainable_variables)
        pairs = [(g, v) for g, v in zip(grads, current.trainable_variables) if g is not None]
        if not pairs:
            raise RuntimeError(f"No gradients at date {date_n}.")
        optimizer.apply_gradients(pairs)
        p_hard = tf.reshape(current(features_n, training=False), (-1,))
        hard_value = tf.where(p_hard >= 0.5, immediate_n, future_reward)
        return objective, tf.reduce_mean(hard_value), tf.reduce_mean(soft)

    return train_step


def train_or_resume(
    cfg: ExperimentConfig,
    dimension: int,
    folder: Path,
    models: Sequence[Optional[StoppingNetwork]],
    optimizers: Sequence[Optional[tf.keras.optimizers.Optimizer]],
    rng: tf.random.Generator,
    manager: tf.train.CheckpointManager,
    current_date: tf.Variable,
    date_step: tf.Variable,
    total_updates: tf.Variable,
    timing_path: Path,
) -> None:
    steps_per_date = cfg.training_steps(dimension)
    if int(current_date.numpy()) <= 0:
        print("[Training] already complete")
        return
    log_path = folder / "training_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["date", "step", "total_update", "soft_objective", "hard_value", "mean_soft_stop", "timestamp"]
            )
    summary_path = folder / "datewise_training_summary.csv"
    if not summary_path.exists():
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["date", "time", "steps_completed", "total_updates", "timestamp"])

    for m in range(int(current_date.numpy()) + 1, cfg.N):
        if models[m] is not None:
            models[m].trainable = False

    while int(current_date.numpy()) >= 1:
        n = int(current_date.numpy())
        model = models[n]
        opt = optimizers[n]
        assert model is not None and opt is not None
        model.trainable = True
        for m in range(n + 1, cfg.N):
            if models[m] is not None:
                models[m].trainable = False
        start_step = int(date_step.numpy())
        print(f"[Training d={dimension}] date {n:02d}/{cfg.N-1:02d}, steps {start_step+1}...{steps_per_date}")
        step_fn = make_training_step(cfg, dimension, n, models, opt, rng)
        segment = time.perf_counter()
        for step in range(start_step, steps_per_date):
            objective, hard_value, mean_stop = step_fn()
            date_step.assign(step + 1)
            total_updates.assign_add(1)
            should_log = (step + 1) % cfg.training_log_every == 0 or step == start_step or step + 1 == steps_per_date
            should_ckpt = (step + 1) % cfg.checkpoint_every == 0 or step + 1 == steps_per_date
            if should_log:
                now = time.perf_counter()
                update_timing(timing_path, "training_seconds", now - segment)
                segment = now
                with log_path.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [n, step + 1, int(total_updates.numpy()), float(objective.numpy()), float(hard_value.numpy()), float(mean_stop.numpy()), time.strftime("%Y-%m-%d %H:%M:%S")]
                    )
                print(
                    f"  date {n:02d}, step {step+1:4d}/{steps_per_date}: "
                    f"objective={float(objective.numpy()):.6f}, hard={float(hard_value.numpy()):.6f}, stop={float(mean_stop.numpy()):.4f}"
                )
            if should_ckpt:
                manager.save(checkpoint_number=int(total_updates.numpy()))
        now = time.perf_counter()
        if now > segment:
            update_timing(timing_path, "training_seconds", now - segment)
        model.trainable = False
        with summary_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([n, n * cfg.T / cfg.N, steps_per_date, int(total_updates.numpy()), time.strftime("%Y-%m-%d %H:%M:%S")])
        current_date.assign(n - 1)
        date_step.assign(0)
        manager.save(checkpoint_number=int(total_updates.numpy()))
        del step_fn
        gc.collect()
    print("[Training] complete")


# Common-random-number evaluation and lower bound

def _paths_from_asset_seeds(
    cfg: ExperimentConfig, dimension: int, seeds: tf.Tensor, n_paths: tf.Tensor
) -> tf.Tensor:
    n = tf.cast(n_paths, tf.int32)
    normals = []
    for i in range(dimension):
        normals.append(tf.random.stateless_normal(tf.stack([n, cfg.N]), seed=seeds[i], dtype=tf.float32))
    eps = tf.stack(normals, axis=-1)  # (paths,N,d)
    dt = np.float32(cfg.T / cfg.N)
    mu = tf.constant((cfg.r - cfg.q - 0.5 * cfg.sigma**2) * dt, tf.float32)
    vol = tf.constant(cfg.sigma * math.sqrt(float(dt)), tf.float32)
    log_paths = tf.math.log(tf.constant(cfg.S0, tf.float32)) + tf.cumsum(mu + vol * eps, axis=1)
    start = tf.fill(tf.stack([n, 1, dimension]), tf.constant(cfg.S0, tf.float32))
    return tf.concat([start, tf.exp(log_paths)], axis=1)


def make_policy_payoff_function(
    cfg: ExperimentConfig,
    dimension: int,
    models: Sequence[Optional[StoppingNetwork]],
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    discounts = tf.constant(np.exp(-cfg.r * np.linspace(0.0, cfg.T, cfg.N + 1)).astype(np.float32))

    @tf.function(reduce_retracing=True)
    def simulate_and_evaluate(seeds: tf.Tensor, n_paths: tf.Tensor) -> tf.Tensor:
        paths = _paths_from_asset_seeds(cfg, dimension, seeds, n_paths)
        n = tf.cast(n_paths, tf.int32)
        alive = tf.ones(tf.stack([n]), tf.bool)
        payoff = tf.zeros(tf.stack([n]), tf.float32)
        for date in range(1, cfg.N):
            model = models[date]
            assert model is not None
            reward = basket_reward(paths[:, date, :], discounts[date], cfg.K)
            p = tf.reshape(model(normalized_features(paths[:, date, :], reward, cfg.K), training=False), (-1,))
            stop = tf.logical_and(alive, p >= 0.5)
            payoff = tf.where(stop, reward, payoff)
            alive = tf.logical_and(alive, tf.logical_not(stop))
        maturity = basket_reward(paths[:, cfg.N, :], discounts[cfg.N], cfg.K)
        return tf.where(alive, maturity, payoff)

    return simulate_and_evaluate


def determine_f0(
    cfg: ExperimentConfig,
    dimension: int,
    policy_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    folder: Path,
) -> Tuple[int, float, float]:
    path = folder / "f0_decision.json"
    if path.exists():
        x = json_load(path, {})
        return int(x["f0"]), float(x["immediate_payoff"]), float(x["estimated_continuation_value"])
    seeds = tf.constant(evaluation_seed_matrix(cfg, 31, 0, dimension), tf.int32)
    values = policy_fn(seeds, tf.constant(cfg.f0_pilot_paths, tf.int32))
    continuation = float(tf.reduce_mean(values).numpy())
    immediate = max(cfg.K - cfg.S0, 0.0)
    f0 = int(immediate >= continuation)
    json_dump(
        path,
        {
            "f0": f0,
            "pilot_paths": cfg.f0_pilot_paths,
            "immediate_payoff": immediate,
            "estimated_continuation_value": continuation,
            "evaluation_seed": cfg.evaluation_seed,
        },
    )
    return f0, immediate, continuation


def open_or_create_memmap(
    path: Path, shape: Tuple[int, ...], dtype: np.dtype, fill_value: Any
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        arr = np.lib.format.open_memmap(path, mode="r+")
        if tuple(arr.shape) != shape or arr.dtype != np.dtype(dtype):
            raise RuntimeError(f"Incompatible memmap {path}")
        return arr
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    arr[...] = fill_value
    arr.flush()
    return arr


def compute_lower_bound(
    cfg: ExperimentConfig,
    dimension: int,
    f0: int,
    immediate: float,
    policy_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    folder: Path,
    timing_path: Path,
) -> Tuple[float, float, float, int]:
    path = folder / "lower_bound_payoffs.npy"
    payoffs = open_or_create_memmap(path, (cfg.K_L,), np.float32, np.nan)
    chunk = resolve_lower_chunk_size(cfg, dimension)
    if f0 == 1:
        payoffs[:] = np.float32(immediate)
        payoffs.flush()
    else:
        n_chunks = cfg.K_L // chunk
        print(f"[Lower d={dimension}] K_L={cfg.K_L:,}, chunk={chunk:,}, chunks={n_chunks}")
        for block in range(n_chunks):
            start, end = block * chunk, (block + 1) * chunk
            if np.all(np.isfinite(payoffs[start:end])):
                continue
            tick = time.perf_counter()
            seeds = tf.constant(evaluation_seed_matrix(cfg, 41, block, dimension), tf.int32)
            values = policy_fn(seeds, tf.constant(chunk, tf.int32)).numpy()
            payoffs[start:end] = values.astype(np.float32, copy=False)
            payoffs.flush()
            update_timing(timing_path, "lower_seconds", time.perf_counter() - tick)
            print(f"  lower block {block+1}/{n_chunks}")
    if not np.all(np.isfinite(payoffs)):
        raise RuntimeError("Lower payoff file is incomplete.")
    x = np.asarray(payoffs, np.float64)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if cfg.K_L > 1 else 0.0
    return mean, sd, sd / math.sqrt(cfg.K_L), chunk


# Becker frozen-future nested dual upper bound

def make_outer_path_function(
    cfg: ExperimentConfig, dimension: int, outer_batch: int
) -> Callable[[tf.Tensor], tf.Tensor]:
    @tf.function(reduce_retracing=True)
    def simulate(seeds: tf.Tensor) -> tf.Tensor:
        return _paths_from_asset_seeds(cfg, dimension, seeds, tf.constant(outer_batch, tf.int32))
    return simulate


def ensure_outer_paths(
    cfg: ExperimentConfig,
    dimension: int,
    outer_batch: int,
    outer_fn: Callable[[tf.Tensor], tf.Tensor],
    folder: Path,
) -> np.memmap:
    path = folder / "upper_outer_paths.npy"
    outer = open_or_create_memmap(path, (cfg.K_U, cfg.N + 1, dimension), np.float32, np.nan)
    n_batches = cfg.K_U // outer_batch
    for batch_id in range(n_batches):
        s, e = batch_id * outer_batch, (batch_id + 1) * outer_batch
        if np.all(np.isfinite(outer[s:e])):
            continue
        seeds = tf.constant(evaluation_seed_matrix(cfg, 51, batch_id, dimension), tf.int32)
        outer[s:e] = outer_fn(seeds).numpy().astype(np.float32, copy=False)
        outer.flush()
    return outer


def make_continuation_sum_function(
    cfg: ExperimentConfig,
    dimension: int,
    models: Sequence[Optional[StoppingNetwork]],
    start_n: int,
    outer_batch: int,
    inner_chunk: int,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    discounts = tf.constant(np.exp(-cfg.r * np.linspace(0.0, cfg.T, cfg.N + 1)).astype(np.float32))
    dt = np.float32(cfg.T / cfg.N)
    mu = tf.constant((cfg.r - cfg.q - 0.5 * cfg.sigma**2) * dt, tf.float32)
    vol = tf.constant(cfg.sigma * math.sqrt(float(dt)), tf.float32)

    @tf.function(
        input_signature=[
            tf.TensorSpec((outer_batch, dimension), tf.float32),
            tf.TensorSpec((outer_batch, inner_chunk, cfg.N, dimension), tf.float32),
        ],
        reduce_retracing=True,
    )
    def continuation_sum(z_n: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
        s = tf.broadcast_to(z_n[:, None, :], (outer_batch, inner_chunk, dimension))
        alive = tf.ones((outer_batch, inner_chunk), tf.bool)
        payoff = tf.zeros((outer_batch, inner_chunk), tf.float32)
        for m in range(start_n + 1, cfg.N + 1):
            s = s * tf.exp(mu + vol * eps[:, :, m - 1, :])
            reward = basket_reward(s, discounts[m], cfg.K)
            if m < cfg.N:
                model = models[m]
                assert model is not None
                flat_s = tf.reshape(s, (-1, dimension))
                flat_r = tf.reshape(reward, (-1,))
                p = tf.reshape(model(normalized_features(flat_s, flat_r, cfg.K), training=False), (outer_batch, inner_chunk))
                stop = tf.logical_and(alive, p >= 0.5)
            else:
                stop = alive
            payoff = tf.where(stop, reward, payoff)
            alive = tf.logical_and(alive, tf.logical_not(stop))
        return tf.reduce_sum(payoff, axis=1)

    return continuation_sum


def _nested_eps(
    cfg: ExperimentConfig,
    dimension: int,
    outer_batch: int,
    inner_chunk: int,
    block_index: int,
) -> tf.Tensor:
    arrays = []
    for asset in range(dimension):
        seed = tf.constant(seed_pair(cfg.evaluation_seed, 61, block_index, asset), tf.int32)
        arrays.append(
            tf.random.stateless_normal((outer_batch, inner_chunk, cfg.N), seed=seed, dtype=tf.float32)
        )
    return tf.stack(arrays, axis=-1)


def _save_partial_upper_state(
    path: Path,
    date: int,
    batch: int,
    next_inner: int,
    payoff_sum: np.ndarray,
) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        date=np.asarray(date, np.int32),
        batch=np.asarray(batch, np.int32),
        next_inner=np.asarray(next_inner, np.int32),
        payoff_sum=np.asarray(payoff_sum, np.float64),
    )
    os.replace(tmp, path)


def _load_partial_upper_state(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with np.load(path) as data:
        return {
            "date": int(data["date"]),
            "batch": int(data["batch"]),
            "next_inner": int(data["next_inner"]),
            "payoff_sum": np.asarray(data["payoff_sum"], np.float64),
        }


def compute_continuation_values(
    cfg: ExperimentConfig,
    dimension: int,
    models: Sequence[Optional[StoppingNetwork]],
    outer: np.memmap,
    folder: Path,
    timing_path: Path,
    outer_batch: int,
    inner_chunk: int,
) -> np.memmap:
    path = folder / "upper_continuation_values.npy"
    cont = open_or_create_memmap(path, (cfg.K_U, cfg.N), np.float32, np.nan)
    partial_path = folder / "upper_active_inner_chunk_state.npz"
    n_batches = cfg.K_U // outer_batch
    n_inner = cfg.J // inner_chunk

    for date in range(cfg.N):
        if np.all(np.isfinite(cont[:, date])):
            continue
        print(f"[Upper:C d={dimension}] date {date+1}/{cfg.N}, inner chunks={n_inner}")
        fn = make_continuation_sum_function(cfg, dimension, models, date, outer_batch, inner_chunk)
        for batch_id in range(n_batches):
            s, e = batch_id * outer_batch, (batch_id + 1) * outer_batch
            if np.all(np.isfinite(cont[s:e, date])):
                continue
            state = _load_partial_upper_state(partial_path)
            if state and state["date"] == date and state["batch"] == batch_id:
                next_inner = state["next_inner"]
                payoff_sum = state["payoff_sum"]
                print(f"  [Resume inner] date={date}, batch={batch_id}, next={next_inner}")
            else:
                if state is not None:
                    partial_path.unlink(missing_ok=True)
                next_inner = 0
                payoff_sum = np.zeros(outer_batch, np.float64)
            z_n = tf.convert_to_tensor(np.asarray(outer[s:e, date, :], np.float32), tf.float32)
            for inner_id in range(next_inner, n_inner):
                tick = time.perf_counter()
                block_index = batch_id * n_inner + inner_id
                eps = _nested_eps(cfg, dimension, outer_batch, inner_chunk, block_index)
                payoff_sum += fn(z_n, eps).numpy().astype(np.float64, copy=False)
                _save_partial_upper_state(partial_path, date, batch_id, inner_id + 1, payoff_sum)
                update_timing(timing_path, "upper_continuation_seconds", time.perf_counter() - tick)
                del eps
            cont[s:e, date] = (payoff_sum / float(cfg.J)).astype(np.float32)
            cont.flush()
            partial_path.unlink(missing_ok=True)
            if batch_id == 0 or (batch_id + 1) % max(1, n_batches // 8) == 0 or batch_id + 1 == n_batches:
                print(f"  continuation batch {batch_id+1}/{n_batches}")
        del fn
        gc.collect()
    if not np.all(np.isfinite(cont)):
        raise RuntimeError("Continuation values are incomplete.")
    return cont


def evaluate_outer_policy(
    cfg: ExperimentConfig,
    dimension: int,
    outer_tf: tf.Tensor,
    models: Sequence[Optional[StoppingNetwork]],
) -> np.ndarray:
    B = int(outer_tf.shape[0])
    decisions = np.zeros((B, cfg.N + 1), np.float64)
    discounts = np.exp(-cfg.r * np.linspace(0.0, cfg.T, cfg.N + 1))
    for n in range(1, cfg.N):
        model = models[n]
        assert model is not None
        states = outer_tf[:, n, :]
        reward = basket_reward(states, tf.constant(discounts[n], tf.float32), cfg.K)
        p = tf.reshape(model(normalized_features(states, reward, cfg.K), training=False), (-1,)).numpy()
        decisions[:, n] = (p >= 0.5).astype(np.float64)
    decisions[:, cfg.N] = 1.0
    return decisions


def compute_dual_payoffs(
    cfg: ExperimentConfig,
    dimension: int,
    models: Sequence[Optional[StoppingNetwork]],
    outer: np.memmap,
    continuation: np.memmap,
    folder: Path,
    timing_path: Path,
    outer_batch: int,
) -> np.memmap:
    path = folder / "upper_bound_dual_payoffs.npy"
    dual = open_or_create_memmap(path, (cfg.K_U,), np.float64, np.nan)
    n_batches = cfg.K_U // outer_batch
    discounts = np.exp(-cfg.r * np.linspace(0.0, cfg.T, cfg.N + 1))
    for batch_id in range(n_batches):
        s, e = batch_id * outer_batch, (batch_id + 1) * outer_batch
        if np.all(np.isfinite(dual[s:e])):
            continue
        tick = time.perf_counter()
        outer_np = np.asarray(outer[s:e], np.float64)
        outer_tf = tf.convert_to_tensor(np.asarray(outer[s:e], np.float32), tf.float32)
        C = np.asarray(continuation[s:e], np.float64)
        basket_mean = np.mean(outer_np, axis=-1)
        g = discounts[None, :] * np.maximum(cfg.K - basket_mean, 0.0)
        f = evaluate_outer_policy(cfg, dimension, outer_tf, models)
        delta = np.empty((outer_batch, cfg.N), np.float64)
        for n in range(1, cfg.N):
            delta[:, n - 1] = f[:, n] * g[:, n] + (1.0 - f[:, n]) * C[:, n] - C[:, n - 1]
        delta[:, cfg.N - 1] = g[:, cfg.N] - C[:, cfg.N - 1]
        M = np.zeros((outer_batch, cfg.N + 1), np.float64)
        M[:, 1:] = np.cumsum(delta, axis=1)
        dual[s:e] = np.max(g - M, axis=1)
        dual.flush()
        update_timing(timing_path, "upper_dual_seconds", time.perf_counter() - tick)
        print(f"  upper dual batch {batch_id+1}/{n_batches}")
        del outer_np, outer_tf, C, basket_mean, g, f, delta, M
        gc.collect()
    if not np.all(np.isfinite(dual)):
        raise RuntimeError("Dual payoff file is incomplete.")
    return dual


def compute_upper_bound(
    cfg: ExperimentConfig,
    dimension: int,
    models: Sequence[Optional[StoppingNetwork]],
    folder: Path,
    timing_path: Path,
) -> Tuple[float, float, float, int, int, Dict[str, Any]]:
    outer_batch, inner_chunk, workspace = resolve_upper_workspace(cfg, dimension)
    metadata = {
        "implementation": UPPER_VERSION,
        "K_U": cfg.K_U,
        "J": cfg.J,
        "N": cfg.N,
        "dimension": dimension,
        "evaluation_seed": cfg.evaluation_seed,
        "common_random_numbers": True,
        **workspace,
    }
    metadata_path = folder / "upper_workspace_metadata.json"
    if metadata_path.exists() and json_load(metadata_path, {}) != metadata:
        raise RuntimeError("Saved upper workspace metadata differs from current settings.")
    json_dump(metadata_path, metadata)
    print(
        f"[Upper d={dimension}] K_U={cfg.K_U:,}, J={cfg.J:,}, "
        f"outer batch={outer_batch}, inner chunk={inner_chunk}"
    )
    outer_fn = make_outer_path_function(cfg, dimension, outer_batch)
    outer = ensure_outer_paths(cfg, dimension, outer_batch, outer_fn, folder)
    del outer_fn
    start = time.perf_counter()
    continuation = compute_continuation_values(
        cfg, dimension, models, outer, folder, timing_path, outer_batch, inner_chunk
    )
    dual = compute_dual_payoffs(
        cfg, dimension, models, outer, continuation, folder, timing_path, outer_batch
    )
    update_timing(timing_path, "upper_seconds", time.perf_counter() - start)
    x = np.asarray(dual, np.float64)
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if cfg.K_U > 1 else 0.0
    return mean, sd, sd / math.sqrt(cfg.K_U), outer_batch, inner_chunk, workspace


# Orchestration

def run_case(
    cfg: ExperimentConfig,
    dimension: int,
    training_seed: int,
    peskir_reference: Optional[float],
    root: Path,
    force_recompute: bool = False,
    skip_upper: bool = False,
) -> Dict[str, Any]:
    folder = case_dir(root, dimension, training_seed)
    if force_recompute and folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)
    signature = config_signature(cfg, dimension, training_seed)
    check_or_write_signature(folder, signature)
    result_path = folder / "case_result.json"
    if result_path.exists() and not force_recompute:
        result = json_load(result_path, {})
        if result.get("status") == "complete":
            print(f"[Skip complete] d={dimension}, seed={training_seed}")
            return result

    timing_path = folder / "timing_state.json"
    print("\n" + "=" * 100)
    print(f"d={dimension}, training seed={training_seed}")
    print("=" * 100)
    tf.keras.backend.clear_session()
    gc.collect()
    set_global_seeds(training_seed)

    models, optimizers = make_models_and_optimizers(cfg, dimension)
    rng = tf.random.Generator.from_seed(training_seed)
    manager, current_date, date_step, total_updates = make_checkpoint(
        cfg, folder, models, optimizers, rng
    )
    train_or_resume(
        cfg,
        dimension,
        folder,
        models,
        optimizers,
        rng,
        manager,
        current_date,
        date_step,
        total_updates,
        timing_path,
    )
    policy_fn = make_policy_payoff_function(cfg, dimension, models)
    f0, immediate, f0_cont = determine_f0(cfg, dimension, policy_fn, folder)
    lower, lower_sd, lower_se, lower_chunk = compute_lower_bound(
        cfg, dimension, f0, immediate, policy_fn, folder, timing_path
    )

    partial: Dict[str, Any] = {
        "algorithm_version": ALGORITHM_VERSION,
        "dimension": dimension,
        "training_seed": training_seed,
        "evaluation_seed": cfg.evaluation_seed,
        "S0_each_asset": cfg.S0,
        "sigma_each_asset": cfg.sigma,
        "rho_off_diagonal": cfg.rho,
        "K": cfg.K,
        "r": cfg.r,
        "q": cfg.q,
        "T": cfg.T,
        "N": cfg.N,
        "number_of_grid_points": cfg.N + 1,
        "hidden_width": cfg.hidden_width(dimension),
        "training_steps_per_date": cfg.training_steps(dimension),
        "training_batch_size": cfg.training_batch_size,
        "K_L": cfg.K_L,
        "K_U": cfg.K_U,
        "J": cfg.J,
        "f0": f0,
        "f0_continuation_estimate": f0_cont,
        "lower_bound": lower,
        "lower_sd": lower_sd,
        "lower_se": lower_se,
        "lower_chunk_size": lower_chunk,
        "status": "partial_lower_complete",
    }
    json_dump(folder / "case_result_partial.json", partial)
    if skip_upper:
        return partial

    upper, upper_sd, upper_se, outer_batch, inner_chunk, workspace = compute_upper_bound(
        cfg, dimension, models, folder, timing_path
    )
    z = confidence_z(cfg)
    point = 0.5 * (lower + upper)
    gap = upper - lower
    ci_low = lower - z * lower_se
    ci_high = upper + z * upper_se
    timing = json_load(timing_path, {})
    total_seconds = float(
        timing.get("training_seconds", 0.0)
        + timing.get("lower_seconds", 0.0)
        + timing.get("upper_seconds", 0.0)
    )
    reference = float(peskir_reference) if dimension == 1 and peskir_reference is not None else None
    error = point - reference if reference is not None else None
    result = {
        **partial,
        "upper_bound": upper,
        "upper_sd": upper_sd,
        "upper_se": upper_se,
        "point_estimate": point,
        "duality_gap": gap,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": cfg.confidence_level,
        "upper_outer_batch_size": outer_batch,
        "upper_inner_chunk_size": inner_chunk,
        "upper_workspace": workspace,
        "training_seconds": float(timing.get("training_seconds", 0.0)),
        "lower_seconds": float(timing.get("lower_seconds", 0.0)),
        "upper_seconds": float(timing.get("upper_seconds", 0.0)),
        "total_seconds": total_seconds,
        "peskir_shiryaev_reference_d1": reference,
        "error": error,
        "absolute_error": abs(error) if error is not None else None,
        "status": "complete",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_dump(result_path, result)
    print(
        f"[Result d={dimension}, seed={training_seed}] L={lower:.8f}, U={upper:.8f}, "
        f"point={point:.8f}, gap={gap:.8f}, CI=[{ci_low:.8f},{ci_high:.8f}]"
    )
    del policy_fn, models, optimizers
    tf.keras.backend.clear_session()
    gc.collect()
    return result
