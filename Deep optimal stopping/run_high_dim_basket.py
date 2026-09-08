from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from becker_basket_engine import (
    DIMENSIONS,
    TRAINING_SEEDS,
    ExperimentConfig,
    configure_tensorflow,
    load_or_compute_peskir_reference,
    quick_config,
    run_case,
)
from summarise_results import summarise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=list(DIMENSIONS),
        help="Dimensions to run. Default: 1 2 3 5 10 20 30 50.",
    )
    parser.add_argument(
        "--seed-values",
        nargs="+",
        type=int,
        default=list(TRAINING_SEEDS),
        help="Training seeds. Default: 202601 202602 202603.",
    )
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--force-peskir", action="store_true")
    parser.add_argument("--skip-upper", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution. Intended only for a tiny quick test.",
    )
    return parser.parse_args()


def validate_selection(values: List[int], allowed: tuple[int, ...], label: str) -> None:
    bad = [x for x in values if x not in allowed]
    if bad:
        raise ValueError(f"Unsupported {label}: {bad}; allowed={list(allowed)}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {label} values are not allowed: {values}")


def main() -> None:
    args = parse_args()
    validate_selection(args.dimensions, DIMENSIONS, "dimensions")
    validate_selection(args.seed_values, TRAINING_SEEDS, "training seeds")

    cfg = ExperimentConfig()
    if args.output_dir:
        cfg = ExperimentConfig(output_dir=args.output_dir)
    if args.quick_test:
        cfg = quick_config(cfg)
        if args.dimensions == list(DIMENSIONS):
            args.dimensions = [1, 2]
        if args.seed_values == list(TRAINING_SEEDS):
            args.seed_values = [TRAINING_SEEDS[0]]
    if args.allow_cpu:
        from dataclasses import replace

        cfg = replace(cfg, require_gpu=False)

    configure_tensorflow(cfg.require_gpu)
    root = Path(cfg.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "master_experiment_configuration.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dimensions": args.dimensions,
                "training_seeds": args.seed_values,
                "execution_order": "seed-major, then dimension",
                "configuration": cfg.__dict__,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    peskir = load_or_compute_peskir_reference(cfg, root, force=args.force_peskir)

    for training_seed in args.seed_values:
        for dimension in args.dimensions:
            run_case(
                cfg,
                dimension,
                training_seed,
                peskir.value_at_S0,
                root,
                force_recompute=args.force_recompute,
                skip_upper=args.skip_upper,
            )
            summarise(root)

    summarise(root)
    print("\nAll selected experiments finished.")
    print(f"Outputs: {root}")


if __name__ == "__main__":
    main()
