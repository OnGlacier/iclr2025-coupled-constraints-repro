"""M2: compare the independent Main implementation with the official one.

For each model we run ``methods.Main`` and ``repro.m2_main_complexity.main_impl.my_main``
from the same zero initial point and compare the full error trajectories and
the per-iteration resource counters.  The expected result is agreement at
machine precision (relative error around 1e-12 or smaller).

Run from the repository root::

    python repro/m2_main_complexity/compare_with_official.py
    python repro/m2_main_complexity/compare_with_official.py --steps 50
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods import Main  # noqa: E402
from models import ConsensusModel, ExampleModel, Model, VFL  # noqa: E402

from repro.common import ensure_output_dirs, save_json, seed_everything  # noqa: E402
from repro.m2_main_complexity.main_impl import my_main  # noqa: E402


def make_model(name: str) -> Model:
    if name == "example_er":
        return ExampleModel(20, 10, 3, "erdos-renyi", edge_prob=5 / 20, gossip=False)
    if name == "example_ring":
        return ExampleModel(10, 5, 2, "ring", gossip=False)
    if name == "consensus_ring":
        return ConsensusModel(5, 10, "ring", condition_number=10000)
    if name == "vfl_small":
        return VFL(
            7,
            lmbd=1e-2,
            title="mushrooms",
            sample_size=100,
            graph="erdos-renyi",
            edge_prob=5 / 100,
            gossip=False,
            labels_distribution=False,
        )
    raise ValueError(f"Unknown model: {name}")


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(b)), 1e-300)
    return float(np.linalg.norm(a - b) / denominator)


def compare_model(name: str, steps: int, seed: int) -> Dict[str, object]:
    try:
        seed_everything(seed)
        model = make_model(name)
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {"model": name, "skipped": True, "error": f"{type(exc).__name__}: {exc}"}

    start = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            official = Main(steps, model)
        official_wall = time.time() - start
    except Exception as exc:  # pragma: no cover - official code is known to fail on APAPC only
        return {"model": name, "skipped": True, "error": f"official Main: {type(exc).__name__}: {exc}"}

    start = time.time()
    independent = my_main(steps, model)
    independent_wall = time.time() - start

    x_err_error = relative_error(independent["x_err"], official["x_err"])
    x_final_error = relative_error(independent["x"], official["x"])
    counts_equal = {
        key: bool(np.array_equal(np.asarray(independent[key]), np.asarray(official[key])))
        for key in ("grad_calls", "mults_A", "communications")
    }
    true_ratio = float(
        np.sum(independent["communications_true"]) / max(1, np.sum(official["communications"]))
    )

    return {
        "model": name,
        "n": int(model.n),
        "m": int(model.m),
        "dim": int(model.dim),
        "steps": int(steps),
        "kappa_f": float(model.kappa_f),
        "kappa_A": float(model.kappa_A),
        "code_kappa_W": float(model.kappa_W),
        "paper_kappa_W_if_sqrt_convention": float(np.sqrt(model.kappa_W)),
        "mu_G": float(independent["mu_G"][0]),
        "L_G": float(independent["L_G"][0]),
        "kappa_G": float(independent["kappa_G"][0]),
        "x_err_relative_error": x_err_error,
        "final_x_relative_error": x_final_error,
        "final_x_err_official": float(official["x_err"][-1]),
        "final_x_err_independent": float(independent["x_err"][-1]),
        "counts_equal": counts_equal,
        "official_communications_per_step": int(independent["communications"][0]),
        "true_communications_per_step": int(independent["communications_true"][0]),
        "true_over_official_communication_ratio": true_ratio,
        "official_wall_seconds": float(official_wall),
        "independent_wall_seconds": float(independent_wall),
        "passed": bool(
            x_err_error <= 1e-8
            and x_final_error <= 1e-8
            and all(counts_equal.values())
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M2: compare independent Main with methods.Main")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["example_er", "example_ring", "consensus_ring", "vfl_small"],
        choices=["example_er", "example_ring", "consensus_ring", "vfl_small"],
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--output", type=str, default="m2_compare.json")
    args = parser.parse_args(argv)

    ensure_output_dirs()
    reports = [compare_model(name, args.steps, args.seed) for name in args.models]

    print("M2: independent Main vs official methods.Main")
    print("=" * 78)
    for report in reports:
        if report.get("skipped"):
            print(f"[{report['model']}] SKIPPED: {report['error']}")
            continue
        print(
            f"[{report['model']}] steps={report['steps']} "
            f"x_err rel={report['x_err_relative_error']:.2e} "
            f"x_final rel={report['final_x_relative_error']:.2e} "
            f"counts={report['counts_equal']} "
            f"-> {'PASS' if report['passed'] else 'FAIL'}"
        )
        print(
            f"  kappa_f={report['kappa_f']:.3f} kappa_A={report['kappa_A']:.3f} "
            f"kappa_W(code)={report['code_kappa_W']:.3f} "
            f"communications/step official={report['official_communications_per_step']} "
            f"true={report['true_communications_per_step']}"
        )

    payload = {
        "seed": args.seed,
        "steps": args.steps,
        "models": reports,
        "all_passed": all(report.get("passed", False) for report in reports),
    }
    path = save_json(payload, args.output)
    print(f"\nReport: {path}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
