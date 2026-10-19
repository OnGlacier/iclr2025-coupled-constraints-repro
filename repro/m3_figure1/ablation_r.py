"""M3: reproduction of the augmentation-parameter (r) ablation.

This mirrors the second experiment in ``example.ipynb``: the same synthetic
model as Figure 1, Main run for 10_000 iterations for each

    r in [1e-7, 1e-6, 2e-5, 1e-3, 1e-1, 4e-1].

The notebook overwrites the cached ``model._r`` for every value; the objective,
constraints and reference solution are unchanged.

Run from the repository root::

    python repro/m3_figure1/ablation_r.py --smoke
    python repro/m3_figure1/ablation_r.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods import Main  # noqa: E402

from repro.common import (  # noqa: E402
    FIGS_DIR,
    configure_plots,
    ensure_output_dirs,
    save_dump,
    save_json,
    seed_everything,
)
from repro.m3_figure1.reproduce_figure1 import build_model  # noqa: E402

DEFAULT_R_VALUES = [1e-7, 1e-6, 2e-5, 1e-3, 1e-1, 4e-1]
RESOURCE_KEYS = ("grad_calls", "mults_A", "communications")
RESOURCE_LABELS = (
    "Gradient calls",
    r"Mult. by $\mathbf{A}$ and $\mathbf{A}^T$",
    "Communications",
)


def run_ablation(
    num_steps: int,
    r_values: Sequence[float],
    seed: int = 307,
) -> Dict[str, object]:
    """Run Main once per r on the Figure 1 model."""
    model = build_model(seed)
    r_initial = float(model.r)
    print(f"[M3] initial r = {r_initial:.3e}")

    results: List[Dict[str, np.ndarray]] = []
    for r in r_values:
        model._r = float(r)
        print(f"[M3] running Main with r = {r:.1e} for {num_steps} iterations ...", flush=True)
        start = time.time()
        output = Main(num_steps=num_steps, model=model)
        wall = time.time() - start
        output["wall_seconds"] = np.asarray([wall])
        results.append(output)
        print(
            f"      done in {wall:.1f}s; final squared error = "
            f"{float(output['x_err'][-1]):.3e}",
            flush=True,
        )
    return {"model": model, "r_initial": r_initial, "results": results}


def plot_ablation(
    r_values: Sequence[float],
    results: Sequence[Dict[str, np.ndarray]],
    *,
    filename: str = "m3_r_ablation",
    title: Optional[str] = None,
) -> List[Path]:
    """Three-panel plot matching the notebook's r-ablation figure."""
    ensure_output_dirs()
    configure_plots()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, key, xlabel in zip(axes, RESOURCE_KEYS, RESOURCE_LABELS):
        for r, output in zip(r_values, results):
            x_values = np.cumsum(np.asarray(output[key]))
            y_values = np.asarray(output["x_err"])
            ax.plot(x_values, y_values, label=rf"$r = {r:.1e}$")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.5), loc="center left")
    fig.text(0.0, 0.5, r"$\| x^k - x^* \|_2^2$", va="center", rotation="vertical")
    if title:
        fig.suptitle(title, fontsize=18)
        fig.tight_layout(rect=(0, 0, 0.9, 0.95))
    else:
        fig.tight_layout(rect=(0, 0, 0.9, 1.0))

    pdf_path = FIGS_DIR / f"{filename}.pdf"
    png_path = FIGS_DIR / f"{filename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return [pdf_path, png_path]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M3: r ablation for Main")
    parser.add_argument("--steps", type=int, default=10_000, help="Main iterations per r")
    parser.add_argument("--r-values", nargs="+", type=float, default=DEFAULT_R_VALUES)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--smoke", action="store_true", help="Use 100 iterations")
    parser.add_argument("--dump", type=str, default="m3_r_ablation.pkl")
    parser.add_argument("--output", type=str, default="m3_r_ablation.json")
    args = parser.parse_args(argv)

    ensure_output_dirs()
    num_steps = 100 if args.smoke else args.steps
    r_values = list(args.r_values)

    run = run_ablation(num_steps, r_values, seed=args.seed)
    model = run["model"]
    results = run["results"]

    dump_path = save_dump(
        {
            "r_values": r_values,
            "results": results,
            "num_steps": num_steps,
            "r_initial": run["r_initial"],
        },
        args.dump,
    )
    figure_paths = plot_ablation(
        r_values,
        results,
        title=(
            rf"$n={model.n}$, $d_i={model.d}$, $m={model.m}$, "
            rf"$\kappa_f={model.kappa_f:.0f}$, $\kappa_A={model.kappa_A:.0f}$"
        ),
    )

    rows = []
    for r, output in zip(r_values, results):
        final_x_err = float(np.asarray(output["x_err"])[-1])
        rows.append(
            {
                "r": float(r),
                "final_x_err": final_x_err,
                "total_grad_calls": int(np.sum(output["grad_calls"])),
                "total_mults_A": int(np.sum(output["mults_A"])),
                "total_communications": int(np.sum(output["communications"])),
                "runtime_seconds": float(np.asarray(output["ts"])[-1]),
                "reached_1e-15": bool(np.any(np.asarray(output["x_err"]) < 1e-15)),
                "diverged": bool(not np.isfinite(final_x_err)),
            }
        )

    payload = {
        "seed": args.seed,
        "num_steps": num_steps,
        "r_initial": run["r_initial"],
        "r_values": r_values,
        "rows": rows,
        "dump": str(dump_path),
        "figures": [str(path) for path in figure_paths],
    }
    report_path = save_json(payload, args.output)

    print("\nM3 r-ablation summary")
    print("=" * 72)
    for row in rows:
        print(
            f"r={row['r']:.1e} final x_err={row['final_x_err']:.3e} "
            f"grad={row['total_grad_calls']} A={row['total_mults_A']} "
            f"comm={row['total_communications']} "
            f"reached 1e-15={row['reached_1e-15']}"
        )
    print(f"\nDump: {dump_path}")
    print(f"Report: {report_path}")
    for path in figure_paths:
        print(f"Figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
