"""M3: reproduce Figure 1 of the paper (example.ipynb, first experiment).

Configuration (identical to the notebook):
    np.random.seed(307)
    ExampleModel(n=20, m=10, d=3, graph='erdos-renyi',
                 edge_prob=5/20, gossip=False)
    Tracking-ADMM: 10 * NUM_STEPS iterations
    DPMM:         100 * NUM_STEPS iterations
    Main:          NUM_STEPS iterations
with NUM_STEPS = 10_000 for the full figure (``--smoke`` uses 100).

The three panels plot the squared error against cumulative gradient calls,
multiplications by A/A^T, and communications, exactly as in the paper.  Note
that the communication axis uses the official ``methods`` counter; see
``repro/m2_main_complexity/README.md`` for its known inconsistency with the
Chebyshev degree.

Run from the repository root::

    python repro/m3_figure1/reproduce_figure1.py --smoke
    python repro/m3_figure1/reproduce_figure1.py
    python repro/m3_figure1/reproduce_figure1.py --replot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods import DPMM, Main, TrackingADMM  # noqa: E402
from models import ExampleModel  # noqa: E402

from repro.common import (  # noqa: E402
    FIGS_DIR,
    configure_plots,
    ensure_output_dirs,
    load_dump,
    save_dump,
    save_json,
    seed_everything,
    summarise_results,
)

METHOD_CLASSES = {"Tracking-ADMM": TrackingADMM, "DPMM": DPMM, "Main": Main}
STEP_SCALES = {"Tracking-ADMM": 10, "DPMM": 100, "Main": 1}
DEFAULT_METHODS = ["Tracking-ADMM", "DPMM", "Main"]

# Axis limits used by the official notebook for the full 10_000-step run.
PAPER_XLIMS = {
    "grad_calls": (0, 10_000),
    "mults_A": (0, 400_000),
    "communications": (0, 400_000),
}
PAPER_YLIM = (1e-21, 1e3)

RESOURCE_KEYS = ("grad_calls", "mults_A", "communications")
RESOURCE_LABELS = (
    "Gradient calls",
    r"Mult. by $\mathbf{A}$ and $\mathbf{A}^T$",
    "Communications",
)


def build_model(seed: int = 307) -> ExampleModel:
    """Construct the Figure 1 model exactly as in example.ipynb."""
    seed_everything(seed)
    return ExampleModel(
        20,
        10,
        3,
        "erdos-renyi",
        edge_prob=5 / 20,
        gossip=False,
    )


def run_experiment(
    num_steps: int,
    methods: Iterable[str],
    seed: int = 307,
) -> Tuple[ExampleModel, Dict[str, Dict[str, np.ndarray]]]:
    """Run the selected methods on one shared model, as in the notebook."""
    model = build_model(seed)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for method in methods:
        steps = num_steps * STEP_SCALES[method]
        print(f"[M3] running {method} for {steps} iterations ...", flush=True)
        start = time.time()
        output = METHOD_CLASSES[method](num_steps=steps, model=model)
        wall = time.time() - start
        output["wall_seconds"] = np.asarray([wall])
        results[method] = output
        print(
            f"      done in {wall:.1f}s; final squared error = "
            f"{float(output['x_err'][-1]):.3e}",
            flush=True,
        )
    return model, results


def plot_figure1(
    results: Dict[str, Dict[str, np.ndarray]],
    *,
    paper_limits: bool = True,
    filename: str = "m3_figure1",
    title: Optional[str] = None,
) -> List[Path]:
    """Three-panel figure matching the official example.ipynb plot."""
    ensure_output_dirs()
    configure_plots()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, key, xlabel in zip(axes, RESOURCE_KEYS, RESOURCE_LABELS):
        for method, output in results.items():
            x_values = np.cumsum(np.asarray(output[key]))
            y_values = np.asarray(output["x_err"])
            label = "Main algorithm" if method == "Main" else method
            ax.plot(x_values, y_values, label=label)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        if paper_limits:
            ax.set_xlim(*PAPER_XLIMS[key])
            if key == "grad_calls":
                ax.set_ylim(*PAPER_YLIM)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right", fontsize=9)

    fig.text(0.0, 0.5, r"$\| x^k - x^* \|_2^2$", va="center", rotation="vertical")
    if title:
        fig.suptitle(title, fontsize=18)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    pdf_path = FIGS_DIR / f"{filename}.pdf"
    png_path = FIGS_DIR / f"{filename}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return [pdf_path, png_path]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M3: reproduce Figure 1")
    parser.add_argument("--num-steps", type=int, default=10_000, help="Steps for Main")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Methods to run",
    )
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 100 Main steps (fast check; disables paper axis limits)",
    )
    parser.add_argument("--replot", action="store_true", help="Replot the saved dump")
    parser.add_argument(
        "--paper-limits",
        action="store_true",
        help="Force the official axis limits even for a short run",
    )
    parser.add_argument("--dump", type=str, default="m3_figure1.pkl")
    parser.add_argument("--output", type=str, default="m3_figure1.json")
    args = parser.parse_args(argv)

    ensure_output_dirs()

    if args.replot:
        payload = load_dump(args.dump)
        results = payload["results"]
        model_info = payload.get("model", {})
        figure_paths = plot_figure1(
            results,
            paper_limits=False,
            title="Figure 1 (replotted from dump)",
        )
        print(f"Replotted {', '.join(results)} from {args.dump}")
        for path in figure_paths:
            print(f"Figure: {path}")
        return 0

    num_steps = 100 if args.smoke else args.num_steps
    paper_limits = (not args.smoke and num_steps >= 10_000) or args.paper_limits

    model, results = run_experiment(num_steps, args.methods, seed=args.seed)

    model_info = {
        "n": int(model.n),
        "m": int(model.m),
        "d": int(model.d),
        "graph": "erdos-renyi",
        "average_degree": 5,
        "kappa_f": float(model.kappa_f),
        "kappa_A": float(model.kappa_A),
        "kappa_W_code": float(model.kappa_W),
        "kappa_W_sqrt_convention_if_paper": float(np.sqrt(model.kappa_W)),
        "r": float(model.r),
    }
    dump_path = save_dump(
        {"model": model_info, "results": results, "num_steps": num_steps},
        args.dump,
    )

    figure_paths = plot_figure1(
        results,
        paper_limits=paper_limits,
        title=(
            rf"Erdős–Rényi, $n={model.n}$, $d_i={model.d}$, $m={model.m}$"
            if paper_limits
            else "Figure 1 (smoke run)"
        ),
    )

    summary = summarise_results(results)
    for method, entry in summary.items():
        entry["iterations"] = int(num_steps * STEP_SCALES[method])
    payload = {
        "seed": args.seed,
        "num_steps_main": num_steps,
        "step_scales": STEP_SCALES,
        "methods": list(args.methods),
        "model": model_info,
        "summary": summary,
        "dump": str(dump_path),
        "figures": [str(path) for path in figure_paths],
        "note": (
            "The communication axis uses the official methods counter "
            "(2*ceil(sqrt(kappa_W_prime))*(n_B+1) per outer step), as the paper does. "
            "It omits the internal Chebyshev degree; see the M2 README."
        ),
    }
    report_path = save_json(payload, args.output)

    print("\nM3 Figure 1 summary")
    print("=" * 60)
    for method, entry in summary.items():
        print(
            f"{method:14s} iterations={entry['iterations']:>7d} "
            f"final x_err={entry['final_x_err']:.3e} "
            f"grad={entry['total_grad_calls']} A={entry['total_mults_A']} "
            f"comm={entry['total_communications']}"
        )
    print(f"\nDump: {dump_path}")
    print(f"Report: {report_path}")
    for path in figure_paths:
        print(f"Figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
