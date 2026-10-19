"""M4: reproduce Figure 2 of the paper (VFL linear regression).

Configuration (paper Section 6; graph density follows the notebook's VFL
convention):
    VFL(num_nodes=7, lmbd=1e-2, title='mushrooms', sample_size=100,
        graph='erdos-renyi', edge_prob=5/100, gossip=False,
        labels_distribution=False)
    Main: 1_000 iterations
    Tracking-ADMM and DPMM: 100 x 1_000 iterations (as in the notebook's
    scaling for slow baselines; the paper's x-axis reaches ~1e5 gradient
    calls, which matches this scaling).

The paper only states that the graph is Erdős–Rényi.  ``edge_prob=5/100``
follows the notebook's ``average_degree/sample_size`` convention;
``get_ER_W`` resamples until the graph is connected, so the actual n=7 graph
has 6 edges and average degree 1.71 (not 5).  Note that ``vfl.ipynb`` also
contains a larger n=14, sample_size=8124, ring experiment, which is not the
paper's Figure 2.

Figure 2 contains three curves: Tracking-ADMM, DPMM and the Main algorithm.

Run from the repository root::

    python repro/m4_vfl_consensus/reproduce_figure2.py --smoke
    python repro/m4_vfl_consensus/reproduce_figure2.py
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
from models import VFL  # noqa: E402

from repro.common import (  # noqa: E402
    FIGS_DIR,
    configure_plots,
    ensure_output_dirs,
    save_dump,
    save_json,
    seed_everything,
    summarise_results,
)

METHOD_CLASSES = {"Tracking-ADMM": TrackingADMM, "DPMM": DPMM, "Main": Main}
STEP_SCALES = {"Tracking-ADMM": 100, "DPMM": 100, "Main": 1}
DEFAULT_METHODS = ["Tracking-ADMM", "DPMM", "Main"]

RESOURCE_KEYS = ("grad_calls", "mults_A", "communications")
RESOURCE_LABELS = (
    "Gradient calls",
    r"Multiplications by $\mathbf{A}$ and $\mathbf{A}^T$",
    "Communications",
)
# Axis ranges of the paper's Figure 2 (read off the published figure).
PAPER_XLIMS = {
    "grad_calls": (0, 1_000),
    "mults_A": (0, 130_000),
    "communications": (0, 100_000),
}
PAPER_YLIM = (1e-18, 2e2)


def build_model(seed: int = 307) -> VFL:
    """Construct the Figure 2 model exactly as in the paper."""
    seed_everything(seed)
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


def run_experiment(
    num_steps: int,
    methods: Iterable[str],
    seed: int = 307,
    scales: Optional[Dict[str, int]] = None,
) -> Tuple[VFL, Dict[str, Dict[str, np.ndarray]]]:
    model = build_model(seed)
    scales = STEP_SCALES if scales is None else scales
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for method in methods:
        steps = num_steps * scales[method]
        print(f"[M4] running {method} for {steps} iterations ...", flush=True)
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


def plot_figure2(
    results: Dict[str, Dict[str, np.ndarray]],
    *,
    filename: str = "m4_figure2",
    title: Optional[str] = None,
    paper_limits: bool = True,
) -> List[Path]:
    ensure_output_dirs()
    configure_plots()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
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
    parser = argparse.ArgumentParser(description="M4: reproduce Figure 2")
    parser.add_argument("--num-steps", type=int, default=1_000, help="Steps for Main")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
    )
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use 20 steps for every method (no 100x scaling; fast check)",
    )
    parser.add_argument("--dump", type=str, default="m4_figure2.pkl")
    parser.add_argument("--output", type=str, default="m4_figure2.json")
    args = parser.parse_args(argv)

    ensure_output_dirs()
    num_steps = 20 if args.smoke else args.num_steps
    scales = {method: 1 for method in DEFAULT_METHODS} if args.smoke else STEP_SCALES

    model, results = run_experiment(
        num_steps, args.methods, seed=args.seed, scales=scales
    )

    # ``model.W`` is the graph Laplacian; its off-diagonal nonzeros are the
    # edges of the (connected) Erdős–Rényi graph generated by ``get_ER_W``.
    W = np.asarray(model.W)
    adjacency = (np.abs(W) > 1e-12).astype(int)
    np.fill_diagonal(adjacency, 0)
    degrees = adjacency.sum(axis=1)

    model_info = {
        "n": int(model.n),
        "m": int(model.m),
        "d": int(model.d),
        "dim": int(model.dim),
        "graph": "erdos-renyi",
        "edge_prob": 5 / 100,
        "num_edges": int(degrees.sum() // 2),
        "actual_average_degree": float(degrees.mean()),
        "degrees": [int(value) for value in degrees],
        "graph_note": (
            "The paper specifies only an Erdős–Rényi graph. edge_prob=5/100 "
            "follows the notebook's VFL convention average_degree/sample_size; "
            "get_ER_W resamples until the graph is connected."
        ),
        "lambda": float(model.lmbd),
        "sample_size": 100,
        "kappa_f": float(model.kappa_f),
        "kappa_A": float(model.kappa_A),
        "kappa_W_code": float(model.kappa_W),
        "kappa_W_sqrt_convention_if_paper": float(np.sqrt(model.kappa_W)),
    }
    dump_path = save_dump(
        {
            "model": model_info,
            "results": results,
            "num_steps": num_steps,
            "scales": scales,
        },
        args.dump,
    )

    figure_paths = plot_figure2(
        results,
        title=rf"VFL, Erdős–Rényi, $n={model.n}$, $m={model.m}$",
        paper_limits=not args.smoke,
    )

    summary = summarise_results(results)
    for method, entry in summary.items():
        entry["iterations"] = int(num_steps * scales[method])
    payload = {
        "seed": args.seed,
        "num_steps_main": num_steps,
        "scales": scales,
        "methods": list(args.methods),
        "model": model_info,
        "summary": summary,
        "dump": str(dump_path),
        "figures": [str(path) for path in figure_paths],
        "note": (
            "Communication axis uses the official methods counter, matching the "
            "paper/notebook; its known omission of the internal Chebyshev degree "
            "is documented in repro/m2_main_complexity/README.md."
        ),
    }
    report_path = save_json(payload, args.output)

    print("\nM4 Figure 2 summary")
    print("=" * 66)
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
