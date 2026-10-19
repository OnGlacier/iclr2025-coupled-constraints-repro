"""M4 (extra): VFL with labels distributed across nodes.

This reproduces the second experiment in ``vfl.ipynb`` (not a figure in the
paper, but part of the official code release):

    VFL(num_nodes=2, lmbd=1.0, title='mushrooms', sample_size=1000,
        graph='erdos-renyi', edge_prob=1/1000, gossip=False,
        labels_distribution=True)
    Tracking-ADMM, DPMM and Main for 100 iterations each.

APAPC is omitted because the official implementation requires the attribute
``model.bB`` (and the Model-level ``grad_G`` requires ``model.bW``), neither
of which is ever assigned by ``Model``/``VFL``; running it raises
``AttributeError``.  APAPC is not part of the paper's Figure 2, which compares
Tracking-ADMM, DPMM and Main only.

Run from the repository root::

    python repro/m4_vfl_consensus/labels_distributed.py --smoke
    python repro/m4_vfl_consensus/labels_distributed.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
DEFAULT_METHODS = ["Tracking-ADMM", "DPMM", "Main"]
RESOURCE_KEYS = ("grad_calls", "mults_A", "communications")
RESOURCE_LABELS = (
    "Gradient calls",
    r"Multiplications by $\mathbf{A}$ and $\mathbf{A}^T$",
    "Communications",
)
PAPER_XLIMS = {
    "grad_calls": (0, 10_000),
    "mults_A": (0, 400_000),
    "communications": (0, 100_000),
}
PAPER_YLIM = (1e-17, 1e3)


def build_model(seed: int = 307) -> VFL:
    seed_everything(seed)
    return VFL(
        2,
        lmbd=1.0,
        title="mushrooms",
        sample_size=1000,
        graph="erdos-renyi",
        edge_prob=1 / 1000,
        gossip=False,
        labels_distribution=True,
    )


def run_experiment(
    num_steps: int,
    methods: List[str],
    seed: int = 307,
) -> Tuple[VFL, Dict[str, Dict[str, np.ndarray]]]:
    model = build_model(seed)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for method in methods:
        print(f"[M4-labels] running {method} for {num_steps} iterations ...", flush=True)
        start = time.time()
        output = METHOD_CLASSES[method](num_steps=num_steps, model=model)
        wall = time.time() - start
        output["wall_seconds"] = np.asarray([wall])
        results[method] = output
        print(
            f"      done in {wall:.1f}s; final squared error = "
            f"{float(output['x_err'][-1]):.3e}",
            flush=True,
        )
    return model, results


def plot_labels_distributed(
    results: Dict[str, Dict[str, np.ndarray]],
    *,
    filename: str = "m4_labels_distributed",
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
    parser = argparse.ArgumentParser(description="M4: VFL with distributed labels")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--smoke", action="store_true", help="Use 10 iterations")
    parser.add_argument("--dump", type=str, default="m4_labels_distributed.pkl")
    parser.add_argument("--output", type=str, default="m4_labels_distributed.json")
    args = parser.parse_args(argv)

    ensure_output_dirs()
    num_steps = 10 if args.smoke else args.num_steps

    model, results = run_experiment(num_steps, list(args.methods), seed=args.seed)

    model_info = {
        "n": int(model.n),
        "m": int(model.m),
        "d": int(model.d),
        "dim": int(model.dim),
        "dimensions": [int(dim) for dim in model.dimensions],
        "graph": "erdos-renyi",
        "edge_probability": 1 / 1000,
        "lambda": float(model.lmbd),
        "sample_size": 1000,
        "labels_distribution": True,
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
        },
        args.dump,
    )

    figure_paths = plot_labels_distributed(
        results,
        title=(
            rf"VFL, labels distributed, Erdős–Rényi, $n={model.n}$, "
            rf"$m={model.m}$, $\lambda=1$"
        ),
        paper_limits=not args.smoke,
    )

    summary = summarise_results(results)
    for entry in summary.values():
        entry["iterations"] = int(num_steps)
    payload = {
        "seed": args.seed,
        "num_steps": num_steps,
        "methods": list(args.methods),
        "model": model_info,
        "summary": summary,
        "dump": str(dump_path),
        "figures": [str(path) for path in figure_paths],
        "apapc_omitted": (
            "Official APAPC requires model.bB and Model.grad_G requires model.bW; "
            "neither is assigned, so APAPC raises AttributeError. It is not in "
            "the paper's Figure 2."
        ),
    }
    report_path = save_json(payload, args.output)

    print("\nM4 labels-distributed summary")
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
