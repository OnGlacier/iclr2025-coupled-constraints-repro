"""M4 (supplementary): consensus optimization, Main vs EXTRA.

This is the experiment in ``consensus.ipynb``.  In the paper, consensus
optimization is an application discussed in Section 1/Appendix A rather than
a figure, so this script is a supplementary validation of the reduction.

Configuration:
    ConsensusModel(n=5, d=10, graph='ring', condition_number=10000)
    Main:        10_000 iterations
    EXTRA:       30 x 10_000 iterations

The official notebook cell 1 references an undefined ``average_degree`` and
cell 2 uses ``results`` before assignment; both are fixed here by passing
``edge_prob=None`` and initialising ``results``.

Run from the repository root::

    python repro/m4_vfl_consensus/consensus_extra.py --smoke
    python repro/m4_vfl_consensus/consensus_extra.py
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

from methods import EXTRA, Main  # noqa: E402
from models import ConsensusModel  # noqa: E402

from repro.common import (  # noqa: E402
    FIGS_DIR,
    configure_plots,
    ensure_output_dirs,
    save_dump,
    save_json,
    seed_everything,
    summarise_results,
)

DEFAULT_METHODS = ["Main", "EXTRA"]
STEP_SCALES = {"Main": 1, "EXTRA": 30}


def build_model(seed: int = 307) -> ConsensusModel:
    seed_everything(seed)
    return ConsensusModel(5, 10, "ring", edge_prob=None, gossip=False, condition_number=10_000)


def run_experiment(
    num_steps: int,
    methods: List[str],
    seed: int = 307,
) -> Tuple[ConsensusModel, Dict[str, Dict[str, np.ndarray]]]:
    model = build_model(seed)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for method in methods:
        steps = num_steps * STEP_SCALES[method]
        print(f"[M4-consensus] running {method} for {steps} iterations ...", flush=True)
        start = time.time()
        if method == "Main":
            output = Main(num_steps=steps, model=model)
        elif method == "EXTRA":
            output = EXTRA(num_steps=steps, model=model)
        else:
            raise ValueError(method)
        wall = time.time() - start
        output["wall_seconds"] = np.asarray([wall])
        results[method] = output
        print(
            f"      done in {wall:.1f}s; final squared error = "
            f"{float(output['x_err'][-1]):.3e}",
            flush=True,
        )
    return model, results


def plot_consensus(
    results: Dict[str, Dict[str, np.ndarray]],
    *,
    filename: str = "m4_consensus",
    title: Optional[str] = None,
) -> List[Path]:
    """Two panels (gradient calls and communications), as in the notebook."""
    ensure_output_dirs()
    configure_plots()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    panels = (
        ("grad_calls", "Gradient calls"),
        ("communications", "Communications"),
    )
    for ax, (key, xlabel) in zip(axes, panels):
        for method, output in results.items():
            x_values = np.cumsum(np.asarray(output[key]))
            y_values = np.asarray(output["x_err"])
            label = "Main algorithm" if method == "Main" else method
            ax.plot(x_values, y_values, label=label)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.yaxis.set_tick_params(labelleft=True)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
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
    parser = argparse.ArgumentParser(description="M4: consensus Main vs EXTRA")
    parser.add_argument("--num-steps", type=int, default=10_000, help="Steps for Main")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--smoke", action="store_true", help="Use 100 Main steps")
    parser.add_argument("--dump", type=str, default="m4_consensus.pkl")
    parser.add_argument("--output", type=str, default="m4_consensus.json")
    args = parser.parse_args(argv)

    ensure_output_dirs()
    num_steps = 100 if args.smoke else args.num_steps

    model, results = run_experiment(num_steps, list(args.methods), seed=args.seed)

    model_info = {
        "n": int(model.n),
        "d": int(model.d),
        "m": int(model.m),
        "dim": int(model.dim),
        "graph": "ring",
        "condition_number": 10_000,
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
            "scales": STEP_SCALES,
        },
        args.dump,
    )

    figure_paths = plot_consensus(
        results,
        title=rf"Consensus, ring, $n={model.n}$, $d={model.d}$",
    )

    summary = summarise_results(results)
    for method, entry in summary.items():
        entry["iterations"] = int(num_steps * STEP_SCALES[method])
    payload = {
        "seed": args.seed,
        "num_steps_main": num_steps,
        "scales": STEP_SCALES,
        "methods": list(args.methods),
        "model": model_info,
        "summary": summary,
        "dump": str(dump_path),
        "figures": [str(path) for path in figure_paths],
        "note": (
            "Consensus is an application/theory check (Appendix A) and an extra "
            "notebook experiment, not a paper figure. DPMM does not support "
            "ConsensusModel (official NotImplementedError), so it is omitted."
        ),
    }
    report_path = save_json(payload, args.output)

    print("\nM4 consensus summary")
    print("=" * 66)
    for method, entry in summary.items():
        print(
            f"{method:6s} iterations={entry['iterations']:>7d} "
            f"final x_err={entry['final_x_err']:.3e} "
            f"grad={entry['total_grad_calls']} comm={entry['total_communications']}"
        )
    print(f"\nDump: {dump_path}")
    print(f"Report: {report_path}")
    for path in figure_paths:
        print(f"Figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
