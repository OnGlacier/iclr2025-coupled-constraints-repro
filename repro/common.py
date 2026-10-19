"""Shared helpers for the four reproduction modules.

Run every script from the repository root, e.g.::

    python repro/m1_theory/lemma_checks.py
    python repro/m3_figure1/reproduce_figure1.py --smoke

The scripts add the repository root to ``sys.path`` themselves, so importing
the official ``models``/``methods``/``utils`` modules works from any cwd.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPRO_ROOT / "outputs"
FIGS_DIR = OUTPUT_ROOT / "figs"
DUMPS_DIR = OUTPUT_ROOT / "dumps"
REPORTS_DIR = OUTPUT_ROOT / "reports"

METHOD_LABELS = {"Main": "Main algorithm"}

RESOURCE_KEYS = ("grad_calls", "mults_A", "communications")
RESOURCE_LABELS = (
    "Gradient calls",
    r"Mult. by $\mathbf{A}$ and $\mathbf{A}^T$",
    "Communications",
)


def ensure_output_dirs() -> None:
    """Create the output directories used by every module."""
    for directory in (FIGS_DIR, DUMPS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = 307) -> None:
    """Use the paper/notebook seed before constructing any model."""
    np.random.seed(seed)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def save_dump(obj, name: str) -> Path:
    """Pickle a results dictionary into ``repro/outputs/dumps``."""
    ensure_output_dirs()
    path = DUMPS_DIR / name
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_dump(name: str):
    with open(DUMPS_DIR / name, "rb") as handle:
        return pickle.load(handle)


def save_json(obj, name: str) -> Path:
    """Save a JSON report with numpy-aware serialisation."""
    ensure_output_dirs()
    path = REPORTS_DIR / name
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)
    return path


def configure_plots() -> None:
    """Match the style used by the official notebooks."""
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 2,
        }
    )


def plot_resource_panels(
    results: Mapping[str, Mapping[str, np.ndarray]],
    filename: str,
    *,
    figsize: Tuple[float, float] = (14, 4),
    title: Optional[str] = None,
    xlims: Optional[Mapping[str, Tuple[float, float]]] = None,
    ylabel: str = r"$\| x^k - x^* \|_2^2$",
) -> Sequence[Path]:
    """Plot ``x_err`` against the three resource counters.

    ``results`` must map method names to dictionaries containing ``x_err`` and
    at least one of ``grad_calls``, ``mults_A`` and ``communications``.
    The function writes both a PDF and a PNG into ``repro/outputs/figs``.
    """
    ensure_output_dirs()
    configure_plots()

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, key, xlabel in zip(axes, RESOURCE_KEYS, RESOURCE_LABELS):
        for method, output in results.items():
            if key not in output or "x_err" not in output:
                continue
            x_values = np.cumsum(np.asarray(output[key]))
            y_values = np.asarray(output["x_err"])
            ax.plot(x_values, y_values, label=METHOD_LABELS.get(method, method))

        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        if xlims is not None and key in xlims:
            ax.set_xlim(*xlims[key])
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right", fontsize=9)

    axes[0].set_ylabel(ylabel)
    if title:
        fig.suptitle(title, fontsize=18)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    stem = Path(filename).with_suffix("")
    pdf_path = FIGS_DIR / f"{stem}.pdf"
    png_path = FIGS_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return pdf_path, png_path


def summarise_results(results: Mapping[str, Mapping[str, np.ndarray]]) -> Dict[str, Dict[str, object]]:
    """Small serialisable summary used by the experiment scripts."""
    summary: Dict[str, Dict[str, object]] = {}
    for method, output in results.items():
        entry: Dict[str, object] = {}
        if "x_err" in output and len(output["x_err"]):
            entry["final_x_err"] = float(np.asarray(output["x_err"])[-1])
        for key in RESOURCE_KEYS:
            if key in output:
                entry[f"total_{key}"] = int(np.sum(output[key]))
        if "ts" in output and len(output["ts"]):
            entry["runtime_seconds"] = float(output["ts"][-1])
        summary[method] = entry
    return summary
