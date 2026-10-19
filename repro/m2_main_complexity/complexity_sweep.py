"""M2: empirical complexity sweep against kappa_f, kappa_A and kappa_W.

For each condition number we build a ``ControlledQuadraticModel`` with a
prescribed value, run the explicit Main implementation until
``||x^k - x^*||^2 < target`` and record the cumulative resources at that
iteration.  A log-log least-squares fit gives the empirical exponent.

Expected exponents (Theorem 1):
    gradient calls         ~ kappa_f^{1/2}                 (0 for kappa_A, kappa_W)
    A / A^T multiplications ~ (kappa_f kappa_A)^{1/2}      (0 for kappa_W)
    communications         ~ (kappa_f kappa_A kappa_W)^{1/2}

The official ``methods.Main`` communication counter uses
``ceil(sqrt(kappa_W_prime))`` (a constant ~2) instead of the internal
Chebyshev degree ``ceil(sqrt(model.kappa_W))``.  We therefore also record
``communications_true`` (actual multiplications by W) and fit that quantity;
the official counter is shown in the third panel for comparison.

Run from the repository root::

    python repro/m2_main_complexity/complexity_sweep.py
    python repro/m2_main_complexity/complexity_sweep.py --families kappa_f
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repro.common import (  # noqa: E402
    FIGS_DIR,
    REPORTS_DIR,
    configure_plots,
    ensure_output_dirs,
    save_json,
)
from repro.m2_main_complexity.controlled_model import ControlledQuadraticModel  # noqa: E402
from repro.m2_main_complexity.main_impl import my_main  # noqa: E402

KAPPA_F_VALUES = [10.0, 30.0, 100.0, 300.0, 1000.0]
KAPPA_A_VALUES = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
KAPPA_W_N_VALUES = [5, 8, 10, 12, 16, 20]

RESOURCE_KEYS = ("grad_calls", "mults_A", "communications", "communications_true")
YLABELS = {
    "grad_calls": "Total gradient calls",
    "mults_A": r"Total A/A$^T$ multiplications",
    "communications_true": "Total communications",
}


def make_model(kind: str, value: float, seed: int) -> ControlledQuadraticModel:
    """Build one controlled instance; suppress the official Timer prints."""
    kwargs: Dict[str, float] = {"seed": seed}
    if kind == "kappa_f":
        kwargs.update(kappa_f=value, kappa_A=10.0)
        n, m, d = 10, 5, 2
    elif kind == "kappa_A":
        kwargs.update(kappa_f=100.0, kappa_A=value)
        n, m, d = 10, 5, 2
    elif kind == "kappa_W":
        # Here ``value`` is interpreted as the ring size n.
        kwargs.update(kappa_f=100.0, kappa_A=10.0)
        n, m, d = int(value), 3, 2
    else:
        raise ValueError(kind)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = ControlledQuadraticModel(n, m, d, "ring", None, **kwargs)
        # Model properties are lazy and print a Timer on first access; warm them
        # up here so the sweep output stays clean.
        for attribute in (
            "mu_f",
            "L_f",
            "kappa_f",
            "mu_A",
            "L_A",
            "kappa_A",
            "mu_W",
            "L_W",
            "kappa_W",
            "r",
        ):
            getattr(model, attribute)
    return model


def fit_slope(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


def run_case(
    family: str,
    model: ControlledQuadraticModel,
    parameter: float,
    max_steps: int,
    target: float,
) -> Dict[str, object]:
    output = my_main(max_steps, model)
    reached_mask = output["x_err"] < target
    reached = bool(np.any(reached_mask))
    iterations = int(np.argmax(reached_mask)) + 1 if reached else max_steps

    row: Dict[str, object] = {
        "family": family,
        "iterations": iterations,
        "reached": reached,
        "parameter": float(parameter),
        "kappa_f": float(model.kappa_f),
        "kappa_A": float(model.kappa_A),
        "kappa_W_code": float(model.kappa_W),
        "kappa_W_sqrt_convention": float(np.sqrt(model.kappa_W)),
        "n": int(model.n),
        "m": int(model.m),
        "d": int(model.d),
        "mu_G": float(output["mu_G"][0]),
        "L_G": float(output["L_G"][0]),
        "kappa_G": float(output["kappa_G"][0]),
        "final_x_err": float(output["x_err"][iterations - 1]),
    }
    for key in RESOURCE_KEYS:
        row[key] = int(np.sum(output[key][:iterations]))
    return row


def plot_family(family: str, rows: List[Dict[str, object]], parameter_label: str, target: float) -> List[Path]:
    reached = [row for row in rows if row["reached"]]
    if not reached:
        return []

    configure_plots()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    x_values = [row["parameter"] for row in reached]
    plot_keys = ("grad_calls", "mults_A", "communications_true")
    slopes: Dict[str, Optional[float]] = {}
    for ax, key in zip(axes, plot_keys):
        y_values = [row[key] for row in reached]
        slope = fit_slope(x_values, y_values)
        slopes[key] = slope
        ax.loglog(x_values, y_values, "o-", label="empirical")
        if slope is not None:
            fit_y = np.exp(np.polyval(np.polyfit(np.log(x_values), np.log(y_values), 1), np.log(x_values)))
            ax.loglog(x_values, fit_y, "--", label=f"slope {slope:.2f}")
        ax.set_xlabel(parameter_label, fontsize=13)
        ax.set_ylabel(YLABELS[key], fontsize=13)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    # Also show the official communication counter on the third panel.
    axes[2].loglog(
        x_values,
        [row["communications"] for row in reached],
        "s:",
        label="official counter",
    )
    axes[2].legend(fontsize=9)

    fig.suptitle(
        f"{family} sweep: total resources until squared error < {target:g}",
        fontsize=15,
    )
    stem = f"m2_sweep_{family}"
    pdf_path = FIGS_DIR / f"{stem}.pdf"
    png_path = FIGS_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return [pdf_path, png_path]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M2: complexity sweep for Main")
    parser.add_argument("--families", nargs="+", default=["kappa_f", "kappa_A", "kappa_W"])
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--target", type=float, default=1e-8, help="Target squared error ||x-x*||^2")
    parser.add_argument("--seed", type=int, default=307)
    args = parser.parse_args(argv)

    ensure_output_dirs()
    families = {
        "kappa_f": (KAPPA_F_VALUES, r"$\kappa_f$"),
        "kappa_A": (KAPPA_A_VALUES, r"$\kappa_A$"),
        "kappa_W": (KAPPA_W_N_VALUES, r"$\kappa_W$ (code convention)"),
    }

    rows: List[Dict[str, object]] = []
    slopes: Dict[str, Dict[str, Optional[float]]] = {}
    figure_paths: List[Path] = []

    for family in args.families:
        if family not in families:
            raise ValueError(f"Unknown family: {family}")
        values, parameter_label = families[family]
        family_rows: List[Dict[str, object]] = []
        print(f"\n[{family}] target squared error < {args.target:g}")
        for value in values:
            model = make_model(family, value, args.seed)
            parameter = {
                "kappa_f": float(model.kappa_f),
                "kappa_A": float(model.kappa_A),
                "kappa_W": float(model.kappa_W),
            }[family]
            row = run_case(family, model, parameter, args.max_steps, args.target)
            family_rows.append(row)
            status = "reached" if row["reached"] else "NOT reached"
            print(
                f"  {family}={row['parameter']:.3f} n={row['n']} d={row['d']} "
                f"iterations={row['iterations']} {status}: "
                f"grad={row['grad_calls']} A={row['mults_A']} "
                f"comm_true={row['communications_true']} comm_official={row['communications']}"
            )

        family_slopes = {
            key: fit_slope(
                [row["parameter"] for row in family_rows if row["reached"]],
                [row[key] for row in family_rows if row["reached"]],
            )
            for key in RESOURCE_KEYS
        }
        slopes[family] = family_slopes
        print("  fitted log-log slopes: " + ", ".join(
            f"{key}={value:.2f}" if value is not None else f"{key}=n/a"
            for key, value in family_slopes.items()
        ))
        figure_paths.extend(plot_family(family, family_rows, parameter_label, args.target))
        rows.extend(family_rows)

    # CSV for easy inspection.
    csv_path = REPORTS_DIR / "m2_complexity_sweep.csv"
    fieldnames = [
        "family",
        "parameter",
        "kappa_f",
        "kappa_A",
        "kappa_W_code",
        "kappa_W_sqrt_convention",
        "n",
        "m",
        "d",
        "iterations",
        "reached",
        "grad_calls",
        "mults_A",
        "communications",
        "communications_true",
        "final_x_err",
        "mu_G",
        "L_G",
        "kappa_G",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    payload = {
        "seed": args.seed,
        "target_squared_error": args.target,
        "max_steps": args.max_steps,
        "note": (
            "kappa_W_code is model.kappa_W = (lambda_max(W)/lambda_min_plus(W))^2; "
            "the paper's Figure 1 reports this value (88.66 for the example_er model). "
            "communications_true counts the internal Chebyshev degree of mulW_prime; "
            "communications is the official methods.Main counter."
        ),
        "rows": rows,
        "slopes": slopes,
        "figures": [str(path) for path in figure_paths],
    }
    json_path = save_json(payload, "m2_complexity_sweep.json")
    print(f"\nCSV: {json_path.with_name('m2_complexity_sweep.csv')}")
    print(f"JSON: {json_path}")
    for path in figure_paths:
        print(f"Figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
