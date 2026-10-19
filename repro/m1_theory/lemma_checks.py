"""M1: numerical verification of Assumptions 1-3 and Lemmas 1-5.

This module does not run a convergence experiment.  It checks, on the paper's
synthetic models and on the consensus model, that the constants and operators
computed by the official code satisfy the statements used in the analysis:

* Assumptions 1-3 (smoothness/strong convexity of f, spectral constants of A
  and W);
* Lemma 1 (the augmented objective G is strongly convex/smooth on
  R^d x L_m^perp, with the paper's r and gamma);
* Lemma 2 (the singular-value bounds for B = [A, gamma W']);
* Lemma 3 (the Chebyshev polynomial compresses the spectrum to
  [11/15, 19/15]);
* Lemma 4 (W' and K have the stated spectral bounds);
* Lemma 5 (K_Chebyshev and mulW_prime implement the polynomial operators).

The script writes a JSON report to ``repro/outputs/reports/lemma_checks.json``.
It uses only the official ``models.py``/``methods.py``/``utils.py`` modules and
does not modify them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as sla

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import ConsensusModel, ExampleModel, Model, VFL  # noqa: E402
from methods import K_Chebyshev, grad_G, mulW_prime  # noqa: E402

from repro.common import (  # noqa: E402
    REPORTS_DIR,
    ensure_output_dirs,
    save_json,
    seed_everything,
)

MU_W_PRIME = (11.0 / 15.0) ** 2
L_W_PRIME = (19.0 / 15.0) ** 2
MU_K = 11.0 / 15.0
L_K = 19.0 / 15.0


# ---------------------------------------------------------------------------
# Scalar polynomial helpers
# ---------------------------------------------------------------------------


def chebyshev_T(degree: int, x: float) -> float:
    """Chebyshev polynomial of the first kind, evaluated stably."""
    if x > 1.0:
        return float(np.cosh(degree * np.arccosh(x)))
    if x < -1.0:
        return float(((-1.0) ** degree) * np.cosh(degree * np.arccosh(-x)))
    return float(np.cos(degree * np.arccos(np.clip(x, -1.0, 1.0))))


def chebyshev_P(t: float, L: float, mu: float, degree: int) -> float:
    """P_M(t) = 1 - T_l((L+mu-2t)/(L-mu)) / T_l((L+mu)/(L-mu))."""
    if L <= mu:
        raise ValueError(f"Expected L > mu, got L={L}, mu={mu}")
    arg_t = (L + mu - 2.0 * t) / (L - mu)
    arg_0 = (L + mu) / (L - mu)
    return 1.0 - chebyshev_T(degree, arg_t) / chebyshev_T(degree, arg_0)


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _positive_eigenvalues(matrix: np.ndarray, rel_tol: float = 1e-10) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(matrix)
    tol = rel_tol * max(1.0, float(np.max(np.abs(eigvals))))
    return eigvals[eigvals > tol]


def basis_perp(n: int, m: int) -> np.ndarray:
    """Orthonormal basis of L_m^perp = {y in (R^m)^n : sum_i y_i = 0}.

    The columns are ``kron(Q0, I_m)`` where ``Q0`` is an orthonormal basis of
    the sum-zero subspace of R^n.
    """
    if n == 1:
        return np.zeros((m, 0))
    q0 = sla.null_space(np.ones((1, n)))
    return np.kron(q0, np.eye(m))


# ---------------------------------------------------------------------------
# Operator construction (the same conventions as methods.Main)
# ---------------------------------------------------------------------------


def build_w_prime(model: Model) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build the dense matrix W' = P_{sqrt(W)}(W) used by mulW_prime.

    Note the official convention: model.mu_W / model.L_W are the *squared*
    smallest/largest non-zero eigenvalues of W, and model.kappa_W is their
    ratio.  ``mulW_prime`` uses degree ceil(sqrt(model.kappa_W)).
    """
    eigvals, eigvecs = np.linalg.eigh(model.W)
    positive = _positive_eigenvalues(model.W)
    mu_scalar = float(positive.min())
    L_scalar = float(eigvals.max())
    degree = int(np.ceil(np.sqrt(model.kappa_W)))
    values = np.array([chebyshev_P(t, L_scalar, mu_scalar, degree) for t in eigvals])
    w_prime = eigvecs @ np.diag(values) @ eigvecs.T
    info = {
        "degree": degree,
        "mu_scalar": mu_scalar,
        "L_scalar": L_scalar,
        "min_positive_eigenvalue": float(values[eigvals > 1e-10 * max(1.0, L_scalar)].min()),
        "max_eigenvalue": float(values.max()),
    }
    return w_prime, info


def build_b_operator(model: Model, w_prime: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """B = [A, gamma W'], with gamma = sqrt((mu_A + L_A) / mu_W')."""
    gamma = float(np.sqrt((model.mu_A + model.L_A) / MU_W_PRIME))
    w_lifted = np.kron(w_prime, np.eye(model.m))
    b_matrix = np.hstack((model.bA, gamma * w_lifted))
    return b_matrix, gamma, w_lifted


def build_k_operator(model: Model, b_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """K = sqrt(P_B(B^T B)) and b' = K u_0 with u_0 a least-squares feasible point."""
    mu_b = float(model.mu_A / 2.0)
    L_b = float(model.L_A + (model.L_A + model.mu_A) * L_W_PRIME / MU_W_PRIME)
    degree = int(np.ceil(np.sqrt(L_b / mu_b)))

    eigvals, eigvecs = np.linalg.eigh(b_matrix.T @ b_matrix)
    p_values = np.array([chebyshev_P(t, L_b, mu_b, degree) for t in eigvals])
    k_matrix = eigvecs @ np.diag(np.sqrt(np.maximum(p_values, 0.0))) @ eigvecs.T

    u_feasible = np.linalg.lstsq(b_matrix, model.bb, rcond=None)[0]
    b_prime = k_matrix @ u_feasible
    return k_matrix, b_prime, mu_b, L_b, degree


def build_operators(model: Model) -> Dict[str, object]:
    w_prime, w_info = build_w_prime(model)
    b_matrix, gamma, w_lifted = build_b_operator(model, w_prime)
    k_matrix, b_prime, mu_b, L_b, k_degree = build_k_operator(model, b_matrix)
    return {
        "W_prime": w_prime,
        "W_lifted": w_lifted,
        "B": b_matrix,
        "K": k_matrix,
        "b_prime": b_prime,
        "gamma": gamma,
        "W_info": w_info,
        "mu_B": mu_b,
        "L_B": L_b,
        "K_degree": k_degree,
    }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _max_abs(value) -> float:
    return float(np.max(np.abs(value))) if np.size(value) else 0.0


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(b)), 1e-300)
    return float(np.linalg.norm(a - b) / denominator)


def check_assumptions(model: Model, rng: np.random.Generator) -> Dict[str, object]:
    hessian = _dense(model.hess_F())
    hess_eigvals = np.linalg.eigvalsh(hessian)
    mu_f_actual = float(hess_eigvals.min())
    L_f_actual = float(hess_eigvals.max())

    b_s = sum(a_i @ a_i.T for a_i in model.A) / model.n
    mu_a_actual = float(_positive_eigenvalues(b_s).min())
    L_a_actual = float(max(np.linalg.eigvalsh(a_i.T @ a_i).max() for a_i in model.A))

    w_eigvals = np.linalg.eigvalsh(model.W)
    mu_w_actual = float(_positive_eigenvalues(model.W).min() ** 2)
    L_w_actual = float(w_eigvals.max() ** 2)

    x1 = rng.standard_normal(model.dim)
    x2 = rng.standard_normal(model.dim)
    delta = x2 - x1
    bregman = float(model.F(x2) - model.F(x1) - model.grad_F(x1) @ delta)
    upper = float(model.L_f / 2.0 * delta @ delta)
    lower = float(model.mu_f / 2.0 * delta @ delta)

    scale = max(1.0, abs(upper), abs(lower), abs(bregman))
    f_violation = max(bregman - upper, lower - bregman, 0.0) / scale

    checks = {
        "L_f": {
            "value": float(model.L_f),
            "direct": L_f_actual,
            "relative_error": abs(model.L_f - L_f_actual) / max(1.0, abs(L_f_actual)),
        },
        "mu_f": {
            "value": float(model.mu_f),
            "direct": mu_f_actual,
            "relative_error": abs(model.mu_f - mu_f_actual) / max(1.0, abs(mu_f_actual)),
        },
        "L_A": {
            "value": float(model.L_A),
            "direct": L_a_actual,
            "relative_error": abs(model.L_A - L_a_actual) / max(1.0, abs(L_a_actual)),
        },
        "mu_A": {
            "value": float(model.mu_A),
            "direct": mu_a_actual,
            "relative_error": abs(model.mu_A - mu_a_actual) / max(1.0, abs(mu_a_actual)),
        },
        "L_W": {
            "value": float(model.L_W),
            "direct": L_w_actual,
            "relative_error": abs(model.L_W - L_w_actual) / max(1.0, abs(L_w_actual)),
        },
        "mu_W": {
            "value": float(model.mu_W),
            "direct": mu_w_actual,
            "relative_error": abs(model.mu_W - mu_w_actual) / max(1.0, abs(mu_w_actual)),
        },
        "kernel_W_is_span_ones": bool(np.linalg.norm(model.W @ np.ones(model.n)) < 1e-10 * max(1.0, model.L_W)),
        "f_bregman_max_normalised_violation": f_violation,
        "kappa_f": float(model.kappa_f),
        "kappa_A": float(model.kappa_A),
        "code_kappa_W": float(model.kappa_W),
        "paper_kappa_W": float(np.sqrt(model.kappa_W)),
    }
    checks["passed"] = bool(
        max(
            checks["L_f"]["relative_error"],
            checks["mu_f"]["relative_error"],
            checks["L_A"]["relative_error"],
            checks["mu_A"]["relative_error"],
            checks["L_W"]["relative_error"],
            checks["mu_W"]["relative_error"],
        )
        <= 1e-6
        and checks["kernel_W_is_span_ones"]
        and f_violation <= 1e-7
    )
    return checks


def check_lemma1(
    model: Model,
    operators: Dict[str, object],
    rng: np.random.Generator,
    trials: int,
) -> Dict[str, object]:
    b_matrix = operators["B"]
    gamma = operators["gamma"]
    w_lifted = operators["W_lifted"]
    q_perp = basis_perp(model.n, model.m)

    mu_g = float(model.mu_f * min(0.5, (model.mu_A + model.L_A) / (4.0 * model.L_A)))
    L_g = float(
        max(
            model.L_f + model.mu_f,
            model.mu_f * (model.mu_A + model.L_A) / model.L_A * L_W_PRIME / MU_W_PRIME,
        )
    )

    def g_value(u: np.ndarray) -> float:
        return float(model.F(u[: model.dim]) + model.r / 2.0 * np.linalg.norm(b_matrix @ u - model.bb) ** 2)

    def g_grad(u: np.ndarray) -> np.ndarray:
        residual = model.r * (b_matrix @ u - model.bb)
        return np.hstack((model.grad_F(u[: model.dim]) + model.bA.T @ residual, gamma * w_lifted.T @ residual))

    max_upper = 0.0
    max_lower = 0.0
    for _ in range(trials):
        x1 = rng.standard_normal(model.dim)
        x2 = rng.standard_normal(model.dim)
        v1 = rng.standard_normal(q_perp.shape[1])
        v2 = rng.standard_normal(q_perp.shape[1])
        u1 = np.hstack((x1, q_perp @ v1))
        u2 = np.hstack((x2, q_perp @ v2))
        delta = u2 - u1
        squared_norm = float(delta @ delta)
        if squared_norm == 0.0:
            continue
        bregman = g_value(u2) - g_value(u1) - g_grad(u1) @ delta
        max_upper = max(max_upper, (bregman - 0.5 * L_g * squared_norm) / (0.5 * L_g * squared_norm))
        max_lower = max(max_lower, (0.5 * mu_g * squared_norm - bregman) / (0.5 * mu_g * squared_norm))

    # Finite-difference check of the gradient formula.
    u = np.hstack((rng.standard_normal(model.dim), q_perp @ rng.standard_normal(q_perp.shape[1])))
    analytic = g_grad(u)
    eps = 1e-6
    finite_difference = np.zeros_like(u)
    for j in range(len(u)):
        direction = np.zeros_like(u)
        direction[j] = eps
        finite_difference[j] = (g_value(u + direction) - g_value(u - direction)) / (2.0 * eps)
    fd_error = _relative_error(finite_difference, analytic)

    return {
        "mu_G": mu_g,
        "L_G": L_g,
        "kappa_G": L_g / mu_g,
        "max_normalised_upper_violation": float(max_upper),
        "max_normalised_lower_violation": float(max_lower),
        "gradient_finite_difference_relative_error": fd_error,
        "passed": bool(max_upper <= 1e-7 and max_lower <= 1e-7 and fd_error <= 1e-4),
    }


def check_lemma2(model: Model, operators: Dict[str, object]) -> Dict[str, object]:
    b_matrix = operators["B"]
    gamma = operators["gamma"]
    w_lifted = operators["W_lifted"]
    q_perp = basis_perp(model.n, model.m)

    # Restrict B to the domain R^d x L_m^perp used in the analysis.
    b_restricted = np.hstack((model.bA, gamma * w_lifted @ q_perp))
    eigvals = np.linalg.eigvalsh(b_restricted.T @ b_restricted)
    positive = eigvals[eigvals > 1e-9 * max(1.0, float(eigvals.max()))]
    mu_b_actual = float(positive.min())
    L_b_actual = float(eigvals.max())
    mu_b = float(operators["mu_B"])
    L_b = float(operators["L_B"])

    return {
        "mu_B_theory": mu_b,
        "mu_B_actual": mu_b_actual,
        "L_B_theory": L_b,
        "L_B_actual": L_b_actual,
        "kappa_B_theory": L_b / mu_b,
        "kappa_B_actual": L_b_actual / mu_b_actual,
        "min_positive_eigenvalue_ratio_to_mu_B": mu_b_actual / mu_b,
        "max_eigenvalue_ratio_to_L_B": L_b_actual / L_b,
        "passed": bool(mu_b_actual >= mu_b * (1.0 - 1e-8) and L_b_actual <= L_b * (1.0 + 1e-8)),
    }


def check_lemma3(model: Model, operators: Dict[str, object]) -> Dict[str, object]:
    w_info = operators["W_info"]
    degree_w = int(w_info["degree"])
    mu_w_scalar = float(w_info["mu_scalar"])
    L_w_scalar = float(w_info["L_scalar"])

    mu_b = float(operators["mu_B"])
    L_b = float(operators["L_B"])
    degree_b = int(operators["K_degree"])

    grid_w = np.linspace(mu_w_scalar, L_w_scalar, 2000)
    p_w = np.array([chebyshev_P(t, L_w_scalar, mu_w_scalar, degree_w) for t in grid_w])
    grid_b = np.geomspace(mu_b, L_b, 2000)
    p_b = np.array([chebyshev_P(t, L_b, mu_b, degree_b) for t in grid_b])

    tolerance = 1e-10
    return {
        "W_degree": degree_w,
        "W_polynomial_min": float(p_w.min()),
        "W_polynomial_max": float(p_w.max()),
        "W_polynomial_at_zero": float(chebyshev_P(0.0, L_w_scalar, mu_w_scalar, degree_w)),
        "B_degree": degree_b,
        "B_polynomial_min": float(p_b.min()),
        "B_polynomial_max": float(p_b.max()),
        "B_polynomial_at_zero": float(chebyshev_P(0.0, L_b, mu_b, degree_b)),
        "required_interval": [MU_K, L_K],
        "passed": bool(
            p_w.min() >= MU_K - tolerance
            and p_w.max() <= L_K + tolerance
            and p_b.min() >= MU_K - tolerance
            and p_b.max() <= L_K + tolerance
            and abs(chebyshev_P(0.0, L_w_scalar, mu_w_scalar, degree_w)) <= 1e-12
            and abs(chebyshev_P(0.0, L_b, mu_b, degree_b)) <= 1e-12
        ),
    }


def check_lemma4(model: Model, operators: Dict[str, object]) -> Dict[str, object]:
    w_prime = operators["W_prime"]
    k_matrix = operators["K"]
    w_info = operators["W_info"]

    w_eigvals = np.linalg.eigvalsh(w_prime)
    positive_w = w_eigvals[w_eigvals > 1e-10 * max(1.0, float(w_eigvals.max()))]
    k_singular = np.linalg.svd(k_matrix, compute_uv=False)
    positive_k = k_singular[k_singular > 1e-6 * max(1.0, float(k_singular.max()))]

    return {
        "W_prime_symmetry_error": float(np.linalg.norm(w_prime - w_prime.T)),
        "W_prime_ones_error": float(np.linalg.norm(w_prime @ np.ones(model.n))),
        "W_prime_min_positive": float(positive_w.min()),
        "W_prime_mu_bound": MU_W_PRIME,
        "W_prime_max": float(w_eigvals.max()),
        "W_prime_L_bound": L_W_PRIME,
        "W_prime_dense_max_eigenvalue": float(w_info["max_eigenvalue"]),
        "K_min_positive_singular_value": float(positive_k.min()),
        "K_max_singular_value": float(positive_k.max()),
        "K_mu_bound": MU_K,
        "K_L_bound": L_K,
        "K_symmetry_error": float(np.linalg.norm(k_matrix - k_matrix.T)),
        "passed": bool(
            np.linalg.norm(w_prime - w_prime.T) <= 1e-10
            and np.linalg.norm(w_prime @ np.ones(model.n)) <= 1e-8
            and positive_w.min() >= MU_W_PRIME * (1.0 - 1e-8)
            and w_eigvals.max() <= L_W_PRIME * (1.0 + 1e-8)
            and positive_k.min() >= MU_K * (1.0 - 1e-8)
            and positive_k.max() <= L_K * (1.0 + 1e-8)
            and np.linalg.norm(k_matrix - k_matrix.T) <= 1e-8
        ),
    }


def check_lemma5(
    model: Model,
    operators: Dict[str, object],
    rng: np.random.Generator,
    trials: int,
) -> Dict[str, object]:
    w_lifted = operators["W_lifted"]
    b_matrix = operators["B"]
    k_matrix = operators["K"]
    b_prime = operators["b_prime"]
    gamma = operators["gamma"]

    mul_errors: List[float] = []
    k_errors: List[float] = []
    grad_errors: List[float] = []
    for _ in range(max(1, trials)):
        y = rng.standard_normal(model.n * model.m)
        mul_errors.append(_relative_error(mulW_prime(y, model), w_lifted @ y))

        u = rng.standard_normal(model.dim + model.n * model.m)
        expected_k = k_matrix.T @ (k_matrix @ u - b_prime)
        k_errors.append(_relative_error(K_Chebyshev(u, model), expected_k))

        residual = model.r * (b_matrix @ u - model.bb)
        expected_grad = np.hstack(
            (model.grad_F(u[: model.dim]) + model.bA.T @ residual, gamma * w_lifted.T @ residual)
        )
        grad_errors.append(_relative_error(grad_G(u, model), expected_grad))

    return {
        "mulW_prime_relative_error": float(max(mul_errors)),
        "K_Chebyshev_relative_error": float(max(k_errors)),
        "grad_G_relative_error": float(max(grad_errors)),
        "passed": bool(max(mul_errors) <= 1e-8 and max(k_errors) <= 1e-8 and max(grad_errors) <= 1e-8),
    }


# ---------------------------------------------------------------------------
# Model construction and main
# ---------------------------------------------------------------------------


def make_model(name: str, seed: int) -> Model:
    seed_everything(seed)
    if name == "example_er":
        return ExampleModel(20, 10, 3, "erdos-renyi", edge_prob=5 / 20, gossip=False)
    if name == "example_ring":
        return ExampleModel(10, 5, 2, "ring", gossip=False)
    if name == "consensus_ring":
        return ConsensusModel(5, 10, "ring", condition_number=10000)
    if name == "vfl_small":
        return VFL(
            3,
            lmbd=1e-2,
            title="mushrooms",
            sample_size=100,
            graph="erdos-renyi",
            edge_prob=1.0,
            gossip=False,
            labels_distribution=False,
        )
    raise ValueError(f"Unknown model name: {name}")


def run_model(name: str, seed: int, trials: int) -> Dict[str, object]:
    try:
        model = make_model(name, seed)
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {"model": name, "skipped": True, "error": f"{type(exc).__name__}: {exc}"}

    rng = np.random.default_rng(seed + 1)
    operators = build_operators(model)
    report: Dict[str, object] = {
        "model": name,
        "n": int(model.n),
        "m": int(model.m),
        "dim": int(model.dim),
        "r": float(model.r),
        "gamma_dense": float(operators["gamma"]),
        "assumptions": check_assumptions(model, rng),
        "lemma1": check_lemma1(model, operators, rng, trials),
        "lemma2": check_lemma2(model, operators),
        "lemma3": check_lemma3(model, operators),
        "lemma4": check_lemma4(model, operators),
        "lemma5": check_lemma5(model, operators, rng, trials),
    }
    report["passed"] = bool(
        report["assumptions"]["passed"]
        and report["lemma1"]["passed"]
        and report["lemma2"]["passed"]
        and report["lemma3"]["passed"]
        and report["lemma4"]["passed"]
        and report["lemma5"]["passed"]
    )
    return report


def print_report(report: Dict[str, object]) -> None:
    name = report["model"]
    if report.get("skipped"):
        print(f"[{name}] SKIPPED: {report['error']}")
        return
    print(f"[{name}] n={report['n']} m={report['m']} dim={report['dim']} r={report['r']:.4e}")
    for key in ("assumptions", "lemma1", "lemma2", "lemma3", "lemma4", "lemma5"):
        status = "PASS" if report[key]["passed"] else "FAIL"
        print(f"  {key:12s} {status}")
    print(f"  overall      {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M1: numerical checks for the paper's lemmas")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["example_er", "example_ring", "consensus_ring"],
        choices=["example_er", "example_ring", "consensus_ring", "vfl_small"],
        help="Models on which to run the checks",
    )
    parser.add_argument("--seed", type=int, default=307)
    parser.add_argument("--trials", type=int, default=20, help="Random trials for Bregman/operator checks")
    parser.add_argument("--output", type=str, default="lemma_checks.json")
    args = parser.parse_args(argv)

    ensure_output_dirs()
    reports = [run_model(name, args.seed, args.trials) for name in args.models]

    print("M1: numerical verification of Assumptions 1-3 and Lemmas 1-5")
    print("=" * 68)
    for report in reports:
        print_report(report)

    payload = {
        "seed": args.seed,
        "trials": args.trials,
        "models": reports,
        "all_passed": all(report.get("passed", False) for report in reports),
    }
    path = save_json(payload, args.output)
    print(f"\nReport: {path}")

    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
