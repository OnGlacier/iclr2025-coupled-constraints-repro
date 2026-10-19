"""M2: an independent, explicit implementation of the paper's Main algorithm.

The official ``methods.Main`` calls ``grad_G``/``K_Chebyshev``/``mulW_prime``,
which make the per-iteration cost implicit.  Here the Chebyshev operators are
materialised as dense matrices once, so the algorithm can be read directly
against Algorithm 1/2 of the paper.  The numerical output matches the official
implementation to round-off.

The function also records two communication counters:

* ``communications``: the counter used by the official ``methods.Main``;
* ``communications_true``: the number of actual multiplications by W, i.e.
  including the internal degree of the Chebyshev ``mulW_prime`` call.

The second one is the appropriate quantity for checking Theorem 1.  See
``README.md`` in this module for the discrepancy.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

MU_W_PRIME = (11.0 / 15.0) ** 2
L_W_PRIME = (19.0 / 15.0) ** 2
KAPPA_W_PRIME = L_W_PRIME / MU_W_PRIME
MU_K = 11.0 / 15.0
L_K = 19.0 / 15.0


def chebyshev_T(degree: int, x: float) -> float:
    """Chebyshev polynomial T_degree(x), evaluated stably.

    For x < -1 we use T_k(x) = (-1)^k cosh(k arccosh(-x)).  This branch is not
    needed on the theoretical interval [mu, L], but protects the construction
    against a numerical eigenvalue slightly outside the bound.
    """
    if x > 1.0:
        return float(np.cosh(degree * np.arccosh(x)))
    if x < -1.0:
        return float(((-1.0) ** degree) * np.cosh(degree * np.arccosh(-x)))
    return float(np.cos(degree * np.arccos(np.clip(x, -1.0, 1.0))))


def chebyshev_P(t: float, L: float, mu: float, degree: int) -> float:
    """P_M(t) = 1 - T_l((L+mu-2t)/(L-mu)) / T_l((L+mu)/(L-mu))."""
    return 1.0 - chebyshev_T(degree, (L + mu - 2.0 * t) / (L - mu)) / chebyshev_T(
        degree, (L + mu) / (L - mu)
    )


def build_operators(model, use_cache: bool = True) -> Dict[str, object]:
    """Materialise W', B, K and b' used by Main.

    The constructions follow ``methods.mulW_prime`` and ``methods.K_Chebyshev``
    exactly, including the official convention that ``model.mu_W`` and
    ``model.L_W`` are squared eigenvalues of W.
    """
    cached = getattr(model, "_repro_main_operators", None)
    if use_cache and cached is not None:
        return cached

    eigvals, eigvecs = np.linalg.eigh(model.W)
    positive = eigvals[eigvals > 1e-10 * max(1.0, float(eigvals.max()))]
    mu_w_scalar = float(positive.min())
    L_w_scalar = float(eigvals.max())
    degree_w = int(np.ceil(np.sqrt(model.kappa_W)))

    w_prime = eigvecs @ np.diag(
        [chebyshev_P(t, L_w_scalar, mu_w_scalar, degree_w) for t in eigvals]
    ) @ eigvecs.T

    gamma = float(np.sqrt((model.mu_A + model.L_A) / MU_W_PRIME))
    w_lifted = np.kron(w_prime, np.eye(model.m))
    b_matrix = np.hstack((model.bA, gamma * w_lifted))

    mu_b = float(model.mu_A / 2.0)
    L_b = float(model.L_A + (model.L_A + model.mu_A) * L_W_PRIME / MU_W_PRIME)
    degree_b = int(np.ceil(np.sqrt(L_b / mu_b)))

    btb_eigvals, btb_eigvecs = np.linalg.eigh(b_matrix.T @ b_matrix)
    p_values = np.array([chebyshev_P(t, L_b, mu_b, degree_b) for t in btb_eigvals])
    k_matrix = btb_eigvecs @ np.diag(np.sqrt(np.maximum(p_values, 0.0))) @ btb_eigvecs.T

    u_feasible = np.linalg.lstsq(b_matrix, model.bb, rcond=None)[0]
    b_prime = k_matrix @ u_feasible

    operators = {
        "W_prime": w_prime,
        "W_lifted": w_lifted,
        "B": b_matrix,
        "K": k_matrix,
        "b_prime": b_prime,
        "gamma": gamma,
        "mu_B": mu_b,
        "L_B": L_b,
        "kappa_B": L_b / mu_b,
        "kappa_W": float(model.kappa_W),
        "kappa_W_prime": float(KAPPA_W_PRIME),
        "n_W": degree_w,
        "n_B": degree_b,
    }
    if use_cache:
        setattr(model, "_repro_main_operators", operators)
    return operators


def my_main(
    num_steps: int,
    model,
    params: Optional[Dict[str, float]] = None,
    *,
    use_cache: bool = True,
) -> Dict[str, np.ndarray]:
    """Explicit implementation of the paper's Main algorithm.

    This mirrors ``methods.Main`` step by step, but replaces the implicit
    ``grad_G`` and ``K_Chebyshev`` calls with dense matrix operations.
    """
    ops = build_operators(model, use_cache=use_cache)
    b_matrix = ops["B"]
    k_matrix = ops["K"]
    b_prime = ops["b_prime"]
    mu_b = float(ops["mu_B"])
    L_b = float(ops["L_B"])
    kappa_b = float(ops["kappa_B"])
    n_w = int(ops["n_W"])
    n_b = int(ops["n_B"])

    mu_g = float(model.mu_f * min(0.5, (model.mu_A + model.L_A) / (4.0 * model.L_A)))
    L_g = float(
        max(
            model.L_f + model.mu_f,
            model.mu_f * (model.mu_A + model.L_A) / model.L_A * L_W_PRIME / MU_W_PRIME,
        )
    )
    kappa_g = L_g / mu_g
    kappa_k = L_K / MU_K

    params = {} if params is None else params
    tau = float(params.get("tau", min(1.0, 0.5 * np.sqrt(kappa_k / kappa_g))))
    eta = float(params.get("eta", 1.0 / (4.0 * tau * L_g)))
    theta = float(params.get("theta", 1.0 / (eta * L_K)))
    alpha = float(params.get("alpha", mu_g))

    assert 0.0 <= tau <= 1.0
    assert alpha > 0.0

    total_dim = model.dim + model.n * model.m
    u = np.zeros(total_dim)
    u_f = np.zeros(total_dim)
    z = np.zeros(total_dim)

    x_star, F_star = model.solution

    x_err = np.zeros(num_steps)
    F_err = np.zeros(num_steps)
    ts = []
    grad_calls = []
    mults_a = []
    communications = []
    communications_true = []

    start = time.time()
    for i in range(num_steps):
        u_g = tau * u + (1.0 - tau) * u_f

        residual = b_matrix @ u_g - model.bb
        gradient = model.r * (b_matrix.T @ residual)
        gradient[: model.dim] += model.grad_F(u_g[: model.dim])

        u_half = (u - eta * (gradient - alpha * u_g + z)) / (1.0 + eta * alpha)
        z = z + theta * (k_matrix.T @ (k_matrix @ u_half - b_prime))

        u_prev = u
        u = (u - eta * (gradient - alpha * u_g + z)) / (1.0 + eta * alpha)
        u_f = u_g + 2.0 * tau / (2.0 - tau) * (u - u_prev)

        ts.append(time.time() - start)
        grad_calls.append(1)
        mults_a.append(2 + 2 * n_b)
        # Exactly the official counter in methods.Main.
        communications.append(int(2 * np.ceil(np.sqrt(KAPPA_W_PRIME)) * (n_b + 1)))
        # Actual number of W multiplications: each mulW_prime call performs
        # n_W Chebyshev iterations, and K_Chebyshev + grad_G make 2*(n_B+1)
        # such calls per outer iteration.
        communications_true.append(int(2 * n_w * (n_b + 1)))

        x_f = u_f[: model.dim]
        x_err[i] = np.linalg.norm(x_f - x_star) ** 2
        F_err[i] = abs(model.F(x_f) - F_star)

    return {
        "x": x_f,
        "x_err": x_err,
        "F_err": F_err,
        "ts": np.asarray(ts),
        "grad_calls": np.asarray(grad_calls),
        "mults_A": np.asarray(mults_a),
        "communications": np.asarray(communications),
        "communications_true": np.asarray(communications_true),
        "tau": np.asarray([tau]),
        "eta": np.asarray([eta]),
        "theta": np.asarray([theta]),
        "alpha": np.asarray([alpha]),
        "mu_G": np.asarray([mu_g]),
        "L_G": np.asarray([L_g]),
        "kappa_G": np.asarray([kappa_g]),
    }
