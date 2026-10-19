"""A controlled quadratic model for the M2 complexity sweep.

The model has exactly prescribed condition numbers:

* ``f_i(x) = 1/2 x^T C_i^T C_i x - (C_i^T d_i)^T x`` with
  ``C_i^T C_i = Q_i diag(lambda_1, ..., lambda_d) Q_i^T`` and
  ``lambda = geomspace(1, kappa_f, d)``; after subtracting theta the Hessian
  has eigenvalues in ``[1-theta, kappa_f-theta]``;
* ``A_i = U diag(sigma) V^T`` with ``sigma = sqrt(geomspace(1, kappa_A, r))``;
  hence ``mu_A = 1``, ``L_A = kappa_A``;
* the graph is a ring or Erdos-Renyi graph, so ``kappa_W`` is determined by
  the official gossip matrix construction.

The KKT system is solved once in the constructor to obtain the reference
solution used by ``x_err``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg as sla

from models import Model


class ControlledQuadraticModel(Model):
    """Quadratic objective with controlled ``kappa_f`` and ``kappa_A``."""

    def __init__(
        self,
        n: int,
        m: int,
        d: int,
        graph: str = "ring",
        edge_prob: float = None,
        *,
        kappa_f: float = 100.0,
        kappa_A: float = 10.0,
        seed: int = 0,
        theta: float = 1e-3,
    ) -> None:
        super().__init__(n, m, d, graph, edge_prob, gossip=False)
        self.model_type = "ControlledQuadratic"
        self.dimensions = [d] * n
        self.theta = theta

        rng = np.random.default_rng(seed)

        # Objective: f_i(x) = 1/2 ||C_i x||^2 - (C_i^T d_i)^T x.
        eigenvalues = np.geomspace(1.0, kappa_f, d)
        self.C = []
        for _ in range(n):
            q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            self.C.append(q @ np.diag(np.sqrt(eigenvalues - theta)) @ q.T)
        self.bC = sla.block_diag(*self.C)
        self.d_ = [rng.standard_normal(d) for _ in range(n)]
        self.bd = np.hstack(self.d_)
        self._bCT_bC = self.bC.T @ self.bC
        self._linear = self.bC.T @ self.bd
        self._bCT_bd = self._linear

        # Constraints: A_i = U diag(sigma) V^T, identical across nodes.
        rank = min(m, d)
        sigma = np.sqrt(np.geomspace(1.0, kappa_A, rank))
        u, _ = np.linalg.qr(rng.standard_normal((m, rank)))
        v, _ = np.linalg.qr(rng.standard_normal((d, rank)))
        a0 = u @ np.diag(sigma) @ v.T
        self.A = [a0.copy() for _ in range(n)]
        self.bA = sla.block_diag(*self.A)
        self.b = [a0 @ rng.standard_normal(d) for _ in range(n)]
        self.bb = np.hstack(self.b)
        self.A_hstacked = np.hstack(self.A)
        self.b_sum = np.sum(self.b, axis=0)
        # The paper's bW is not needed by this model; keep bB absent like the
        # official models (Main uses bA/bW constructions internally).

        self.solution = self._solve_kkt()

    # ------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------

    def F(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self._bCT_bC @ x - self._linear @ x)

    def grad_F(self, x: np.ndarray) -> np.ndarray:
        return self._bCT_bC @ x - self._linear

    def hess_F(self) -> np.ndarray:
        return self._bCT_bC

    @property
    def bCT_bd(self) -> np.ndarray:
        return self._linear

    # ------------------------------------------------------------------
    # Reference solution
    # ------------------------------------------------------------------

    def _solve_kkt(self) -> Tuple[np.ndarray, float]:
        hessian = self._bCT_bC
        a_matrix = self.A_hstacked
        kkt = np.block(
            [
                [hessian, a_matrix.T],
                [a_matrix, np.zeros((a_matrix.shape[0], a_matrix.shape[0]))],
            ]
        )
        rhs = np.hstack((self._linear, self.b_sum))
        solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        x_star = solution[: self.dim]
        return x_star, self.F(x_star)
