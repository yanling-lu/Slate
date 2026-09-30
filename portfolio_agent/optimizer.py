"""Linear program that turns alphas into weights under every limit at once.

    maximise   sum(alpha * w) - trade_penalty * sum|w - w0|
    subject to w = long - short, 0 <= long <= ub_long, 0 <= short <= ub_short
               sum(long + short) <= gross_max
               net_lo <= sum(w) <= net_hi
               |sum(beta * w)| <= beta_tol
               |sum(w in sector g)| <= sector_tol   for every sector
               sum|w - w0| <= turnover_two_way
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

EPS = 1e-6


def solve(alpha, w0, ub_long, ub_short, beta, sectors, *, gross_max, net_lo, net_hi,
          beta_tol, sector_tol, turnover_two_way, trade_penalty):
    """Returns the weight vector, or None if the constraints are infeasible."""
    n = len(alpha)
    one, zero = np.ones(n), np.zeros(n)
    # variables: [long, short, buy, sell]
    c = np.concatenate([-alpha + EPS, alpha + EPS, one * trade_penalty, one * trade_penalty])
    eye = sparse.identity(n, format="csr")
    a_eq = sparse.hstack([eye, -eye, -eye, eye], format="csr")

    rows, rhs = [], []

    def add(long_coef, short_coef, trade_coef, bound):
        rows.append(np.concatenate([long_coef, short_coef, trade_coef, trade_coef]))
        rhs.append(bound)

    add(one, one, zero, gross_max)
    add(one, -one, zero, net_hi)
    add(-one, one, zero, -net_lo)
    add(beta, -beta, zero, beta_tol)
    add(-beta, beta, zero, beta_tol)
    for g in np.unique(sectors):
        ind = (sectors == g).astype(float)
        add(ind, -ind, zero, sector_tol)
        add(-ind, ind, zero, sector_tol)
    if turnover_two_way is not None:
        if turnover_two_way < 0:
            return None
        add(zero, zero, one, turnover_two_way)

    bounds = ([(0, u) for u in ub_long] + [(0, u) for u in ub_short] + [(0, None)] * (2 * n))
    res = linprog(c, A_ub=sparse.csr_matrix(np.vstack(rows)), b_ub=np.array(rhs),
                  A_eq=a_eq, b_eq=w0, bounds=bounds, method="highs")
    if res.status != 0:
        return None
    w = res.x[:n] - res.x[n:2 * n]
    w[np.abs(w) < EPS] = 0.0
    return w
