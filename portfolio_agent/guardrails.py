"""Independent verification of a finished book. Knows nothing about how the
weights were produced, so it can also audit a portfolio from any other source."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import Mandate, Policy

TOL = 1e-6


class MandateViolation(RuntimeError):
    """A hard competition limit is broken. Nothing is written."""


@dataclass
class Check:
    name: str
    level: str      # "hard" = competition mandate, "policy" = team limit
    value: float
    limit: str
    ok: bool

    def as_dict(self):
        return asdict(self)


def check_portfolio(w: pd.Series, mandate: Mandate, policy: Policy, *, beta=None,
                    sector=None, turnover=None, short_eligible=None) -> list[Check]:
    """w: signed weights (fraction of NAV) indexed by permno, non-zero names only.
    beta / sector / short_eligible: Series on the same index. turnover: one-way,
    or None for the initial build."""
    out = []

    def add(name, level, value, limit, ok):
        out.append(Check(name, level, float(value), limit, bool(ok)))

    clean = bool(np.isfinite(w).all() and w.index.is_unique and (w != 0).all())
    n, gross, net = len(w), w.abs().sum(), w.sum()
    add("weights_finite_unique", "hard", clean, "True", clean)
    add("positions", "hard", n, f"[{mandate.min_positions}, {mandate.max_positions}]",
        mandate.min_positions <= n <= mandate.max_positions)
    add("gross_exposure", "hard", gross, f"<= {mandate.max_gross}",
        gross <= mandate.max_gross + TOL)
    add("net_exposure", "hard", net, f"[{mandate.min_net}, {mandate.max_net}]",
        mandate.min_net - TOL <= net <= mandate.max_net + TOL)

    add("dollar_neutral", "policy", net - policy.net_target, f"|.| <= {policy.net_band}",
        abs(net - policy.net_target) <= policy.net_band + TOL)
    add("gross_vs_target", "policy", gross,
        f"[{0.9 * policy.gross_target:.3f}, {policy.gross_target}]",
        0.9 * policy.gross_target <= gross <= policy.gross_target + TOL)
    add("max_position", "policy", w.abs().max() if n else 0.0, f"<= {policy.max_weight}",
        n == 0 or w.abs().max() <= policy.max_weight + TOL)
    if turnover is not None:
        add("turnover_one_way", "policy", turnover, f"<= {policy.max_turnover}",
            turnover <= policy.max_turnover + TOL)
    if beta is not None:
        b = float((w * beta.reindex(w.index)).sum())
        add("beta_ex_ante", "policy", b, f"|.| <= {policy.beta_tol}",
            abs(b) <= policy.beta_tol + TOL)
    if sector is not None:
        s = w.groupby(sector.reindex(w.index)).sum()
        worst = s.abs().max() if len(s) else 0.0
        add("max_sector_net", "policy", worst, f"<= {policy.sector_tol}",
            worst <= policy.sector_tol + TOL)
    if short_eligible is not None:
        shorts = w[w < 0]
        bad = shorts[~short_eligible.reindex(shorts.index).fillna(False).astype(bool)]
        share = bad.abs().sum() / shorts.abs().sum() if len(shorts) else 0.0
        add("illiquid_short_share", "policy", share, "== 0", share <= TOL)
    return out


def hard_failures(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.level == "hard" and not c.ok]


def enforce(checks: list[Check], label: str) -> None:
    bad = hard_failures(checks)
    if bad:
        detail = "; ".join(f"{c.name}={c.value:.4g} (limit {c.limit})" for c in bad)
        raise MandateViolation(f"{label}: {detail}")
