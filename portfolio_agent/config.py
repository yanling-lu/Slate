"""Mandate (competition rules) and policy (team's tighter internal limits)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mandate:
    """Hard limits from the challenge's Trading Criteria. Never relaxed."""

    min_positions: int = 100   # long + short names together
    max_positions: int = 500
    max_gross: float = 2.00    # 200% of NAV
    min_net: float = -0.50     # at every rebalancing date
    max_net: float = 0.50


@dataclass
class Policy:
    """Internal targets, all stricter than the mandate. Weights are fractions of NAV."""

    # --- exposure ---------------------------------------------------------
    gross_target: float = 1.90      # buffer under the 200% cap
    net_target: float = 0.0         # dollar neutral
    net_band: float = 0.0           # allowed |net - net_target| at rebalance
    max_weight: float = 0.0075      # per-name cap -> >= gross_target / max_weight names
    # --- neutrality beyond dollar ------------------------------------------
    beta_tol: float = 0.05          # |sum(w * beta_60m)|, ex ante
    sector_tol: float = 0.05        # |net weight| per GICS sector
    sector_neutral_alpha: bool = True
    # --- turnover ----------------------------------------------------------
    # One-way turnover per month = 0.5 * sum|w_new - w_drifted|, fraction of NAV.
    # Full replacement of a 190% gross book is 1.90.
    max_turnover: float = 0.30
    trade_penalty: float = 0.02     # alpha z-units charged per unit traded (hysteresis)
    # --- tradable universe (formation-month information only) --------------
    size_groups: tuple = ("mega", "large", "small")
    min_price: float = 5.0
    min_dolvol: float = 2e6         # dolvol_126d, USD per day
    short_size_groups: tuple = ("mega", "large", "small")
    short_min_dolvol: float = 5e6   # shorts must be more liquid (borrow)
    # --- signal combination ------------------------------------------------
    source_weights: dict = field(default_factory=dict)  # source -> weight, default 1
    # Sparse, already-scaled sources (e.g. the LLM signal agent, score in [-2, 2]) are
    # divided by this constant instead of being re-standardised across the few names
    # they cover. source -> scale.
    fixed_scale_sources: dict = field(default_factory=lambda: {"signal_agent": 1.0})
    winsor_z: float = 3.0
    # --- return accounting (backtest only) ---------------------------------
    missing_return_fill: float = 0.0  # excess return assumed when a held name has no outcome
    tcost_bps: float = 0.0            # per unit of two-way traded notional

    def validate(self, mandate: Mandate) -> None:
        if not 0 < self.gross_target <= mandate.max_gross:
            raise ValueError("gross_target must be in (0, mandate.max_gross]")
        lo, hi = self.net_target - self.net_band, self.net_target + self.net_band
        if lo < mandate.min_net or hi > mandate.max_net:
            raise ValueError("policy net range must sit inside the mandate band")
        if self.gross_target / self.max_weight < mandate.min_positions:
            raise ValueError(
                "max_weight too large: a full book could hold fewer than "
                f"{mandate.min_positions} names"
            )


# Soft limits are relaxed in this order when a month is infeasible. Each step is
# logged in the guardrail report. Mandate limits are in every step.
RELAX_LADDER = (
    {},
    {"sector": 2.0},
    {"sector": 2.0, "beta": 2.0},
    {"sector": 2.0, "beta": 2.0, "turnover": 1.5},
    {"sector": 2.0, "beta": 2.0, "turnover": None},  # None = unconstrained
)
