"""Return accounting and the exposure / performance summary.

Capital convention: NAV sits in cash earning the risk-free rate, shorts finance
longs and the short rebate is paid at the same rate, so

    R_total(t+1) = rf(t+1) + sum_i w_i * ret_exc_lead1m_i  - trading costs

Outcomes are joined here, after every month's weights are final.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from .config import Policy
from .data import OUTCOME_COL


def monthly_returns(holdings: pd.DataFrame, monthly: pd.DataFrame, outcomes: pd.DataFrame,
                    rf: pd.Series, policy: Policy, tb3ms: pd.Series | None = None,
                    market: pd.Series | None = None) -> pd.DataFrame:
    """tb3ms: annualised percent by holding month-end (FRED TB3MS).
    market: S&P 500 monthly total return (decimal) by holding month-end."""
    h = holdings.merge(outcomes, how="left", left_on=["PERMNO", "eom"], right_on=["permno", "eom"])
    h["missing"] = h[OUTCOME_COL].isna()
    h["pnl"] = h["WEIGHT"] * h[OUTCOME_COL].fillna(policy.missing_return_fill)
    h["long_pnl"] = h["pnl"].where(h["WEIGHT"] > 0, 0.0)
    h["short_pnl"] = h["pnl"].where(h["WEIGHT"] < 0, 0.0)
    out = h.groupby("eom").agg(long_contribution=("long_pnl", "sum"),
                               short_contribution=("short_pnl", "sum"),
                               names_missing_return=("missing", "sum"))
    out.index = out.index + MonthEnd(1)          # holding month
    out.index.name = "month"
    m = monthly.set_index(monthly["eom"] + MonthEnd(1))
    traded = 2 * m["turnover_one_way"].astype(float).fillna(m["gross"] / 2)
    out["trading_cost"] = traded * policy.tcost_bps / 1e4
    out["rf"] = rf.reindex(out.index)
    out["excess_return"] = out["long_contribution"] + out["short_contribution"] - out["trading_cost"]
    out["total_return"] = out["rf"] + out["excess_return"]
    if tb3ms is not None:
        out["benchmark"] = tb3ms.reindex(out.index) / 100 / 12 + 0.04 / 12
        out["benchmark_source"] = "TB3MS + 4%"
    else:
        out["benchmark"] = out["rf"] + 0.04 / 12
        out["benchmark_source"] = "PROXY: panel rf + 4% (supply TB3MS)"
    out["active_return"] = out["total_return"] - out["benchmark"]
    if market is not None:
        out["market_excess"] = market.reindex(out.index) - out["rf"]
        out["market_source"] = "S&P 500"
    else:
        o = outcomes.dropna(subset=[OUTCOME_COL, "me"])
        vw = (o[OUTCOME_COL] * o["me"]).groupby(o["eom"]).sum() / o["me"].groupby(o["eom"]).sum()
        vw.index = vw.index + MonthEnd(1)
        out["market_excess"] = vw.reindex(out.index)
        out["market_source"] = "PROXY: cap-weighted panel (supply S&P 500)"
    for col in ("gross", "net", "n_long", "n_short", "turnover_one_way"):
        out[col] = m[col].reindex(out.index)
    return out.reset_index()


def summary(returns: pd.DataFrame, monthly: pd.DataFrame, holdings: pd.DataFrame,
            panel: pd.DataFrame) -> dict:
    r = returns.dropna(subset=["total_return"])
    exc, act = r["excess_return"], r["active_return"]
    nav = (1 + r["total_return"]).cumprod()
    x = np.column_stack([np.ones(len(r)), r["market_excess"]])
    coef, res, *_ = np.linalg.lstsq(x, exc.values, rcond=None)
    resid = exc.values - x @ coef
    se = np.sqrt(np.diag(resid @ resid / (len(r) - 2) * np.linalg.inv(x.T @ x)))
    to = monthly["turnover_one_way"].astype(float).dropna()
    shorts = holdings[holdings["WEIGHT"] < 0].merge(
        panel[["permno", "eom", "me", "dolvol_126d", "size_grp"]],
        left_on=["PERMNO", "eom"], right_on=["permno", "eom"], how="left")
    w = holdings["WEIGHT"].abs()
    return {
        "months": int(len(r)),
        "first_month": str(r["month"].min().date()), "last_month": str(r["month"].max().date()),
        "returns_are": "gross of trading costs" if (r["trading_cost"] == 0).all() else "net of trading costs",
        "benchmark_source": r["benchmark_source"].iloc[0],
        "market_source": r["market_source"].iloc[0],
        "mean_monthly_return": float(r["total_return"].mean()),
        "annualized_return_arithmetic": float(r["total_return"].mean() * 12),
        "cagr": float(nav.iloc[-1] ** (12 / len(r)) - 1),
        "cumulative_return": float(nav.iloc[-1] - 1),
        "annualized_vol": float(r["total_return"].std() * np.sqrt(12)),
        "sharpe": float(exc.mean() / exc.std() * np.sqrt(12)),
        "information_ratio_vs_cash_plus_4": float(act.mean() / act.std() * np.sqrt(12)),
        "hit_rate_active": float((act > 0).mean()),
        "max_drawdown": float((nav / nav.cummax() - 1).min()),
        "best_month": [str(r.loc[r["total_return"].idxmax(), "month"].date()), float(r["total_return"].max())],
        "worst_month": [str(r.loc[r["total_return"].idxmin(), "month"].date()), float(r["total_return"].min())],
        "alpha_annualized": float(coef[0] * 12), "alpha_t": float(coef[0] / se[0]),
        "beta": float(coef[1]), "beta_se": float(se[1]),
        "correlation_with_market": float(np.corrcoef(exc, r["market_excess"])[0, 1]),
        "long_leg_mean_monthly_contribution": float(r["long_contribution"].mean()),
        "short_leg_mean_monthly_contribution": float(r["short_contribution"].mean()),
        "avg_long_names": float(monthly["n_long"].mean()),
        "avg_short_names": float(monthly["n_short"].mean()),
        "min_total_names": int((monthly["n_long"] + monthly["n_short"]).min()),
        "max_total_names": int((monthly["n_long"] + monthly["n_short"]).max()),
        "gross_avg": float(monthly["gross"].mean()), "gross_max": float(monthly["gross"].max()),
        "net_avg": float(monthly["net"].mean()),
        "net_min": float(monthly["net"].min()), "net_max": float(monthly["net"].max()),
        "position_weight_avg": float(w.mean()), "position_weight_max": float(w.max()),
        "top10_share_avg": float(monthly["top10_share"].mean()),
        "turnover_one_way_avg": float(to.mean()),
        "turnover_one_way_min": float(to.min()), "turnover_one_way_max": float(to.max()),
        "ex_ante_beta_max_abs": float(monthly["chk_beta_ex_ante"].abs().max()),
        "months_with_relaxed_limits": int((monthly["relax_level"] > 0).sum()),
        "short_book_avg_market_cap_musd": float(shorts["me"].mean()),
        "short_book_avg_dollar_volume": float(shorts["dolvol_126d"].mean()),
        "short_book_share_small_cap_or_below": float(
            shorts["size_grp"].isin(["small", "micro", "nano"]).mean()),
        "held_names_missing_return": int(r["names_missing_return"].sum()),
    }
