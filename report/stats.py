"""Deck statistics from the portfolio agent's output files. Pure pandas/numpy.

Conventions (also written to methodology.md):
  total return       = panel risk-free + portfolio excess return (cash earns the rf the data
                       pipeline used; shorts finance longs). Gross of trading costs unless the
                       backtest was run with --tcost-bps.
  benchmark          = TB3MS/100/12 + 0.04/12 (decimal, monthly)
  active return      = total return - benchmark
  information ratio  = sqrt(12) * mean(active) / stdev(active)
  excess (for Sharpe, alpha, beta) = total return - TB3MS/100/12; market excess likewise
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SMALL = {"small", "micro", "nano"}
HARD_TO_BORROW_DOLVOL = 1e7     # proxy: below $10M average daily dollar volume


def load_run(run_dir):
    from pathlib import Path
    d = Path(run_dir)
    ret = pd.read_csv(d / "portfolio_returns.csv", parse_dates=["month"]).set_index("month")
    hold = pd.read_csv(d / "holdings.csv", parse_dates=["Date"])
    pnl = pd.read_csv(d / "position_pnl.csv", parse_dates=["Date", "eom"])
    verify = d / "labels_to_verify.csv"
    flagged = set()
    if verify.exists():
        v = pd.read_csv(verify, parse_dates=["Date"])
        flagged = set(zip(v["Date"], v["PERMNO"]))
    return ret, hold, pnl, flagged


def attach_market(ret: pd.DataFrame, tb3ms: pd.Series, sp500: pd.Series) -> pd.DataFrame:
    """Add benchmark, S&P 500 and TB3MS-based excess columns. Fails loudly on gaps."""
    r = ret.copy()
    for name, s in (("TB3MS", tb3ms), ("S&P 500", sp500)):
        missing = r.index.difference(s.index)
        if len(missing):
            raise ValueError(f"{name} has no value for {len(missing)} holding months, "
                             f"first {missing[0].date()}")
    r["rf_tb"] = tb3ms.reindex(r.index) / 100 / 12
    r["benchmark"] = r["rf_tb"] + 0.04 / 12
    r["sp500"] = sp500.reindex(r.index)
    r["active"] = r["total_return"] - r["benchmark"]
    r["ex_tb"] = r["total_return"] - r["rf_tb"]
    r["mkt_ex"] = r["sp500"] - r["rf_tb"]
    return r


# ----------------------------------------------------------------------- helpers
def nav(x: pd.Series) -> pd.Series:
    return (1 + x).cumprod()


def drawdown(x: pd.Series) -> pd.Series:
    n = nav(x)
    return n / n.cummax() - 1


def ols(y: np.ndarray, x: np.ndarray, lags: int = 3):
    """y = a + b x. Returns (a, b, se_a, se_b, t_a, nw_se_a, nw_se_b, nw_t_a)."""
    X = np.column_stack([np.ones(len(x)), x])
    XtXi = np.linalg.inv(X.T @ X)
    coef = XtXi @ X.T @ y
    e = y - X @ coef
    n = len(y)
    se = np.sqrt(np.diag(XtXi) * (e @ e) / (n - 2))
    u = X * e[:, None]                                   # Newey-West (Bartlett kernel)
    S = u.T @ u
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        G = u[k:].T @ u[:-k]
        S += w * (G + G.T)
    nw = np.sqrt(np.diag(XtXi @ S @ XtXi))
    return coef[0], coef[1], se[0], se[1], coef[0] / se[0], nw[0], nw[1], coef[0] / nw[0]


def rolling_beta(ex: pd.Series, mkt: pd.Series, window: int = 12) -> pd.Series:
    out = {}
    for i in range(window, len(ex) + 1):
        y, x = ex.iloc[i - window:i].values, mkt.iloc[i - window:i].values
        xc = x - x.mean()
        out[ex.index[i - 1]] = (xc @ (y - y.mean())) / (xc @ xc)
    return pd.Series(out)


def rolling_active(active: pd.Series, window: int = 12):
    ret = active.rolling(window).sum()                   # active return over trailing 12m
    ir = np.sqrt(12) * active.rolling(window).mean() / active.rolling(window).std()
    return ret.dropna(), ir.dropna()


def calendar_years(r: pd.DataFrame) -> pd.DataFrame:
    g = r.groupby(r.index.year)
    out = pd.DataFrame({
        "strategy": g["total_return"].apply(lambda s: (1 + s).prod() - 1),
        "benchmark": g["benchmark"].apply(lambda s: (1 + s).prod() - 1),
        "sp500_price": g["sp500"].apply(lambda s: (1 + s).prod() - 1),
        "months": g.size()})
    out["partial_year"] = out["months"] < 12
    return out


def leg_returns(r: pd.DataFrame, hold: pd.DataFrame) -> pd.DataFrame:
    """Each leg's P&L as a return on its own gross exposure."""
    h = hold.assign(month=hold["Date"] + pd.offsets.MonthEnd(0))
    long_x = h[h["WEIGHT"] > 0].groupby("month")["WEIGHT"].sum()
    short_x = -h[h["WEIGHT"] < 0].groupby("month")["WEIGHT"].sum()
    out = pd.DataFrame({"long_exposure": long_x, "short_exposure": short_x})
    out["long_return"] = r["long_contribution"] / out["long_exposure"]
    out["short_return"] = r["short_contribution"] / out["short_exposure"]   # + = shorts fell
    return out


def top_holdings(pnl: pd.DataFrame, flagged: set, n: int = 10) -> pd.DataFrame:
    """Average weight over all out-of-sample months (a month not held counts as 0)."""
    months = pnl["Date"].nunique()
    label = (pnl.sort_values("Date").groupby("PERMNO")[["TICKER", "COMPANY NAME"]].last())
    rows = []
    for side, sign in (("long", 1), ("short", -1)):
        p = pnl[pnl["WEIGHT"] * sign > 0]
        g = p.groupby("PERMNO").agg(avg_weight=("WEIGHT", lambda w: w.sum() / months),
                                    months_held=("WEIGHT", "size"))
        g = g.sort_values("avg_weight", ascending=(side == "short")).head(n)
        g = g.join(label).reset_index()
        g.insert(0, "side", side)
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["avg_weight_pct"] = out["avg_weight"] * 100
    keys = pnl.groupby("PERMNO")["Date"].apply(set)
    out["label_needs_verification"] = [
        any((d, p) in flagged for d in keys.get(p, ())) for p in out["PERMNO"]]
    return out.rename(columns={"TICKER": "ticker", "COMPANY NAME": "company_name"})


def top_contributors(pnl: pd.DataFrame, flagged: set, n: int = 10) -> pd.DataFrame:
    label = pnl.sort_values("Date").groupby("PERMNO")[["TICKER", "COMPANY NAME"]].last()
    g = pnl.groupby("PERMNO").agg(total_pnl=("pnl", "sum"), months_held=("pnl", "size"),
                                  avg_weight=("WEIGHT", "mean")).join(label).reset_index()
    keys = pnl.groupby("PERMNO")["Date"].apply(set)
    g["label_needs_verification"] = [any((d, p) in flagged for d in keys[p]) for p in g["PERMNO"]]
    g["total_pnl_pct_nav"] = g["total_pnl"] * 100
    g = g.rename(columns={"TICKER": "ticker", "COMPANY NAME": "company_name"})
    top = g.nlargest(n, "total_pnl").assign(rank_type="top_positive")
    bot = g.nsmallest(n, "total_pnl").assign(rank_type="top_negative")
    return pd.concat([top, bot], ignore_index=True)


def short_book(pnl: pd.DataFrame) -> dict:
    s = pnl[pnl["WEIGHT"] < 0]
    w = s["WEIGHT"].abs()
    small = s["size_grp"].isin(SMALL)
    htb = small | (s["dolvol_126d"] < HARD_TO_BORROW_DOLVOL)
    return {"avg_market_cap_musd": float(s["me"].mean()),
            "median_market_cap_musd": float(s["me"].median()),
            "avg_dollar_volume_usd": float(s["dolvol_126d"].mean()),
            "share_small_cap_or_below_by_count": float(small.mean()),
            "share_small_cap_or_below_by_weight": float((w * small).sum() / w.sum()),
            "share_plausibly_hard_to_borrow_by_weight": float((w * htb).sum() / w.sum()),
            "hard_to_borrow_proxy": f"small/micro/nano size group or 126d dollar volume < ${HARD_TO_BORROW_DOLVOL:,.0f}"}


# ------------------------------------------------------------------------ summary
def _series_stats(x: pd.Series, rf: pd.Series) -> dict:
    ex = x - rf
    n = (1 + x).prod()
    return {
        "avg_monthly_return_arithmetic": x.mean(),
        "avg_monthly_return_geometric": n ** (1 / len(x)) - 1,
        "annualized_return_arithmetic": x.mean() * 12,
        "annualized_return_geometric_cagr": n ** (12 / len(x)) - 1,
        "cumulative_return": n - 1,
        "annualized_volatility": x.std() * np.sqrt(12),
        "sharpe_ratio_annualized": (ex.mean() / ex.std() * np.sqrt(12)) if ex.std() > 1e-12 else np.nan,
        "max_drawdown": drawdown(x).min(),
        "best_month": x.max(), "best_month_date": str(x.idxmax().date()),
        "worst_month": x.min(), "worst_month_date": str(x.idxmin().date()),
    }


def summarize(r: pd.DataFrame, hold: pd.DataFrame, pnl: pd.DataFrame) -> dict:
    strat = _series_stats(r["total_return"], r["rf_tb"])
    bench = _series_stats(r["benchmark"], r["rf_tb"])
    sp = _series_stats(r["sp500"], r["rf_tb"])
    a, b, se_a, se_b, t_a, nw_se_a, nw_se_b, nw_t_a = ols(r["ex_tb"].values, r["mkt_ex"].values)
    act = r["active"]
    legs = leg_returns(r, hold)
    h = hold.assign(abs_w=hold["WEIGHT"].abs())
    top10 = h.groupby("Date")["abs_w"].apply(lambda s: s.nlargest(10).sum() / s.sum())
    to = r["turnover_one_way"].dropna()
    risk = {
        "information_ratio_vs_cash_plus_4pct": float(np.sqrt(12) * act.mean() / act.std()),
        "mean_active_return_annualized": float(act.mean() * 12),
        "hit_rate_positive_active_return": float((act > 0).mean()),
        "alpha_annualized": float(a * 12), "alpha_t_stat_ols": float(t_a),
        "alpha_t_stat_newey_west_3": float(nw_t_a),
        "beta": float(b), "beta_se_ols": float(se_b), "beta_se_newey_west_3": float(nw_se_b),
        "correlation_with_sp500": float(np.corrcoef(r["ex_tb"], r["mkt_ex"])[0, 1]),
    }
    expo = {
        "avg_long_positions": float(r["n_long"].mean()), "avg_short_positions": float(r["n_short"].mean()),
        "gross_avg": float(r["gross"].mean()), "gross_min": float(r["gross"].min()), "gross_max": float(r["gross"].max()),
        "net_avg": float(r["net"].mean()), "net_min": float(r["net"].min()), "net_max": float(r["net"].max()),
        "position_weight_avg_abs": float(h["abs_w"].mean()), "position_weight_max_abs": float(h["abs_w"].max()),
        "top10_share_of_gross_avg": float(top10.mean()),
        "turnover_one_way_avg": float(to.mean()), "turnover_one_way_min": float(to.min()),
        "turnover_one_way_max": float(to.max()),
        "turnover_note": "first month (initial build) excluded",
    }
    legs_s = {
        "long_leg_avg_monthly_return_on_exposure": float(legs["long_return"].mean()),
        "short_leg_avg_monthly_pnl_on_exposure": float(legs["short_return"].mean()),
        "long_leg_annualized_arithmetic": float(legs["long_return"].mean() * 12),
        "short_leg_annualized_arithmetic": float(legs["short_return"].mean() * 12),
        "long_leg_avg_contribution_to_nav": float(r["long_contribution"].mean()),
        "short_leg_avg_contribution_to_nav": float(r["short_contribution"].mean()),
    }
    return {"period": f"{r.index.min().date()} to {r.index.max().date()}", "months": int(len(r)),
            "returns_are": "gross of trading costs" if (r["trading_cost"] == 0).all() else "net of trading costs",
            "strategy": strat, "benchmark": bench, "sp500_price_return": sp,
            "risk_adjusted": risk, "exposure": expo, "legs": legs_s, "short_book": short_book(pnl),
            "held_names_missing_return": int(pnl["return_missing"].sum())}
