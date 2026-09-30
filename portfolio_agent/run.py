"""Command line entry point.

    python -m portfolio_agent.run --panel "Provided Documents/chars_final_with_names.parquet" \
        --signals inputs/signals.parquet --da inputs/devils_advocate.csv --out output

Use --demo-baseline to run end to end before the signal agents are ready.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import MonthEnd

from .agent import PortfolioConstructionAgent
from .backtest import monthly_returns, summary
from .config import Mandate, Policy
from .contracts import read_table
from .data import load_outcomes, load_panel, month_end, rf_by_month

DEMO_CHARS = ["ret_12_1", "qmj", "be_me"]


def demo_signals(panel_path) -> pd.DataFrame:
    """Placeholder input, NOT the strategy: equal-weight rank of momentum, quality
    and value. Fixed a priori, nothing is fitted, only month-t characteristics."""
    df = pd.read_parquet(panel_path, columns=["permno", "eom"] + DEMO_CHARS)
    df["eom"] = month_end(df["eom"])
    ranks = df.groupby("eom")[DEMO_CHARS].rank(pct=True)
    df["score"] = ranks.mean(axis=1, skipna=False)
    df["source"] = "demo_baseline"
    return df.dropna(subset=["score"])[["permno", "eom", "score", "source"]]


def _fred(path) -> pd.Series:
    """FRED csv (date column first, value column second) -> Series by month-end."""
    df = pd.read_csv(path)
    return pd.Series(df.iloc[:, 1].astype(float).values, index=month_end(df.iloc[:, 0]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--signals")
    ap.add_argument("--da", help="devil's-advocate verdicts (optional)")
    ap.add_argument("--demo-baseline", action="store_true")
    ap.add_argument("--out", default="output")
    ap.add_argument("--start", default="2021-01", help="first holding month")
    ap.add_argument("--end", default="2026-08", help="last holding month")
    ap.add_argument("--tb3ms", help="FRED TB3MS csv for the benchmark")
    ap.add_argument("--sp500", help="csv: month, S&P 500 monthly total return (decimal)")
    for name, value in asdict(Policy()).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            ap.add_argument("--" + name.replace("_", "-"), type=float, default=None)
    a = ap.parse_args(argv)

    policy = Policy(**{k: v for k, v in vars(a).items()
                       if k in asdict(Policy()) and v is not None})
    if a.demo_baseline:
        signals = demo_signals(a.panel)
    elif a.signals:
        signals = read_table(a.signals)
    else:
        ap.error("give --signals or --demo-baseline")
    da = read_table(a.da) if a.da else None

    panel = load_panel(a.panel)
    agent = PortfolioConstructionAgent(panel, Mandate(), policy)
    first = pd.Timestamp(a.start) + MonthEnd(0) - MonthEnd(1)   # formation = month before
    last = pd.Timestamp(a.end) + MonthEnd(0) - MonthEnd(1)
    res = agent.run(signals, da, first, last)

    rets = monthly_returns(res.holdings, res.monthly, load_outcomes(a.panel), rf_by_month(panel),
                           policy, _fred(a.tb3ms) if a.tb3ms else None,
                           _fred(a.sp500) if a.sp500 else None)
    stats = summary(rets, res.monthly, res.holdings, panel)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sub = ["Date", "PERMNO", "TICKER", "COMPANY NAME", "WEIGHT"]
    res.holdings[sub].to_csv(out / "holdings.csv", index=False, date_format="%Y-%m-%d")
    rets.to_csv(out / "portfolio_returns.csv", index=False, date_format="%Y-%m-%d")
    res.trades.to_csv(out / "trade_log.csv", index=False, date_format="%Y-%m-%d")
    res.monthly.to_csv(out / "guardrail_report.csv", index=False, date_format="%Y-%m-%d")
    res.checks.to_csv(out / "guardrail_checks.csv", index=False, date_format="%Y-%m-%d")
    res.holdings[res.holdings["label_status"] != "verified_in_month"].to_csv(
        out / "labels_to_verify.csv", index=False, date_format="%Y-%m-%d")
    (out / "summary.json").write_text(json.dumps(
        {"mandate": asdict(Mandate()), "policy": asdict(policy), "stats": stats}, indent=2))
    print(json.dumps(stats, indent=2))
    return res, rets, stats


if __name__ == "__main__":
    main()
