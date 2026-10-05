"""Command line entry point.

    python -m portfolio_agent.run --panel "Provided Documents/chars_final_with_names.parquet" \
        --signals inputs/signals.csv --da inputs/devils_advocate.csv --out output

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
from .backtest import monthly_returns, position_pnl, summary
from .config import Mandate, Policy
from .contracts import read_table
from .data import load_outcomes, load_panel, month_end, rf_by_month
from .parquet_io import read_parquet

DEMO_CHARS = ["ret_12_1", "qmj", "be_me"]


def demo_signals(panel_path) -> pd.DataFrame:
    """Placeholder input, NOT the strategy: equal-weight rank of momentum, quality
    and value. Fixed a priori, nothing is fitted, only month-t characteristics."""
    df = read_parquet(panel_path, columns=["permno", "eom"] + DEMO_CHARS)
    df["eom"] = month_end(df["eom"])
    ranks = df.groupby("eom")[DEMO_CHARS].rank(pct=True)
    df["score"] = ranks.mean(axis=1, skipna=False)
    df["source"] = "demo_baseline"
    return df.dropna(subset=["score"])[["permno", "eom", "score", "source"]]


def _fred(path) -> pd.Series:
    """FRED csv (date column first, value column second) -> Series by month-end."""
    df = pd.read_csv(path)
    return pd.Series(df.iloc[:, 1].astype(float).values, index=month_end(df.iloc[:, 0]))


def _create_hint(path: str, panel: str) -> str:
    """How to create a generated input file, judged from its folder name."""
    name = str(path).replace("\\", "/")
    if "out_composite" in name:
        return f'python -m base_signal.composites --panel "{panel}" --out out_composite'
    if "out_base" in name:
        return f'python -m base_signal.model --panel "{panel}" --out out_base'
    if "out_8k" in name:
        return 'python -m agents.eightk_signal --filings "Provided Documents/8k_20150101_20260831_identified.parquet" --max-calls <N> --out out_8k   (needs OPENAI_API_KEY; run --estimate first)'
    return f'python -m base_signal.composites --panel "{panel}" --out <folder>'


def check_inputs(panel: str, signals, da=None) -> None:
    """Stop with a plain message, not a traceback, when an input file is missing."""
    if not Path(panel).exists():
        raise SystemExit(f"Data file not found: {panel}\n"
                         "The challenge data files are not stored in git. Put chars_final_with_names.parquet in the "
                         '"Provided Documents" folder (or pass its full path with --panel).')
    for path in signals or []:
        if not Path(path).exists():
            raise SystemExit(f"Signals file not found: {path}\n"
                             "Signal files are generated, not stored in git (out_*/ folders are ignored). Create it first:\n"
                             f"  {_create_hint(path, panel)}\nthen re-run this command.")
    if da and not Path(da).exists():
        raise SystemExit(f"Devil's-advocate file not found: {da}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--signals", nargs="+", help="one or more signal files (e.g. base signal + 8-K signal)")
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
    check_inputs(a.panel, None if a.demo_baseline else a.signals, a.da)

    policy = Policy(**{k: v for k, v in vars(a).items()
                       if k in asdict(Policy()) and v is not None})
    if a.demo_baseline:
        signals = demo_signals(a.panel)
    elif a.signals:
        signals = pd.concat([read_table(p) for p in a.signals], ignore_index=True)
    else:
        ap.error("give --signals or --demo-baseline")
    da = read_table(a.da) if a.da else None

    panel = load_panel(a.panel)
    agent = PortfolioConstructionAgent(panel, Mandate(), policy)
    first = pd.Timestamp(a.start) + MonthEnd(0) - MonthEnd(1)   # formation = month before
    last = pd.Timestamp(a.end) + MonthEnd(0) - MonthEnd(1)
    res = agent.run(signals, da, first, last)

    outcomes = load_outcomes(a.panel)
    rets = monthly_returns(res.holdings, res.monthly, outcomes, rf_by_month(panel),
                           policy, _fred(a.tb3ms) if a.tb3ms else None,
                           _fred(a.sp500) if a.sp500 else None)
    stats = summary(rets, res.monthly, res.holdings, panel)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sub = ["Date", "PERMNO", "TICKER", "COMPANY NAME", "WEIGHT"]
    res.holdings[sub].to_csv(out / "holdings.csv", index=False, date_format="%Y-%m-%d")
    position_pnl(res.holdings, outcomes, panel, policy).to_csv(
        out / "position_pnl.csv", index=False, date_format="%Y-%m-%d")
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
