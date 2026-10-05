"""Build the deck pack: tables, charts and a methodology note.

    python -m report.build --run-dir output_base --out report_out

Reads the CSVs the portfolio agent wrote (portfolio_returns.csv, holdings.csv,
position_pnl.csv, labels_to_verify.csv). Needs pandas, numpy, matplotlib; no parquet.
Benchmark and S&P 500 come from FRED (no key) unless --tb3ms / --sp500 csvs are given.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import charts, market, stats as S

METHOD = """# Methodology and assumptions

Period: {period} ({months} monthly holding months, out of sample). Returns are **{basis}**.

- **Strategy total return** = panel risk-free + portfolio excess return. Weights are signed fractions of NAV;
  NAV sits in cash, shorts finance longs, and cash and the short rebate earn the risk-free rate.
- **Benchmark** = TB3MS/100/12 + 0.04/12 per month, from FRED series TB3MS ({tb_src}).
- **Active return** = strategy total return - benchmark. **Information ratio** = sqrt(12) x mean(active) / stdev(active).
- **Sharpe, alpha and beta** use returns in excess of TB3MS/100/12. Alpha and beta are the OLS intercept and slope of
  strategy excess return on S&P 500 excess return over the same holding months. Standard errors are shown both plain
  OLS and Newey-West (3 lags). Alpha is annualised x12.
- **S&P 500** = FRED series SP500, month-end to month-end **price** return ({sp_src}). It excludes dividends
  (about 1.3% a year), so total-return S&P figures would be slightly higher.
- **Turnover** = one-way: half the sum of absolute weight changes, as a fraction of NAV, after drifting last month's
  weights with realised returns. The initial build is excluded from the average.
- **Leg returns** = each side's P&L divided by that side's gross exposure. Positive short-leg return means shorts fell.
- **Held names with no next-month return** ({missing} position-months) are assumed to earn zero excess return.
- **Hard-to-borrow proxy**: no borrow data was supplied. A short is flagged if it is small/micro/nano cap or its
  126-day dollar volume is below $10M. Borrow costs are not modelled.
- Daily measurement was not done; all figures are monthly.
- Labels marked `label_needs_verification` are not verified in-month in the supplied panel; verify against an
  authoritative source before they appear in the deck.
"""


def pct(x):
    return f"{x:.2%}"


def build_report(run_dir, out_dir, tb3ms: pd.Series, sp500: pd.Series, srcs=("FRED TB3MS", "FRED SP500")):
    out = Path(out_dir)
    (out / "charts").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    ret, hold, pnl, flagged = S.load_run(run_dir)
    r = S.attach_market(ret, tb3ms, sp500)
    st = S.summarize(r, hold, pnl)
    basis = st["returns_are"]

    top_h, contrib = S.top_holdings(pnl, flagged), S.top_contributors(pnl, flagged)
    cy = S.calendar_years(r)
    legs = S.leg_returns(r, hold)

    comp = pd.DataFrame({"strategy": st["strategy"], "cash_plus_4pct_benchmark": st["benchmark"],
                         "sp500_price_return": st["sp500_price_return"]})
    comp.to_csv(out / "tables" / "return_statistics.csv", index_label="metric")
    pd.DataFrame({k: v for k, v in {**st["risk_adjusted"], **st["exposure"], **st["legs"],
                                    **st["short_book"]}.items()}.items(),
                 columns=["metric", "value"]).to_csv(out / "tables" / "risk_exposure_statistics.csv", index=False)
    cy.to_csv(out / "tables" / "calendar_year_returns.csv", index_label="year")
    top_h.to_csv(out / "tables" / "top10_holdings.csv", index=False)
    contrib.to_csv(out / "tables" / "top10_pnl_contributors.csv", index=False)
    r.join(legs).to_csv(out / "tables" / "monthly_series.csv", index_label="month")
    (out / "tables" / "summary.json").write_text(json.dumps(st, indent=2, default=float))

    figs = charts.all_charts(r, contrib, out / "charts", basis, st["risk_adjusted"]["beta"])
    (out / "methodology.md").write_text(METHOD.format(
        period=st["period"], months=st["months"], basis=basis, tb_src=srcs[0], sp_src=srcs[1],
        missing=st["held_names_missing_return"]))
    return st, figs


def headline(st) -> str:
    s, b, ra, ex = st["strategy"], st["benchmark"], st["risk_adjusted"], st["exposure"]
    return (f"{st['period']} ({st['returns_are']}): strategy CAGR {pct(s['annualized_return_geometric_cagr'])} vs "
            f"hurdle {pct(b['annualized_return_geometric_cagr'])}; information ratio "
            f"{ra['information_ratio_vs_cash_plus_4pct']:+.2f}; Sharpe {s['sharpe_ratio_annualized']:+.2f}; "
            f"alpha {pct(ra['alpha_annualized'])} (t={ra['alpha_t_stat_newey_west_3']:+.1f} NW); "
            f"beta {ra['beta']:+.2f} (SE {ra['beta_se_newey_west_3']:.2f} NW); max drawdown {pct(s['max_drawdown'])}; "
            f"gross {ex['gross_avg']:.0%} (max {ex['gross_max']:.0%}), net {ex['net_avg']:+.1%} "
            f"[{ex['net_min']:+.1%}, {ex['net_max']:+.1%}]; turnover {ex['turnover_one_way_avg']:.0%}/month.")


def latest_run_dir(root=".") -> Path:
    """The newest folder here that holds a portfolio run (output*/ with portfolio_returns.csv)."""
    found = sorted((p for p in Path(root).glob("output*") if (p / "portfolio_returns.csv").exists()),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not found:
        raise SystemExit(
            "No run folder found. A run folder is what portfolio_agent.run writes (it holds portfolio_returns.csv).\n"
            "Create one, then re-run this:\n"
            '  python -m portfolio_agent.run --panel "Provided Documents/chars_final_with_names.parquet" '
            "--signals out_composite/signals.parquet --out output_run\n"
            "or point at an existing one:  python report/build.py --run-dir <folder>")
    if len(found) > 1:
        print(f"Using the newest run folder: {found[0]}  (others: {', '.join(map(str, found[1:]))}; pick one with --run-dir)")
    return found[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-dir", help="folder written by portfolio_agent.run (default: the newest output*/ folder here)")
    ap.add_argument("--out", default="report_out")
    ap.add_argument("--tb3ms", help="FRED TB3MS csv (default: download)")
    ap.add_argument("--sp500", help="FRED SP500 csv (default: download)")
    ap.add_argument("--offline", action="store_true", help="use cached FRED files only")
    a = ap.parse_args(argv)
    a.run_dir = a.run_dir or str(latest_run_dir())
    out = Path(a.out)
    data = out / "market_data"
    tb_path = Path(a.tb3ms) if a.tb3ms else market.fetch_fred("TB3MS", data / "TB3MS.csv", refresh=not a.offline)
    sp_path = Path(a.sp500) if a.sp500 else market.fetch_fred("SP500", data / "SP500.csv", refresh=not a.offline)
    st, figs = build_report(a.run_dir, out, market.tb3ms_monthly(tb_path), market.sp500_monthly_return(sp_path),
                            (f"{tb_path.name}", f"{sp_path.name}, price index"))
    print(headline(st))
    print(f"\n{len(figs)} charts -> {out / 'charts'}\ntables -> {out / 'tables'}\n{out / 'methodology.md'}")


if __name__ == "__main__":
    main()
