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

import numpy as np
import pandas as pd

try:                                   # normal use: python -m report.build ...
    from . import charts, market, stats as S
except ImportError:                    # run as a script: python report/build.py ...
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from report import charts, market, stats as S

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


def _p(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}%}"


def _n(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def sections(st: dict, cy: pd.DataFrame, top_h: pd.DataFrame, contrib: pd.DataFrame) -> list[tuple]:
    """Every deck table as (title, header, rows). Used for the terminal and for REPORT.md."""
    s, b, m = st["strategy"], st["benchmark"], st["sp500_price_return"]
    ra, ex, lg, sb = st["risk_adjusted"], st["exposure"], st["legs"], st["short_book"]

    def row(name, key, f=_p):
        return (name, f(s[key]), f(b[key]), f(m[key]))

    returns = [row("Average monthly return (arithmetic)", "avg_monthly_return_arithmetic"),
               row("Average monthly return (geometric)", "avg_monthly_return_geometric"),
               row("Annualized return (arithmetic)", "annualized_return_arithmetic"),
               row("Annualized return (geometric, CAGR)", "annualized_return_geometric_cagr"),
               row("Cumulative return", "cumulative_return", lambda x: _p(x, 1)),
               row("Annualized volatility", "annualized_volatility"),
               ("Sharpe ratio (vs T-bill)", _n(s["sharpe_ratio_annualized"]), "n/a", _n(m["sharpe_ratio_annualized"])),
               row("Maximum drawdown", "max_drawdown", lambda x: _p(x, 1)),
               ("Best month", f"{_p(s['best_month'])} {s['best_month_date']}", "", f"{_p(m['best_month'])} {m['best_month_date']}"),
               ("Worst month", f"{_p(s['worst_month'])} {s['worst_month_date']}", "", f"{_p(m['worst_month'])} {m['worst_month_date']}")]
    risk = [("Information ratio vs cash + 4%  (HEADLINE)", _n(ra["information_ratio_vs_cash_plus_4pct"])),
            ("Mean active return, annualized", _p(ra["mean_active_return_annualized"])),
            ("Hit rate (months with positive active return)", _p(ra["hit_rate_positive_active_return"], 0)),
            ("Alpha vs S&P 500, annualized", _p(ra["alpha_annualized"])),
            ("Alpha t-stat (OLS / Newey-West)", f"{_n(ra['alpha_t_stat_ols'])} / {_n(ra['alpha_t_stat_newey_west_3'])}"),
            ("Beta vs S&P 500", _n(ra["beta"])),
            ("Beta standard error (OLS / Newey-West)", f"{_n(ra['beta_se_ols'])} / {_n(ra['beta_se_newey_west_3'])}"),
            ("Correlation with S&P 500", _n(ra["correlation_with_sp500"]))]
    expo = [("Average holdings (long / short)", f"{ex['avg_long_positions']:.0f} / {ex['avg_short_positions']:.0f}"),
            ("Gross exposure (avg, min, max)", f"{ex['gross_avg']:.0%}, {ex['gross_min']:.0%}, {ex['gross_max']:.0%}"),
            ("Net exposure (avg, min, max)", f"{ex['net_avg']:+.1%}, {ex['net_min']:+.1%}, {ex['net_max']:+.1%}"),
            ("Position weight (avg, max)", f"{ex['position_weight_avg_abs']:.2%}, {ex['position_weight_max_abs']:.2%}"),
            ("Top-10 names, share of gross", _p(ex["top10_share_of_gross_avg"], 1)),
            ("One-way monthly turnover (avg, min, max)", f"{ex['turnover_one_way_avg']:.0%}, {ex['turnover_one_way_min']:.0%}, {ex['turnover_one_way_max']:.0%}"),
            ("Long book: avg monthly return on its exposure", _p(lg["long_leg_avg_monthly_return_on_exposure"])),
            ("Short book: avg monthly P&L on its exposure", _p(lg["short_leg_avg_monthly_pnl_on_exposure"])),
            ("Short book: avg market cap / daily $ volume", f"${sb['avg_market_cap_musd']:,.0f}M / ${sb['avg_dollar_volume_usd'] / 1e6:,.0f}M"),
            ("Short book: small-cap or below / hard-to-borrow proxy",
             f"{_p(sb['share_small_cap_or_below_by_weight'], 0)} / {_p(sb['share_plausibly_hard_to_borrow_by_weight'], 0)}")]
    years = [(f"{int(y)}{'*' if r.partial_year else ''}", _p(r.strategy, 1), _p(r.benchmark, 1), _p(r.sp500_price, 1))
             for y, r in cy.iterrows()]

    def label(r):
        t = r["ticker"] if isinstance(r["ticker"], str) else "n/a"
        n = r["company_name"] if isinstance(r["company_name"], str) else "unlabeled"
        return f"{t} {str(n)[:30]}" + (" *" if r["label_needs_verification"] else "")

    hold = [(r["side"], label(r), f"{r['avg_weight_pct']:+.2f}%", int(r["months_held"])) for _, r in top_h.iterrows()]
    pos, neg = contrib[contrib["rank_type"] == "top_positive"], contrib[contrib["rank_type"] == "top_negative"]
    pnl = [(i + 1, label(a), f"{a['total_pnl_pct_nav']:+.2f}%", label(c), f"{c['total_pnl_pct_nav']:+.2f}%")
           for i, ((_, a), (_, c)) in enumerate(zip(pos.iterrows(), neg.iterrows()))]
    return [("RETURN STATISTICS", ["", "Strategy", "Cash + 4%", "S&P 500 (context)"], returns),
            ("RISK-ADJUSTED PERFORMANCE", ["", ""], risk),
            ("EXPOSURE, CONCENTRATION, IMPLEMENTATION", ["", ""], expo),
            ("CALENDAR-YEAR RETURNS (* partial year)", ["Year", "Strategy", "Cash + 4%", "S&P 500"], years),
            ("TOP 10 LONG / SHORT HOLDINGS (average weight over the period; months not held count as 0)",
             ["Side", "Ticker Company", "Avg weight", "Months"], hold),
            ("10 LARGEST GAINS / LOSSES (total P&L, % of NAV; * = label not verified in-month)",
             ["#", "Gain", "P&L", "Loss", "P&L"], pnl)]


def print_results(st: dict, secs: list, title: str = "RESULTS") -> None:
    """Print everything to the terminal as aligned plain-text tables."""
    bar = "=" * 100
    print(f"\n{bar}\n{title}\n{bar}")
    print(f"Period {st['period']} ({st['months']} holding months, out of sample). Returns are {st['returns_are']}.")
    print("Benchmark = T-bill (FRED TB3MS) + 4%. The S&P 500 (price index) is shown for context only.\n")
    print(headline(st))
    for name, header, rows in secs:
        print(f"\n{name}\n" + "-" * min(len(name), 100))
        has_header = any(header)
        table = ([header] if has_header else []) + [[str(c) for c in r] for r in rows]
        widths = [max(len(str(r[i])) for r in table) for i in range(len(table[0]))]
        for k, r in enumerate(table):
            print("  " + "  ".join(str(c).ljust(widths[i]) if i == 0 else str(c).rjust(widths[i]) for i, c in enumerate(r)))
            if k == 0 and has_header:
                print("  " + "  ".join("-" * w for w in widths))


def write_markdown(out: Path, st: dict, secs: list, figs, title: str, notes: str = "") -> Path:
    """REPORT.md: every table and chart on one page that GitHub renders."""
    parts = [f"# {title}\n", notes,
             f"**Period:** {st['period']} ({st['months']} holding months, out of sample). **Returns are {st['returns_are']}.** "
             "Benchmark: T-bill (FRED TB3MS) + 4%; the S&P 500 (price index) is context only. "
             "Definitions: [methodology.md](methodology.md). Raw tables: [tables/](tables/).\n", f"> {headline(st)}\n"]
    for name, header, rows in secs:
        head = header if any(header) else ["Metric", "Value"]
        parts.append(f"## {name.capitalize()}\n")
        parts.append("| " + " | ".join(head) + " |\n|" + "|".join("---" for _ in head) + "|\n"
                     + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n")
    parts.append("## Charts\n\n" + "\n\n".join(f"![{f.stem}](charts/{f.name})" for f in figs) + "\n")
    path = out / "REPORT.md"
    path.write_text("\n".join(parts))
    return path


def build_report(run_dir, out_dir, tb3ms: pd.Series, sp500: pd.Series, srcs=("FRED TB3MS", "FRED SP500"),
                 title: str = "Strategy report", notes: str = ""):
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
    secs = sections(st, cy, top_h, contrib)
    write_markdown(out, st, secs, figs, title, notes)
    st["_sections"] = secs
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
    ap.add_argument("--title", default="Strategy report")
    ap.add_argument("--notes", default="", help="text shown at the top of REPORT.md")
    ap.add_argument("--quiet", action="store_true", help="do not print the tables to the terminal")
    a = ap.parse_args(argv)
    a.run_dir = a.run_dir or str(latest_run_dir())
    out = Path(a.out)
    data = out / "market_data"
    tb_path = Path(a.tb3ms) if a.tb3ms else market.fetch_fred("TB3MS", data / "TB3MS.csv", refresh=not a.offline)
    sp_path = Path(a.sp500) if a.sp500 else market.fetch_fred("SP500", data / "SP500.csv", refresh=not a.offline)
    st, figs = build_report(a.run_dir, out, market.tb3ms_monthly(tb_path), market.sp500_monthly_return(sp_path),
                            (f"{tb_path.name}", f"{sp_path.name}, price index"), a.title,
                            (a.notes + "\n") if a.notes else "")
    secs = st.pop("_sections")
    if not a.quiet:
        print_results(st, secs, a.title.upper())
    print(f"\n{len(figs)} charts -> {out / 'charts'}\ntables -> {out / 'tables'}\n"
          f"readable page -> {out / 'REPORT.md'}\n{out / 'methodology.md'}")


if __name__ == "__main__":
    main()
