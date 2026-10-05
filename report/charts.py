"""Deck charts (PNG, 16:9-friendly). matplotlib only.

Design rules (dataviz skill): one y-axis per chart, thin 2px lines, validated categorical
slots in fixed order (blue = strategy, orange = cash + 4% benchmark); S&P 500 = neutral gray context, a legend on
every multi-series chart plus direct end labels, text in neutral ink (never series color),
recessive grid, no dual axes. Every title states what, the period, and gross/net of costs.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

try:
    from . import stats as S
except ImportError:                    # run as a script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from report import stats as S

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e6e5e1"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#d03b3b"
CONTEXT = "#898781"        # S&P 500 is shown for context only: neutral gray, not a competing series
BLUE_FILL = "#cde2fb"
SIZE, DPI = (11, 5.8), 200


def _setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2,
    })


def _period(r):
    return f"{r.index.min():%b %Y} to {r.index.max():%b %Y}"


def _title(ax, title, r, basis):
    ax.set_title(title, pad=22)
    ax.text(0, 1.02, f"Out-of-sample {_period(r)} | {basis}", transform=ax.transAxes,
            fontsize=10, color=INK2, va="bottom")


def _save(fig, out, name):
    fig.tight_layout()
    path = out / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                color=INK, fontsize=10, annotation_clip=False)
    ax.plot([x], [y], "o", color=color, ms=6, mec=SURFACE, mew=2, clip_on=False)


def _pct(ax, axis="y", decimals=0):
    f = PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def _start(r):
    return r.index.min() - pd.offsets.MonthEnd(1)


def cumulative(r, out, basis):
    fig, ax = plt.subplots(figsize=SIZE)
    for col, name, c in (("total_return", "Strategy", BLUE), ("benchmark", "Cash + 4% (TB3MS + 4%)", ORANGE),
                         ("sp500", "S&P 500, context only (price return)", CONTEXT)):
        n = S.nav(r[col])
        n.loc[_start(r)] = 1.0
        n = n.sort_index()
        ax.plot(n.index, n.values, color=c, label=name)
        _end_label(ax, n.index[-1], n.iloc[-1], f"{n.iloc[-1] - 1:+.0%}", c)
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.margins(x=0.01)
    ax.set_xlim(right=r.index.max() + pd.Timedelta(days=75))
    _title(ax, "Cumulative return: strategy vs cash + 4% benchmark (S&P 500 for context)", r, basis)
    return _save(fig, out, "01_cumulative_return.png")


def drawdown(r, out, basis):
    fig, ax = plt.subplots(figsize=SIZE)
    d, m = S.drawdown(r["total_return"]), S.drawdown(r["sp500"])
    ax.fill_between(d.index, d.values, 0, color=BLUE_FILL, lw=0)
    ax.plot(d.index, d.values, color=BLUE, label="Strategy")
    ax.plot(m.index, m.values, color=CONTEXT, label="S&P 500, context only (price return)")
    ax.axhline(0, color=MUTED, lw=1)
    i = d.idxmin()
    ax.annotate(f"Max drawdown {d.min():.1%}", (i, d.min()), xytext=(10, -2), textcoords="offset points",
                color=INK, fontsize=10)
    _pct(ax)
    ax.set_ylabel("Drawdown from peak")
    ax.legend(loc="lower left")
    _title(ax, "Underwater chart: strategy vs S&P 500", r, basis)
    return _save(fig, out, "02_drawdown.png")


def rolling_active(r, out, basis):
    ret, ir = S.rolling_active(r["active"])
    fig, (a, b) = plt.subplots(2, 1, figsize=(SIZE[0], SIZE[1] + 1.4), sharex=True)
    a.plot(ret.index, ret.values, color=BLUE)
    a.axhline(0, color=MUTED, lw=1)
    _pct(a)
    a.set_ylabel("Active return, trailing 12m")
    b.plot(ir.index, ir.values, color=BLUE)
    b.axhline(0, color=MUTED, lw=1)
    b.set_ylabel("Information ratio, trailing 12m")
    a.set_title("Rolling 12-month active return and information ratio vs cash + 4%", pad=22)
    a.text(0, 1.02, f"Out-of-sample {_period(r)} | {basis}", transform=a.transAxes, fontsize=10, color=INK2)
    for ax, s in ((a, ret), (b, ir)):
        ax.annotate(f"{s.iloc[-1]:+.1%}" if ax is a else f"{s.iloc[-1]:+.2f}", (s.index[-1], s.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=10, color=INK, annotation_clip=False)
    return _save(fig, out, "03_rolling_active_return_and_ir.png")


def rolling_beta(r, out, basis, full_beta):
    rb = S.rolling_beta(r["ex_tb"], r["mkt_ex"])
    fig, ax = plt.subplots(figsize=SIZE)
    ax.plot(rb.index, rb.values, color=BLUE, label="Rolling 12-month beta")
    ax.axhline(0, color=INK2, lw=1.2)
    ax.axhline(full_beta, color=BLUE, lw=1, ls=(0, (4, 3)), label=f"Full-period beta {full_beta:+.2f}")
    ax.set_ylabel("Beta to S&P 500")
    ax.legend(loc="best")
    _title(ax, "Rolling 12-month beta to the S&P 500 (neutral = 0)", r, basis)
    return _save(fig, out, "04_rolling_beta.png")


def histogram(r, out, basis):
    x = r["total_return"]
    fig, ax = plt.subplots(figsize=SIZE)
    bins = np.linspace(x.min() - 0.002, x.max() + 0.002, 22)
    ax.hist(x, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=2)
    lo, hi, mean = r["benchmark"].min(), r["benchmark"].max(), r["benchmark"].mean()
    ax.axvspan(lo, hi, color=ORANGE, alpha=0.25, lw=0)
    ax.axvline(mean, color=ORANGE, lw=2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)                 # headroom for the hurdle label
    ax.annotate(f"Monthly hurdle (TB3MS+4%)\navg {mean:.2%}, range {lo:.2%} to {hi:.2%}", (mean, ax.get_ylim()[1]),
                xytext=(8, -8), textcoords="offset points", va="top", fontsize=10, color=INK)
    _pct(ax, "x", 1)
    ax.set_xlabel("Monthly strategy return")
    ax.set_ylabel("Number of months")
    ax.grid(axis="x", visible=False)
    _title(ax, "Distribution of monthly returns vs the monthly hurdle", r, basis)
    return _save(fig, out, "05_monthly_return_histogram.png")


def calendar(r, out, basis):
    cy = S.calendar_years(r)
    fig, ax = plt.subplots(figsize=SIZE)
    w, x = 0.26, np.arange(len(cy))
    for k, (col, name, c) in enumerate((("strategy", "Strategy", BLUE), ("benchmark", "Cash + 4%", ORANGE),
                                        ("sp500_price", "S&P 500, context (price)", CONTEXT))):
        bars = ax.bar(x + (k - 1) * (w + 0.02), cy[col], w, color=c, label=name, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, cy[col]):
            ax.annotate(f"{v:.1%}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -11),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, [f"{y}{'*' if p else ''}" for y, p in zip(cy.index, cy["partial_year"])])
    _pct(ax)
    ax.grid(axis="x", visible=False)
    top = max(cy[["strategy", "benchmark", "sp500_price"]].max().max(), 0.05)
    ax.set_ylim(ax.get_ylim()[0], top * 1.22)              # headroom so the legend never meets a label
    ax.legend(loc="upper center", ncols=3)
    ax.text(1, -0.1, "* partial year (months available: " + ", ".join(
        f"{y}: {m}" for y, m, p in zip(cy.index, cy["months"], cy["partial_year"]) if p) + ")",
        transform=ax.transAxes, ha="right", fontsize=9, color=INK2)
    _title(ax, "Calendar-year returns", r, basis)
    return _save(fig, out, "06_calendar_year_returns.png")


def exposures(r, out, basis):
    fig, (a, b) = plt.subplots(2, 1, figsize=(SIZE[0], SIZE[1] + 1.4), sharex=True,
                               gridspec_kw={"height_ratios": [3, 2]})
    a.plot(r.index, r["gross"], color=BLUE, label="Gross exposure")
    a.plot(r.index, r["net"], color=ORANGE, label="Net exposure")
    a.axhline(2.0, color=MUTED, lw=1, ls=(0, (4, 3)))
    a.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    a.axhline(-0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    a.text(r.index[0], 2.03, "Mandate limits: gross 200%, net -50% to +50%", fontsize=9, color=INK2)
    a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a.set_ylim(-0.7, 2.25)
    a.set_ylabel("% of NAV")
    a.legend(loc="center right")
    b.plot(r.index, r["n_long"], color=BLUE, label="Long positions")
    b.plot(r.index, r["n_short"], color=ORANGE, label="Short positions")
    b.set_ylabel("Positions")
    b.set_ylim(0, max(r["n_long"].max(), r["n_short"].max()) * 1.25)
    b.legend(loc="upper right", ncols=2)
    a.set_title("Exposure and position count at each rebalance", pad=22)
    a.text(0, 1.02, f"Out-of-sample {_period(r)} | {basis}", transform=a.transAxes, fontsize=10, color=INK2)
    return _save(fig, out, "07_exposures_and_positions.png")


def turnover(r, out, basis):
    t = r["turnover_one_way"].dropna()
    fig, ax = plt.subplots(figsize=SIZE)
    ax.bar(t.index, t.values, width=22, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axhline(t.mean(), color=ORANGE, lw=2)
    ax.annotate(f"Average {t.mean():.0%} one-way per month", (t.index[0], t.mean()), xytext=(0, 6),
                textcoords="offset points", fontsize=10, color=INK)
    _pct(ax)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("One-way turnover, % of NAV")
    ax.set_ylim(0, max(t.max() * 1.25, 0.4))
    _title(ax, "Monthly portfolio turnover (initial build excluded)", r, basis)
    return _save(fig, out, "08_turnover.png")


def leg_cumulative(r, out, basis):
    fig, ax = plt.subplots(figsize=SIZE)
    for col, name, c in (("long_contribution", "Long book", BLUE), ("short_contribution", "Short book", ORANGE)):
        s = r[col].cumsum()
        s.loc[_start(r)] = 0.0
        s = s.sort_index()
        ax.plot(s.index, s.values, color=c, label=name)
        _end_label(ax, s.index[-1], s.iloc[-1], f"{s.iloc[-1]:+.1%}", c)
    ax.axhline(0, color=MUTED, lw=1)
    _pct(ax)
    ax.set_ylabel("Cumulative P&L contribution, % of NAV")
    ax.set_xlim(right=r.index.max() + pd.Timedelta(days=75))
    ax.legend(loc="upper left")
    _title(ax, "Which side is working: cumulative contribution of long and short books", r, basis)
    return _save(fig, out, "09_long_vs_short_contribution.png")


def contributors(contrib, r, out, basis):
    top = contrib[contrib["rank_type"] == "top_positive"].sort_values("total_pnl")
    bot = contrib[contrib["rank_type"] == "top_negative"].sort_values("total_pnl", ascending=False)
    fig, (a, b) = plt.subplots(1, 2, figsize=(SIZE[0] + 2, SIZE[1] + 0.8), sharex=False)
    for ax, d, c, ttl in ((a, top, BLUE, "10 largest gains"), (b, bot, RED, "10 largest losses")):
        lab = [f"{t} ({str(n).title()[:26].rstrip(' ,.-(')})" + ("*" if v else "")
               for t, n, v in zip(d["ticker"].fillna("n/a"), d["company_name"].fillna("unlabeled"),
                                  d["label_needs_verification"])]
        ax.barh(lab, d["total_pnl_pct_nav"], color=c, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(d["total_pnl_pct_nav"]):
            ax.annotate(f"{v:+.2f}%", (v, i), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=9, color=INK)
        ax.axvline(0, color=INK2, lw=1)
        ax.grid(axis="y", visible=False)
        ax.set_title(ttl, fontsize=12, loc="left")
        ax.set_xlabel("Total P&L contribution, % of NAV")
        ax.margins(x=0.30)
    fig.suptitle("Largest P&L contributors by stock", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.93, f"Out-of-sample {_period(r)} | {basis} | * label not verified in-month: confirm before publishing",
             fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out / "10_top_pnl_contributors.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def all_charts(r, contrib, out, basis, full_beta):
    _setup()
    return [cumulative(r, out, basis), drawdown(r, out, basis), rolling_active(r, out, basis),
            rolling_beta(r, out, basis, full_beta), histogram(r, out, basis), calendar(r, out, basis),
            exposures(r, out, basis), turnover(r, out, basis), leg_cumulative(r, out, basis),
            contributors(contrib, r, out, basis)]


if __name__ == "__main__":
    print("This file only draws charts and prints nothing by itself.\n"
          "Run from the repo root:  python -m report.build --run-dir <run folder> --out report_out")
