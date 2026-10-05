"""Report statistics checked against hand-computable cases, plus a full build smoke test."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from report import build, stats as S

MONTHS = pd.date_range("2021-01-31", periods=20, freq="ME")


def returns(seed=0):
    rng = np.random.default_rng(seed)
    n = len(MONTHS)
    r = pd.DataFrame({
        "long_contribution": rng.normal(0.004, 0.01, n), "short_contribution": rng.normal(-0.002, 0.01, n),
        "names_missing_return": 0, "trading_cost": 0.0, "rf": 0.001, "gross": 1.9, "net": 0.0,
        "n_long": 130, "n_short": 130, "turnover_one_way": [np.nan] + [0.3] * (n - 1)}, index=MONTHS)
    r["excess_return"] = r["long_contribution"] + r["short_contribution"]
    r["total_return"] = r["rf"] + r["excess_return"]
    r.index.name = "month"
    return r


def market(n_extra=0):
    idx = MONTHS
    tb = pd.Series(3.6, index=idx)                                  # annualised percent
    sp = pd.Series(np.random.default_rng(1).normal(0.01, 0.04, len(idx)), index=idx)
    return tb, sp


def holdings_and_pnl():
    rows = []
    for k, t in enumerate(MONTHS):
        d = t.replace(day=1)                                          # holdings are dated the 1st of the holding month
        for p, w in ((1, 0.0075), (2, 0.0075), (3, -0.0075), (4, -0.0075)):
            if p == 2 and k >= 10:                                   # permno 2 held only 10 of 20 months
                continue
            rows.append({"Date": d, "PERMNO": p, "TICKER": f"T{p}", "COMPANY NAME": f"Co {p}",
                         "WEIGHT": w, "eom": t - pd.offsets.MonthEnd(1), "ret_exc_lead1m": 0.01 * p,
                         "me": 1000.0 * p, "prc": 20.0, "dolvol_126d": 5e6 * p,
                         "size_grp": "small" if p == 3 else "large", "return_missing": False,
                         "pnl": w * 0.01 * p})
    pnl = pd.DataFrame(rows)
    return pnl[["Date", "PERMNO", "TICKER", "COMPANY NAME", "WEIGHT"]], pnl


class StatsTest(unittest.TestCase):
    def test_calendar_years_compound_and_flag_partial(self):
        r = S.attach_market(returns(), *market())
        cy = S.calendar_years(r)
        self.assertAlmostEqual(cy.loc[2021, "strategy"], (1 + r.loc["2021", "total_return"]).prod() - 1)
        self.assertEqual(cy["months"].to_dict(), {2021: 12, 2022: 8})
        self.assertEqual(cy["partial_year"].to_dict(), {2021: False, 2022: True})

    def test_benchmark_and_active_follow_the_challenge_formula(self):
        r = S.attach_market(returns(), *market())
        self.assertAlmostEqual(r["benchmark"].iloc[0], 3.6 / 100 / 12 + 0.04 / 12)
        self.assertTrue(np.allclose(r["active"], r["total_return"] - r["benchmark"]))

    def test_missing_market_month_fails_loudly(self):
        tb, sp = market()
        with self.assertRaises(ValueError):
            S.attach_market(returns(), tb.iloc[:-2], sp)

    def test_rolling_beta_and_ols_recover_known_slope(self):
        x = pd.Series(np.random.default_rng(2).normal(0, 0.04, 30), index=pd.date_range("2021-01-31", periods=30, freq="ME"))
        y = 0.5 * x + 0.002
        self.assertTrue(np.allclose(S.rolling_beta(y, x), 0.5))
        a, b, *_ = S.ols(y.values, x.values)
        self.assertAlmostEqual(b, 0.5)
        self.assertAlmostEqual(a, 0.002)

    def test_leg_returns_divide_by_own_exposure(self):
        r = returns()
        hold, _ = holdings_and_pnl()
        legs = S.leg_returns(r, hold)
        first = legs.iloc[0]
        self.assertAlmostEqual(first["long_exposure"], 0.015)
        self.assertAlmostEqual(first["short_exposure"], 0.015)
        self.assertAlmostEqual(first["long_return"], r["long_contribution"].iloc[0] / 0.015)

    def test_top_holdings_count_unheld_months_as_zero(self):
        _, pnl = holdings_and_pnl()
        top = S.top_holdings(pnl, set(), n=2).set_index("PERMNO")
        self.assertAlmostEqual(top.loc[1, "avg_weight"], 0.0075)
        self.assertAlmostEqual(top.loc[2, "avg_weight"], 0.0075 * 10 / 20)      # held half the months
        self.assertAlmostEqual(top.loc[3, "avg_weight"], -0.0075)
        self.assertEqual(top.loc[2, "months_held"], 10)

    def test_contributors_and_short_book(self):
        _, pnl = holdings_and_pnl()
        c = S.top_contributors(pnl, {(pnl["Date"].iloc[0], 1)}, n=1).set_index("rank_type")
        self.assertEqual(c.loc["top_positive", "PERMNO"], 1)         # +0.0075 * 0.01 * 20 months
        self.assertTrue(c.loc["top_positive", "label_needs_verification"])
        self.assertEqual(c.loc["top_negative", "PERMNO"], 4)
        sb = S.short_book(pnl)
        self.assertAlmostEqual(sb["share_small_cap_or_below_by_count"], 0.5)
        self.assertAlmostEqual(sb["share_plausibly_hard_to_borrow_by_weight"], 0.5)   # 3 small; 4 is $20M volume


class BuildTest(unittest.TestCase):
    def test_full_build_writes_every_deck_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            run, out = Path(d) / "run", Path(d) / "out"
            run.mkdir()
            returns().to_csv(run / "portfolio_returns.csv")
            hold, pnl = holdings_and_pnl()
            hold.to_csv(run / "holdings.csv", index=False)
            pnl.to_csv(run / "position_pnl.csv", index=False)
            st, figs = build.build_report(run, out, *market())
            self.assertEqual(len(figs), 10)
            self.assertTrue(all(f.exists() and f.stat().st_size > 5000 for f in figs))
            for t in ("return_statistics.csv", "risk_exposure_statistics.csv", "calendar_year_returns.csv",
                      "top10_holdings.csv", "top10_pnl_contributors.csv", "monthly_series.csv", "summary.json"):
                self.assertTrue((out / "tables" / t).exists(), t)
            self.assertIn("gross of trading costs", (out / "methodology.md").read_text())
            md = (out / "REPORT.md").read_text()
            for must in ("## Return statistics", "## Risk-adjusted performance", "## Calendar-year returns",
                         "charts/01_cumulative_return.png", "Information ratio vs cash + 4%", "T1 Co 1"):
                self.assertIn(must, md)
            import contextlib
            import io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                build.print_results(st, st.pop("_sections"), "TEST")
            printed = buf.getvalue()
            for must in ("RETURN STATISTICS", "RISK-ADJUSTED PERFORMANCE", "CALENDAR-YEAR RETURNS", "TOP 10 LONG / SHORT",
                         "10 LARGEST GAINS / LOSSES", "Beta vs S&P 500", "Maximum drawdown"):
                self.assertIn(must, printed)
            self.assertIn("information ratio", build.headline(st).lower())


if __name__ == "__main__":
    unittest.main()
