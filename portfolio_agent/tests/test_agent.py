"""Run with:  python -m unittest discover -s portfolio_agent/tests -t ."""
import unittest

import numpy as np
import pandas as pd

from portfolio_agent.agent import PortfolioConstructionAgent
from portfolio_agent.config import Mandate, Policy
from portfolio_agent.contracts import ContractError, validate_da, validate_signals
from portfolio_agent.data import prepare_panel
from portfolio_agent.guardrails import MandateViolation, check_portfolio, enforce

MONTHS = pd.date_range("2020-12-31", periods=8, freq="ME")


def make_panel(n=700, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for t in MONTHS:
        ret = rng.normal(0.01, 0.10, n)
        frames.append(pd.DataFrame({
            "permno": np.arange(10000, 10000 + n), "eom": t,
            "ticker": [f"T{i}" for i in range(n)], "company_name": [f"Co {i}" for i in range(n)],
            "me": rng.lognormal(8, 1, n), "prc": rng.uniform(6, 200, n),
            "dolvol_126d": rng.lognormal(17, 1, n),
            "size_grp": rng.choice(["mega", "large", "small"], n),
            "gics": rng.choice([f"{s}101010" for s in range(10, 65, 5)], n),
            "beta_60m": np.where(rng.random(n) < 0.15, np.nan, rng.normal(1.1, 0.4, n)),
            "ret": ret, "ret_exc": ret - 0.003,
        }))
    return prepare_panel(pd.concat(frames, ignore_index=True))


def make_signals(panel, seed=1):
    rng = np.random.default_rng(seed)
    s = panel[["permno", "eom"]].copy()
    s["score"] = rng.normal(size=len(s))
    return s


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = make_panel()
        cls.signals = make_signals(cls.panel)
        cls.policy = Policy()
        cls.res = PortfolioConstructionAgent(cls.panel, policy=cls.policy).run(
            cls.signals, start=MONTHS[0], end=MONTHS[-1])

    def test_mandate_and_policy_every_month(self):
        m, p = self.res.monthly, self.policy
        self.assertEqual(len(m), len(MONTHS))
        n = m["n_long"] + m["n_short"]
        self.assertTrue(((n >= 100) & (n <= 500)).all())
        self.assertTrue((m["gross"] <= p.gross_target + 1e-6).all())
        self.assertTrue((m["net"].abs() <= 1e-6).all())
        self.assertTrue((m["max_weight"] <= p.max_weight + 1e-6).all())
        self.assertTrue((m["chk_beta_ex_ante"].abs() <= p.beta_tol + 1e-6).all())
        self.assertTrue((m["chk_max_sector_net"] <= p.sector_tol + 1e-6).all())
        self.assertTrue((m["relax_level"] == 0).all())
        self.assertTrue(self.res.checks["ok"].all())

    def test_turnover_budget(self):
        to = self.res.monthly["turnover_one_way"]
        self.assertTrue(pd.isna(to.iloc[0]))                     # initial build
        self.assertTrue((to.iloc[1:] <= self.policy.max_turnover + 1e-6).all())
        self.assertGreater(to.iloc[1:].min(), 0.05)              # random signals: budget binds

    def test_holdings_match_trades(self):
        h, t = self.res.holdings, self.res.trades
        self.assertEqual(list(h.columns[:5]), ["Date", "PERMNO", "TICKER", "COMPANY NAME", "WEIGHT"])
        self.assertTrue((h["Date"].dt.day == 1).all())
        self.assertFalse(h.duplicated(["Date", "PERMNO"]).any())
        self.assertTrue((t["reason"].str.len() > 0).all())
        first = t[t["date"] == h["Date"].min()]
        self.assertTrue(first["action"].isin(["OPEN_LONG", "OPEN_SHORT"]).all())

    def test_devils_advocate(self):
        t0 = MONTHS[0]
        base = self.res.holdings[self.res.holdings["eom"] == t0]
        longs = base[base["WEIGHT"] > 0]["PERMNO"].tolist()
        da = pd.DataFrame({
            "permno": longs[:10], "eom": t0,
            "action": ["veto"] * 5 + ["haircut"] * 5, "severity": 0.6,
            "reason": "disconfirming 8-K",
        })
        res = PortfolioConstructionAgent(self.panel, policy=self.policy).run(
            self.signals, da, start=t0, end=t0)
        w = res.holdings.set_index("PERMNO")["WEIGHT"]
        self.assertFalse(set(longs[:5]) & set(w.index))
        for permno in longs[5:10]:
            self.assertLessEqual(abs(w.get(permno, 0.0)), self.policy.max_weight * 0.4 + 1e-6)

    def test_objection_score_adapter(self):
        raw = pd.DataFrame({"PERMNO": [1, 2, 3, 4], "eom": MONTHS[0],
                            "objection_score": [0, 1, 2, 3], "side": "long"})
        out = validate_da(raw)
        self.assertEqual(out["action"].tolist(), ["pass", "pass", "haircut", "veto"])
        self.assertEqual(out["severity"].tolist(), [0.0, 0.0, 0.5, 1.0])
        for bad in (4, -1, 1.5, None):
            with self.assertRaises(ContractError):
                validate_da(raw.assign(objection_score=[0, 1, 2, bad]))
        with self.assertRaises(ContractError):
            validate_da(raw.assign(action="veto"))

    def test_delisted_name_is_closed(self):
        victim = int(self.res.holdings[self.res.holdings["eom"] == MONTHS[0]]["PERMNO"].iloc[0])
        panel = self.panel[~((self.panel["permno"] == victim) & (self.panel["eom"] >= MONTHS[1]))]
        res = PortfolioConstructionAgent(panel, policy=self.policy).run(
            self.signals, start=MONTHS[0], end=MONTHS[1])
        second = res.holdings[res.holdings["eom"] == MONTHS[1]]
        self.assertNotIn(victim, second["PERMNO"].values)
        row = res.trades[(res.trades["permno"] == victim) & (res.trades["weight_after"] == 0)]
        self.assertIn("forced exit", row["reason"].iloc[0])

    def test_leakage_is_rejected(self):
        s = self.signals.copy()
        s["ret_exc_lead1m"] = 0.0
        with self.assertRaises(ContractError):
            validate_signals(s)
        s = self.signals.copy()
        s["asof_date"] = s["eom"] + pd.Timedelta(days=3)
        with self.assertRaises(ContractError):
            validate_signals(s)
        with self.assertRaises(ContractError):
            PortfolioConstructionAgent(self.panel.assign(ret_exc_lead1m=0.0))

    def test_too_few_names_raises(self):
        small = self.panel[self.panel["permno"] < 10060]
        with self.assertRaises(MandateViolation):
            PortfolioConstructionAgent(small).run(self.signals, start=MONTHS[0], end=MONTHS[0])

    def test_guardrails_catch_bad_books(self):
        m, p = Mandate(), Policy()
        few = pd.Series(0.02, index=range(40))                   # 40 names, 80% net long
        failed = {c.name for c in check_portfolio(few, m, p) if not c.ok and c.level == "hard"}
        self.assertEqual(failed, {"positions", "net_exposure"})
        levered = pd.Series([0.011] * 100 + [-0.011] * 100, index=range(200))   # 220% gross
        with self.assertRaises(MandateViolation):
            enforce(check_portfolio(levered, m, p), "test")


if __name__ == "__main__":
    unittest.main()
