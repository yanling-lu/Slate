"""Pipeline-contract tests for agents.portfolio.PortfolioReconstructionAgent.

Run from the repo root:  python -m pytest tests/test_portfolio_adapter.py
                    or:  python -m unittest tests.test_portfolio_adapter
"""
import unittest

import pandas as pd

from agents.portfolio import (LLM_SOURCE, PortfolioReconstructionAgent, challenges_to_da,
                              signals_to_frame, to_eom)
from portfolio_agent.contracts import ContractError
from portfolio_agent.guardrails import MandateViolation
from portfolio_agent.tests.test_agent import MONTHS, make_panel, make_signals

try:
    from guardrails.policy import check_constraints
    from schemas.models import Position, Signal
except Exception:                                   # pydantic not installed
    check_constraints = None


def signal(permno, direction="long", score=1.5, conf=0.7, asof="2020-12-31", **kw):
    d = dict(permno=permno, ticker="ZZZ", as_of_date=asof, direction=direction, score=score,
             confidence=conf, catalyst="Item 2.02 beat", time_horizon_months=1,
             rationale="raised guidance", evidence=[], insufficient_evidence=False)
    d.update(kw)
    return d


def challenge(permno, rec="proceed", resilience=0.8, **kw):
    d = dict(permno=permno, ticker="ZZZ", signal_id="x", resilience_score=resilience,
             recommendation=rec, counterarguments=[{"type": "crowding", "claim": "crowded",
                                                      "evidence": [], "severity": "high"}])
    d.update(kw)
    return d


def batch(*sigs, asof="2020-12-31"):
    return {"as_of_date": asof, "signals": list(sigs), "audit_trail": []}


class AdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = make_panel()
        cls.base = make_signals(cls.panel).assign(source="chars_model")

    def agent(self):
        return PortfolioReconstructionAgent(model="gpt-4o", panel=self.panel, base_signals=self.base)

    # ---------------------------------------------------------------- output shape
    def test_returns_portfolio_target_shape(self):
        out = self.agent().run(batch(), [], "2020-12-31")
        self.assertEqual(out["as_of_date"], "2020-12-31")
        self.assertEqual(out["holdings_date"], "2021-01-01")
        tw = out["target_weights"]
        self.assertTrue(all(set(p) == {"permno", "ticker", "weight"} for p in tw))
        self.assertTrue(100 <= len(tw) <= 500)
        self.assertLessEqual(sum(abs(p["weight"]) for p in tw), 2.0)
        self.assertLessEqual(abs(sum(p["weight"] for p in tw)), 0.5)
        self.assertEqual(set(out["sectors"]), {p["permno"] for p in tw})
        self.assertFalse(out["human_approval_required"])
        self.assertTrue(all(t["reason"] for t in out["trade_rationale"]))

    @unittest.skipIf(check_constraints is None, "pydantic not installed")
    def test_passes_the_graphs_policy_node(self):
        out = self.agent().run(batch(), [], "2020-12-31")
        positions = [Position(**p) for p in out["target_weights"]]
        self.assertEqual(check_constraints(positions, out["sectors"]), [])

    @unittest.skipIf(check_constraints is None, "pydantic not installed")
    def test_accepts_real_pydantic_objects(self):
        from schemas.models import SignalBatch
        sb = SignalBatch.model_validate(batch(signal(10005, evidence=[{
            "kind": "characteristic", "feature_id": "be_me", "value": 1.0,
            "z_score_vs_month": 2.0, "direction": "high", "interpretation": "cheap"}])))
        out = self.agent().run(sb, [], "2020-12-31")
        self.assertTrue(out["target_weights"])

    # ------------------------------------------------------------ signal handling
    def test_llm_signal_tilts_the_book(self):
        # pick a name the dense signal likes least and give it a strong long view
        m0 = self.base[self.base["eom"] == MONTHS[0]].sort_values("score")
        name = int(m0["permno"].iloc[5])
        quiet = {p["permno"]: p["weight"] for p in self.agent().run(batch(), [], "2020-12-31")["target_weights"]}
        loud = {p["permno"]: p["weight"] for p in self.agent().run(
            batch(signal(name, score=2.0, conf=1.0)), [], "2020-12-31")["target_weights"]}
        self.assertLess(quiet.get(name, 0.0), 0.0)
        self.assertGreater(loud.get(name, 0.0), quiet.get(name, 0.0))

    def test_neutral_and_insufficient_evidence_are_dropped(self):
        eom = to_eom("2020-12-31")
        df = signals_to_frame(batch(
            signal(1, direction="neutral", score=0.1, conf=0.3),
            signal(2, insufficient_evidence=True, conf=0.3),
            signal(3)), eom)
        self.assertEqual(df["permno"].tolist(), [3])
        self.assertEqual(df["source"].tolist(), [LLM_SOURCE])

    def test_multiple_signals_for_a_stock_are_merged(self):
        eom = to_eom("2020-12-31")
        df = signals_to_frame(batch(signal(7, score=2.0, conf=0.9), signal(7, score=-1.0, conf=0.1)), eom)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df["score"].iloc[0], (2.0 * 0.9 - 0.1) / 1.0)
        self.assertEqual(df["confidence"].iloc[0], 0.9)

    def test_future_dated_signal_is_rejected(self):
        with self.assertRaises(ContractError):
            signals_to_frame(batch(signal(1, asof="2021-01-15")), to_eom("2020-12-31"))
        with self.assertRaises(ContractError):
            signals_to_frame(batch(signal(1), asof="2021-01-31"), to_eom("2020-12-31"))

    def test_mid_month_as_of_is_rejected(self):
        with self.assertRaises(ContractError):
            to_eom("2020-12-15")
        self.assertEqual(to_eom("2020-12-30"), pd.Timestamp("2020-12-31"))

    # --------------------------------------------------------- devil's advocate
    def test_challenge_mapping(self):
        eom = to_eom("2020-12-31")
        sb = batch(signal(1), signal(2, direction="short", score=-1.0), signal(3), signal(4))
        da = challenges_to_da([
            challenge(1, "reject"), challenge(2, "revise", resilience=0.4),
            challenge(3, "proceed"), challenge(4, "proceed", objection_score=2)], sb, eom
        ).set_index("permno")
        self.assertEqual(da.loc[1, ["action", "severity", "side"]].tolist(), ["veto", 1.0, "long"])
        self.assertEqual(da.loc[2, ["action", "side"]].tolist(), ["haircut", "short"])
        self.assertAlmostEqual(da.loc[2, "severity"], 0.6)
        self.assertEqual(da.loc[3, "action"], "pass")
        self.assertEqual(da.loc[4, ["action", "severity"]].tolist(), ["haircut", 0.5])
        self.assertIn("crowded", da.loc[1, "reason"])

    def test_revise_severity_is_bounded(self):
        da = challenges_to_da([challenge(1, "revise", resilience=0.99),
                               challenge(2, "revise", resilience=0.0)], batch(), to_eom("2020-12-31"))
        self.assertEqual(da.set_index("permno")["severity"].round(2).to_dict(), {1: 0.25, 2: 0.9})

    def test_reject_vetoes_the_long(self):
        a = self.agent()
        base = {p["permno"]: p["weight"] for p in a.run(batch(), [], "2020-12-31")["target_weights"]}
        longs = [p for p, w in base.items() if w > 0]
        victim = longs[0]
        out = self.agent().run(batch(signal(victim, score=0.5)), [challenge(victim, "reject")], "2020-12-31")
        w = {p["permno"]: p["weight"] for p in out["target_weights"]}
        self.assertLessEqual(w.get(victim, 0.0), 0.0)

    # ------------------------------------------------------------------- state
    def test_accumulates_within_a_month_and_commits_on_the_next(self):
        a = self.agent()
        a.run(batch(signal(10001)), [], "2020-12-31")
        out = a.run(batch(signal(10002)), [], "2020-12-31")          # same month, second stock
        self.assertEqual(set(a._sig), {10001, 10002})
        self.assertTrue(out["target_weights"])
        nxt = a.run(batch(asof="2021-01-29"), [], "2021-01-29")
        self.assertEqual(nxt["as_of_date"], "2021-01-31")
        self.assertEqual(set(a._sig), set())                          # new month starts clean
        turnover = nxt["constraint_report"]["diagnostics"]["turnover_one_way"]
        self.assertIsNotNone(turnover)                                # a prior book exists now
        self.assertLessEqual(turnover, 0.30 + 1e-6)
        self.assertEqual(sorted(a.holdings_frame()["eom"].unique().tolist()),
                         [MONTHS[0], MONTHS[1]])

    def test_out_of_order_and_gaps_are_rejected(self):
        a = self.agent()
        a.run(batch(), [], "2021-01-29")
        with self.assertRaises(ContractError):
            a.run(batch(), [], "2020-12-31")
        b = self.agent()
        b.run(batch(), [], "2020-12-31")
        with self.assertRaises(ContractError):
            b.run(batch(), [], "2021-02-26")

    def test_no_dense_signal_fails_loudly(self):
        a = PortfolioReconstructionAgent(panel=self.panel)
        with self.assertRaises(ContractError):
            a.run(batch(signal(10001)), [], "2020-12-31")

    def test_llm_alone_cannot_fill_the_book(self):
        sparse = self.base[self.base["permno"] < 10030]          # only 30 names have a base score
        a = PortfolioReconstructionAgent(panel=self.panel, base_signals=sparse)
        with self.assertRaises(MandateViolation):
            a.run(batch(), [], "2020-12-31")

    def test_reserved_source_name_is_rejected(self):
        a = PortfolioReconstructionAgent(panel=self.panel, base_signals=self.base.assign(source=LLM_SOURCE))
        with self.assertRaises(ContractError):
            a.run(batch(), [], "2020-12-31")


if __name__ == "__main__":
    unittest.main()
