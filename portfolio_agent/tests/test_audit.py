import copy
import unittest

import numpy as np
import pandas as pd

from portfolio_agent.audit import audit, truncation_test
from portfolio_agent.splits import split_for_year, test_year_of_formation

MONTHS = pd.date_range("2020-12-31", "2026-07-31", freq="ME")
N = 450


def make_raw(seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for t in MONTHS:
        frames.append(pd.DataFrame({
            "permno": np.arange(N), "eom": t, "x": rng.normal(size=N),
            "ret_exc_lead1m": rng.normal(0, 0.1, N)}))
    return pd.concat(frames, ignore_index=True)


def clean_gen(panel):
    return panel[["permno", "eom"]].assign(score=panel["x"])


def leaky_gen(panel):                       # peeks at next month's return
    return panel[["permno", "eom"]].assign(
        score=panel["x"] + 5 * panel["ret_exc_lead1m"].fillna(0))


def fullsample_gen(panel):                  # standardises with the whole history
    return panel[["permno", "eom"]].assign(score=panel["x"] - panel["x"].mean())


def manifest():
    years = []
    for y in range(2021, 2027):
        s = split_for_year(y)
        years.append({
            "year": y, "train_target_end": str(s["train_end"].date()),
            "selection_window": "validation", "fit_data_end": str(s["val_end"].date()),
            "feature_selection_data_end": str(s["train_end"].date()),
            "characteristics": ["ret_12_1", "be_me"], "events": ["2.02"], "mixtures_tried": 12})
    return {"years": years}


class SplitTest(unittest.TestCase):
    def test_schedule_matches_challenge(self):
        s = split_for_year(2021)
        self.assertEqual(s["train_end"], pd.Timestamp("2018-12-31"))
        self.assertEqual((s["val_start"], s["val_end"]),
                         (pd.Timestamp("2019-01-31"), pd.Timestamp("2020-12-31")))
        s = split_for_year(2022)
        self.assertEqual(s["train_end"], pd.Timestamp("2019-12-31"))
        self.assertEqual(split_for_year(2026)["test_end"], pd.Timestamp("2026-08-31"))
        self.assertEqual(test_year_of_formation("2020-12-31"), 2021)
        with self.assertRaises(ValueError):
            split_for_year(2020)


class TruncationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = make_raw()

    def test_clean_generator_passes(self):
        self.assertFalse(truncation_test(clean_gen, self.raw).failed)

    def test_target_peeking_fails(self):
        self.assertTrue(truncation_test(leaky_gen, self.raw).failed)

    def test_full_sample_statistics_fail(self):
        self.assertTrue(truncation_test(fullsample_gen, self.raw).failed)


class StaticAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = make_raw()
        cls.panel = cls.raw[["permno", "eom", "ret_exc_lead1m"]]

    def levels(self, rep, check):
        return {f.level for f in rep.items if f.check.startswith(check)}

    def test_clean_signal_passes(self):
        rep = audit(self.panel, clean_gen(self.raw), manifest())
        self.assertFalse(rep.failed, [f for f in rep.items if f.level == "FAIL"])

    def test_leaky_signal_flagged_by_ic(self):
        rep = audit(self.panel, leaky_gen(self.raw), manifest())
        self.assertIn("FAIL", self.levels(rep, "ic["))

    def test_missing_months_and_manifest(self):
        sig = clean_gen(self.raw)
        rep = audit(self.panel, sig[sig["eom"] < "2023-01-01"], None)
        self.assertIn("FAIL", self.levels(rep, "coverage"))
        self.assertIn("FAIL", self.levels(rep, "manifest"))

    def test_target_filtered_universe_flagged(self):
        panel = self.panel.copy()
        panel.loc[panel["permno"] < 100, "ret_exc_lead1m"] = np.nan
        sig = clean_gen(self.raw)
        sig = sig[sig["permno"] >= 100]                  # only names with a target are scored
        self.assertIn("FAIL", self.levels(audit(panel, sig, manifest()), "universe_vs_target"))

    def test_manifest_violations(self):
        m = copy.deepcopy(manifest())
        m["years"][0]["selection_window"] = "test"
        m["years"][1]["fit_data_end"] = "2022-03-31"      # fitted on the test year
        m["years"][2]["characteristics"] = ["a"] * 8
        rep = audit(self.panel, clean_gen(self.raw), m)
        bad = [f.check for f in rep.items if f.level == "FAIL"]
        for y in (2021, 2022, 2023):
            self.assertIn(f"manifest[{y}]", bad)

    def test_future_dated_input_is_a_contract_failure(self):
        sig = clean_gen(self.raw).assign(asof_date=lambda d: d["eom"] + pd.Timedelta(days=5))
        self.assertIn("FAIL", self.levels(audit(self.panel, sig, manifest()), "contract"))


if __name__ == "__main__":
    unittest.main()
