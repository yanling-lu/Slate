"""Synthetic check of the walk-forward base signal: it finds a planted signal, uses
only allowed data, and is unchanged when the future is removed."""
import unittest

import numpy as np
import pandas as pd

from base_signal.model import SOURCE, build_signals
from portfolio_agent.audit import truncation_test

FACTORS = [f"f{i}" for i in range(6)]
MONTHS = pd.date_range("2015-01-31", "2026-08-31", freq="ME")


def panel(seed=0, n=250):
    rng = np.random.default_rng(seed)
    rows = []
    for t in MONTHS:
        X = rng.normal(size=(n, len(FACTORS)))
        d = pd.DataFrame(X, columns=FACTORS)
        d["permno"], d["eom"] = np.arange(n), t
        d["prc"], d["dolvol_126d"], d["size_grp"] = 20.0, 5e7, "large"
        d["_noise"] = rng.normal(size=n)
        d["_x0"] = X[:, 0]
        rows.append(d)
    df = pd.concat(rows, ignore_index=True).sort_values(["permno", "eom"])
    # next month's return loads on f0 of this month (planted, genuine signal)
    df["ret_exc_lead1m"] = (0.02 * df["_x0"] + 0.05 * df["_noise"]).shift(0)
    last = df.groupby("permno")["eom"].transform("max") == df["eom"]
    df.loc[last, "ret_exc_lead1m"] = np.nan
    return df.drop(columns=["_noise", "_x0"]).sort_values(["eom", "permno"]).reset_index(drop=True)


class BaseSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = panel()
        cls.sig, cls.man = build_signals(cls.df, FACTORS)

    def test_finds_planted_feature_and_covers_every_formation_month(self):
        for e in self.man["years"]:
            self.assertIn("f0", e["characteristics"])
        self.assertEqual(self.sig["eom"].min(), pd.Timestamp("2020-12-31"))
        self.assertEqual(self.sig["eom"].max(), pd.Timestamp("2026-07-31"))
        self.assertEqual(set(self.sig["source"]), {SOURCE})

    def test_manifest_respects_schedule_and_limits(self):
        for e in self.man["years"]:
            y = e["year"]
            self.assertEqual(e["selection_window"], "validation")
            self.assertEqual(e["fit_data_end"], f"{y - 1}-12-31")
            self.assertEqual(e["train_target_end"], f"{y - 3}-12-31")
            self.assertLessEqual(len(e["characteristics"]), 7)

    def test_scores_unchanged_without_the_future(self):
        gen = lambda p: build_signals(p, FACTORS)[0][["permno", "eom", "score", "source"]]
        rep = truncation_test(gen, self.df, cutoffs=("2022-06-30",))
        self.assertFalse(rep.failed, [f for f in rep.items])

    def test_planted_signal_is_recovered_out_of_sample(self):
        m = self.sig.merge(self.df[["permno", "eom", "ret_exc_lead1m"]], on=["permno", "eom"]).dropna()
        ic = m.groupby("eom").apply(lambda g: g["score"].rank().corr(g["ret_exc_lead1m"].rank())).mean()
        self.assertGreater(ic, 0.15)


if __name__ == "__main__":
    unittest.main()
