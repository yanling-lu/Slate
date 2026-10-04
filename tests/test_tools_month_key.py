"""The tools must match the panel's month-END `eom` key and respect the as-of cutoff."""
import unittest
from datetime import date
from unittest import mock

import numpy as np
import pandas as pd

from tools import characteristics, peers
from tools.data_loader import month_slice

FACTORS = ["f1", "f2", "f3"]


def panel():
    rng = np.random.default_rng(0)
    rows = []
    for eom, last_td in [("2024-05-31", "2024-05-31"), ("2024-06-30", "2024-06-28")]:
        for p in range(1, 41):
            rows.append({"permno": p, "eom": pd.Timestamp(eom), "date": date.fromisoformat(last_td),
                         "sic": 3571 if p <= 10 else 2834, "market_equity": 100.0 + p,
                         "f1": rng.normal(), "f2": rng.normal(), "f3": rng.normal(),
                         "ret_exc_lead1m": 0.5})
    df = pd.DataFrame(rows)
    df.loc[(df["permno"] == 1) & (df["eom"] == "2024-06-30"), "f1"] = 25.0   # extreme high
    df.loc[(df["permno"] == 1) & (df["eom"] == "2024-06-30"), "f2"] = -25.0  # extreme low
    return df


class MonthKeyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = panel()

    def patched(self):
        return mock.patch.multiple(characteristics, load_chars=lambda: self.df,
                                   load_factor_list=lambda: FACTORS)

    def test_month_slice_uses_month_end(self):
        s = month_slice(self.df, date(2024, 6, 30))
        self.assertEqual(set(s["eom"]), {pd.Timestamp("2024-06-30")})
        self.assertEqual(len(s), 40)

    def test_snapshot_not_visible_before_its_date(self):
        # the June snapshot was taken on 2024-06-28; on 2024-06-27 it did not exist yet
        self.assertTrue(month_slice(self.df, date(2024, 6, 27)).empty)
        self.assertEqual(len(month_slice(self.df, date(2024, 6, 28))), 40)

    def test_get_characteristics_returns_data(self):
        with self.patched():
            out = characteristics.get_characteristics(1, date(2024, 6, 30), top_k=2)
        self.assertEqual({o["feature_id"] for o in out}, {"f1", "f2"})

    def test_sign_and_direction_are_preserved(self):
        with self.patched():
            out = {o["feature_id"]: o for o in characteristics.get_characteristics(1, date(2024, 6, 30))}
        self.assertGreater(out["f1"]["z_score_vs_month"], 3)
        self.assertEqual(out["f1"]["direction"], "high")
        self.assertLess(out["f2"]["z_score_vs_month"], -3)
        self.assertEqual(out["f2"]["direction"], "low")

    def test_target_column_is_never_returned(self):
        with self.patched():
            out = characteristics.get_characteristics(1, date(2024, 6, 30), top_k=50)
        self.assertNotIn("ret_exc_lead1m", {o["feature_id"] for o in out})

    def test_unknown_permno_and_early_date_are_empty(self):
        with self.patched():
            self.assertEqual(characteristics.get_characteristics(999, date(2024, 6, 30)), [])
            self.assertEqual(characteristics.get_characteristics(1, date(2024, 6, 27)), [])

    def test_peers_found(self):
        with mock.patch.object(peers, "load_chars", lambda: self.df):
            out = peers.get_peer_context(1, date(2024, 6, 30))
        self.assertEqual(out["n_peers"], 10)
        self.assertEqual(out["sector"], "3571")
        with mock.patch.object(peers, "load_chars", lambda: self.df):
            self.assertEqual(peers.get_peer_context(1, date(2024, 6, 27)), {})


if __name__ == "__main__":
    unittest.main()
