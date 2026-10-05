"""Composite-factor model: signs, stability filter, cost rule, weights, and no look-ahead."""
import unittest

import numpy as np
import pandas as pd

from base_signal.composites import FAMILIES, EXTRA_FAMILIES, build_composite_signals, check_families, composites
from portfolio_agent.audit import Report, check_manifest, truncation_test

MONTHS = pd.date_range("2015-01-31", "2026-08-31", freq="ME")
N = 300
FAMS = {"good": {"a1": 1, "a2": 1}, "bad": {"b1": 1, "b2": 1}, "neg": {"c1": -1, "c2": -1}}


def panel(seed=0, bad_sign=-1.0):
    """Next-month return loads +0.03 on `good`, bad_sign*0.03 on `bad`, -0.03 on c (so `neg`, sign -1, is good)."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in MONTHS:
        f = {k: rng.normal(size=N) for k in ("a1", "a2", "b1", "b2", "c1", "c2")}
        d = pd.DataFrame(f)
        d["ret_exc_lead1m"] = (0.03 * (f["a1"] + f["a2"]) / 2 + bad_sign * 0.03 * (f["b1"] + f["b2"]) / 2
                               - 0.03 * (f["c1"] + f["c2"]) / 2 + 0.05 * rng.normal(size=N))
        d["permno"], d["eom"] = np.arange(N), t
        d["prc"], d["dolvol_126d"], d["size_grp"] = 20.0, 5e7, "large"
        d["gics"] = np.where(np.arange(N) % 2 == 0, "10101010", "20202020")
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df.loc[df["eom"] == MONTHS[-1], "ret_exc_lead1m"] = np.nan
    return df.sort_values(["eom", "permno"]).reset_index(drop=True)


def run(df, fams=FAMS, **kw):
    return build_composite_signals(df, fams, log=lambda *a, **k: None, **kw)


class CompositeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = panel()
        cls.sig, cls.man = run(cls.df)

    def test_default_families_are_known_columns_with_fixed_signs(self):
        factors = pd.read_csv("data/factor_char_list.csv")["variable"].tolist()
        check_families({**FAMILIES, **EXTRA_FAMILIES}, factors)
        self.assertEqual(set(FAMILIES), {"value", "profitability", "momentum"})
        self.assertTrue(all(s in (-1, 1) for f in FAMILIES.values() for s in f.values()))
        with self.assertRaises(ValueError):
            check_families({"x": {"not_a_column": 1}}, factors)

    def test_good_families_kept_and_wrong_way_family_dropped_every_year(self):
        for e in self.man["years"]:
            f = e["families"]
            self.assertTrue(f["good"]["kept"] and f["neg"]["kept"], e["year"])      # sign -1 handled
            self.assertFalse(f["bad"]["kept"], e["year"])
            self.assertTrue(any("sub-periods" in r for r in f["bad"]["reasons"]))
            self.assertEqual(e["weights"]["bad"], 0.0)
            self.assertAlmostEqual(sum(e["weights"].values()), 1.0)
            self.assertFalse(e["all_families_failed_stability_equal_weights_used"])

    def test_wrong_fixed_sign_is_not_rescued_by_the_data(self):
        wrong = {"good": {"a1": -1, "a2": -1}}               # sign flipped against theory
        sig, man = run(self.df, wrong)
        self.assertTrue(all(e["all_families_failed_stability_equal_weights_used"] for e in man["years"]))
        self.assertTrue(all(abs(e["weights"]["good"] - 1.0) < 1e-9 for e in man["years"]))   # equal = 1 family

    def test_cost_rule_drops_a_family_when_costs_are_prohibitive(self):
        _, man = run(self.df, cost_bps=100000)
        for e in man["years"]:
            for f in e["families"].values():
                self.assertFalse(f["kept"])
                self.assertTrue(any("turnover costs" in r for r in f["reasons"]))
            self.assertTrue(e["all_families_failed_stability_equal_weights_used"])
            self.assertTrue(all(abs(w - 1 / 3) < 1e-3 for w in e["weights"].values()))

    def test_scores_cover_every_formation_month_and_recover_the_planted_signal(self):
        self.assertEqual(self.sig["eom"].min(), pd.Timestamp("2020-12-31"))
        self.assertEqual(self.sig["eom"].max(), pd.Timestamp("2026-07-31"))
        m = self.sig.merge(self.df[["permno", "eom", "ret_exc_lead1m"]], on=["permno", "eom"]).dropna()
        ic = m.groupby("eom").apply(lambda g: g["score"].rank().corr(g["ret_exc_lead1m"].rank())).mean()
        self.assertGreater(ic, 0.2)
        self.assertTrue(self.sig["rationale"].str.contains("good").all())      # explanation per stock

    def test_composite_uses_only_the_same_month(self):
        a = composites(self.df, FAMS)["good"]
        shuffled = self.df.copy()
        later = shuffled["eom"] > pd.Timestamp("2022-01-31")
        shuffled.loc[later, "a1"] = shuffled.loc[later, "a1"].sample(frac=1, random_state=1).to_numpy()
        b = composites(shuffled, FAMS)["good"]
        np.testing.assert_allclose(a[~later.to_numpy()], b[~later.to_numpy()])

    def test_scores_unchanged_without_the_future(self):
        gen = lambda p: build_composite_signals(p, FAMS, log=lambda *a, **k: None)[0][["permno", "eom", "score", "source"]]
        self.assertFalse(truncation_test(gen, self.df, cutoffs=("2022-06-30",)).failed)

    def test_manifest_passes_the_audit_for_a_composite_feature_set(self):
        factors = {c for fam in FAMS.values() for c in fam}
        rep = Report()
        check_manifest(self.man, factors, rep)
        self.assertFalse(rep.failed, [f for f in rep.items if f.level == "FAIL"])
        self.assertEqual(sum(f.level == "PASS" for f in rep.items), 6)
        bad = {**self.man, "years": [{**e, "characteristics": e["characteristics"] + ["not_a_factor"]} for e in self.man["years"]]}
        rep2 = Report()
        check_manifest(bad, factors, rep2)
        self.assertTrue(rep2.failed)


class OutputIsCsvTest(unittest.TestCase):
    def test_run_writes_csv_not_parquet_and_the_portfolio_contract_reads_it_back(self):
        import tempfile
        from pathlib import Path

        from base_signal.composites import run
        from portfolio_agent.contracts import read_table, validate_signals
        df = panel()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            df.to_parquet(d / "panel.parquet")                         # the challenge data stays parquet
            pd.DataFrame({"variable": [c for fam in FAMS.values() for c in fam]}).to_csv(d / "factors.csv", index=False)
            sig, _ = run(d / "panel.parquet", d / "factors.csv", d / "out", families=FAMS)
            self.assertTrue((d / "out" / "signals.csv").exists())
            self.assertEqual(list(d.glob("out/*.parquet")), [])
            back = validate_signals(read_table(d / "out" / "signals.csv"))
            self.assertEqual(len(back), len(sig))
            self.assertEqual(back["eom"].min(), pd.Timestamp("2020-12-31"))
            np.testing.assert_allclose(np.sort(back["score"].to_numpy()), np.sort(sig["score"].to_numpy()), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
