"""Look-ahead / leakage audit for the files the upstream agents hand to the portfolio agent.

    python -m portfolio_agent.audit --panel "Provided Documents/chars_final_with_names.parquet" \
        --signals inputs/signals.parquet --manifest inputs/manifest.json --da inputs/da.csv

Exit code 1 if any check FAILs. WARNs need a human decision. See SIGNALS_SPEC.md.

Static checks run on files. `truncation_test` re-runs a generator on a panel cut off
at a past date: a score that changes when future rows disappear used the future.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from .contracts import ContractError, read_table, validate_da, validate_signals
from .data import OUTCOME_COL, month_end
from .splits import FIRST_TEST_YEAR, LAST_TEST_YEAR, split_for_year

IC_WARN, IC_FAIL = 0.10, 0.20          # real monthly rank ICs are ~0.02-0.08
R2_WARN = 0.05                         # real OOS R2 for returns is ~1-2% (Gu et al.)
MIN_SCORED = 400                       # names scored per month, below this WARN
MAX_CHARS, MAX_EVENTS = 7, 3           # devil's-advocate shortlist sizes


@dataclass
class Finding:
    level: str        # PASS | WARN | FAIL | INFO
    check: str
    detail: str


class Report:
    def __init__(self):
        self.items: list[Finding] = []

    def add(self, level, check, detail=""):
        self.items.append(Finding(level, check, detail))

    @property
    def failed(self):
        return any(f.level == "FAIL" for f in self.items)

    def print(self):
        for f in self.items:
            print(f"[{f.level:4}] {f.check}: {f.detail}")
        n = {k: sum(f.level == k for f in self.items) for k in ("FAIL", "WARN", "PASS")}
        print(f"\n{n['FAIL']} FAIL, {n['WARN']} WARN, {n['PASS']} PASS")

    def to_json(self):
        return [f.__dict__ for f in self.items]


# --------------------------------------------------------------------------- helpers
def formation_months():
    return pd.date_range("2020-12-31", "2026-07-31", freq="ME")


def monthly_rank_ic(df: pd.DataFrame, score: str, target: str) -> pd.Series:
    """Spearman IC per formation month."""
    d = df.dropna(subset=[score, target])
    def one(g):
        return g[score].rank().corr(g[target].rank()) if len(g) > 30 else np.nan
    return d.groupby("eom")[[score, target]].apply(one).dropna()


def combined_score(sig: pd.DataFrame) -> pd.DataFrame:
    """Mean of per-source monthly z-scores, as the portfolio agent forms its alpha."""
    z = sig.groupby(["eom", "source"])["score"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
    return sig.assign(z=z).groupby(["permno", "eom"], as_index=False)["z"].mean().rename(
        columns={"z": "score"})


# --------------------------------------------------------------------------- checks
def check_coverage(sig, panel, rep):
    have = sig.groupby("eom")["permno"].nunique()
    missing = [t.date() for t in formation_months() if t not in have.index]
    if missing:
        rep.add("FAIL", "coverage", f"no scores for {len(missing)} formation months, "
                f"first {missing[0]}")
    else:
        rep.add("PASS", "coverage", "every formation month 2020-12..2026-07 has scores")
    thin = have[have < MIN_SCORED]
    if len(thin):
        rep.add("WARN", "breadth", f"{len(thin)} months score fewer than {MIN_SCORED} names "
                f"(min {int(thin.min())} in {thin.idxmin().date()})")
    outside = ~sig.set_index(["permno", "eom"]).index.isin(panel.set_index(["permno", "eom"]).index)
    if outside.any():
        rep.add("WARN", "unknown_rows", f"{int(outside.sum())} score rows are not in the panel")
    late = sig["eom"] > pd.Timestamp("2026-07-31")
    if late.any():
        rep.add("FAIL", "terminal_month", "scores for 2026-08 formation: no target month exists")


def check_universe(sig, panel, rep):
    """Target availability must not decide which stocks get scored."""
    p = panel[(panel["eom"] >= "2020-12-31") & (panel["eom"] <= "2026-07-31")].copy()
    key = pd.MultiIndex.from_frame(sig[["permno", "eom"]].drop_duplicates())
    p["scored"] = pd.MultiIndex.from_frame(p[["permno", "eom"]]).isin(key)
    null = p[OUTCOME_COL].isna()
    if null.sum() < 50:
        rep.add("INFO", "universe_vs_target", "too few null-target rows to test")
        return
    a, b = p.loc[null, "scored"].mean(), p.loc[~null, "scored"].mean()
    if b > 0 and a < 0.5 * b:
        rep.add("FAIL", "universe_vs_target", f"{a:.0%} of stock-months with no next-month "
                f"return are scored vs {b:.0%} of the rest: the prediction universe looks "
                "filtered on the target")
    else:
        rep.add("PASS", "universe_vs_target", f"scored share {a:.0%} (no target) vs {b:.0%}")


def check_ic(sig, panel, rep):
    """A score that is 'too good' against next month's return is a leak until proven otherwise."""
    d = panel[["permno", "eom", OUTCOME_COL]]
    for name, s in [("combined", combined_score(sig))] + [
            (src, g.rename(columns={})) for src, g in sig.groupby("source")]:
        m = s.merge(d, on=["permno", "eom"], how="inner")
        ic = monthly_rank_ic(m, "score", OUTCOME_COL)
        if ic.empty:
            rep.add("WARN", f"ic[{name}]", "no overlap with outcomes")
            continue
        t = ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else np.nan
        msg = f"mean rank IC {ic.mean():+.3f} (t={t:.1f}, {len(ic)} months, worst month {ic.min():+.3f})"
        if abs(ic.mean()) >= IC_FAIL:
            rep.add("FAIL", f"ic[{name}]", msg + f": above {IC_FAIL}, almost certainly look-ahead")
        elif abs(ic.mean()) >= IC_WARN:
            rep.add("WARN", f"ic[{name}]", msg + f": above {IC_WARN}, check how it was built")
        else:
            rep.add("PASS", f"ic[{name}]", msg)


def check_pred_r2(sig_raw, panel, rep):
    if "pred_ret" not in sig_raw.columns:
        rep.add("INFO", "oos_r2", "no pred_ret column; deck needs OOS R2 from the fitting code")
        return
    m = sig_raw.merge(panel[["permno", "eom", OUTCOME_COL]], on=["permno", "eom"]).dropna(
        subset=["pred_ret", OUTCOME_COL])
    r, p = m[OUTCOME_COL].values, m["pred_ret"].values
    r2 = 1 - ((r - p) ** 2).sum() / (r ** 2).sum()      # Gu, Kelly, Xiu: benchmark is zero
    msg = f"OOS R2 = {r2:.4%} on {len(m)} stock-months"
    rep.add("WARN" if (r2 > R2_WARN or r2 < 0) else "PASS", "oos_r2",
            msg + (" (implausibly high)" if r2 > R2_WARN else " (negative: worse than zero)" if r2 < 0 else ""))


def check_manifest(man, factor_list, rep):
    years = {int(y["year"]): y for y in man.get("years", [])}
    miss = [y for y in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1) if y not in years]
    if miss:
        rep.add("FAIL", "manifest_years", f"no entry for forecast years {miss}")
    for y, e in sorted(years.items()):
        try:
            s = split_for_year(y)
        except ValueError as ex:
            rep.add("FAIL", f"manifest[{y}]", str(ex))
            continue
        errs = []
        tag = f"manifest[{y}]"
        if e.get("selection_window") != "validation":
            errs.append(f"mixtures were chosen on '{e.get('selection_window')}', must be 'validation'")
        for fld, lim in [("train_target_end", s["train_end"]), ("fit_data_end", s["val_end"]),
                         ("feature_selection_data_end", s["val_end"])]:
            if fld not in e:
                errs.append(f"missing {fld}")
            elif month_end(e[fld]) > lim:
                errs.append(f"{fld} {e[fld]} is after the allowed {lim.date()}")
        chars, events = e.get("characteristics", []), e.get("events", [])
        if e.get("feature_set") in ("all", "composite"):   # declared departure from the 7-feature shortlist
            if factor_list is not None:
                if set(chars) - set(factor_list):
                    errs.append(f"not in factor_char_list.csv: {sorted(set(chars) - set(factor_list))}")
                if e["feature_set"] == "all" and set(chars) != set(factor_list):
                    errs.append("feature_set 'all' but characteristics differ from factor_char_list.csv")
            if len(events) > MAX_EVENTS:
                errs.append(f"{len(events)} events, limit {MAX_EVENTS}")
        elif len(chars) > MAX_CHARS or len(events) > MAX_EVENTS:
            errs.append(f"{len(chars)} characteristics / {len(events)} events, "
                        f"limits {MAX_CHARS}/{MAX_EVENTS}")
        unknown = [c for c in chars if factor_list is not None and c not in factor_list]
        if unknown and e.get("feature_set") not in ("all", "composite"):
            errs.append(f"not in factor_char_list.csv: {unknown}")
        if "mixtures_tried" not in e:
            errs.append("missing mixtures_tried (needed to judge multiple testing)")
        if errs:
            rep.add("FAIL", tag, "; ".join(errs))
        else:
            rep.add("PASS", tag, f"{len(chars)} chars + {len(events)} events, "
                    f"{e['mixtures_tried']} mixtures tried, selection on validation, "
                    f"data through {e['fit_data_end']}")
    if man.get("llm"):
        for k in ("model", "knowledge_cutoff", "lookahead_control"):
            if not man["llm"].get(k):
                rep.add("FAIL", "manifest_llm", f"missing llm.{k}: the deck must say how "
                        "model-side look-ahead was ruled out")
                break
        else:
            rep.add("PASS", "manifest_llm", f"{man['llm']['model']}: {man['llm']['lookahead_control']}")


def truncation_test(generate, raw_panel: pd.DataFrame, cutoffs=("2022-06-30", "2024-12-31"),
                    tol=1e-9) -> Report:
    """generate(panel) -> signals dataframe (permno, eom, score, source).

    For each cutoff c the panel is cut to eom <= c and the target of the cutoff row is
    blanked (it is realised after c). Scores for months <= c must be identical to the
    full-sample run; if not, they depended on rows from the future."""
    rep = Report()
    full = validate_signals(generate(raw_panel.copy()))
    for c in cutoffs:
        c = month_end(c)
        cut = raw_panel[raw_panel["eom"] <= c].copy()
        cut.loc[cut["eom"] == c, OUTCOME_COL] = np.nan
        part = validate_signals(generate(cut))
        m = part.merge(full, on=["permno", "eom", "source"], how="outer",
                       suffixes=("_cut", "_full"), indicator=True)
        m = m[m["eom"] <= c]
        only = m[m["_merge"] != "both"]
        diff = (m["score_cut"] - m["score_full"]).abs()
        bad = diff[diff > tol]
        if len(only) or len(bad):
            rep.add("FAIL", f"truncation[{c.date()}]",
                    f"{len(bad)} scores changed (max {bad.max() if len(bad) else 0:.3g}) and "
                    f"{len(only)} rows appeared/vanished when data after the cutoff was removed")
        else:
            rep.add("PASS", f"truncation[{c.date()}]", f"{len(m)} scores identical without the future")
    return rep


# --------------------------------------------------------------------------------- run
def audit(panel: pd.DataFrame, signals: pd.DataFrame, manifest: dict | None = None,
          da: pd.DataFrame | None = None, factor_list=None) -> Report:
    rep = Report()
    raw = signals.copy()
    raw.columns = [str(c).lower() for c in raw.columns]
    try:
        sig = validate_signals(signals)
        rep.add("PASS", "contract", f"{len(sig)} rows, sources: {sorted(sig['source'].unique())}")
    except ContractError as e:
        rep.add("FAIL", "contract", str(e))
        return rep
    panel = panel.copy()
    panel["eom"] = month_end(panel["eom"])
    check_coverage(sig, panel, rep)
    check_universe(sig, panel, rep)
    check_ic(sig, panel, rep)
    raw["eom"] = month_end(raw["eom"])
    check_pred_r2(raw, panel, rep)
    if manifest is None:
        rep.add("FAIL", "manifest", "no manifest supplied: cannot verify the walk-forward schedule")
    else:
        check_manifest(manifest, factor_list, rep)
    if da is not None:
        try:
            d = validate_da(da)
            known = d.set_index(["permno", "eom"]).index.isin(
                panel.set_index(["permno", "eom"]).index)
            rep.add("PASS" if known.all() else "WARN", "devils_advocate",
                    f"{len(d)} objections, {int((~known).sum())} not in panel")
        except ContractError as e:
            rep.add("FAIL", "devils_advocate", str(e))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--da")
    ap.add_argument("--factor-list", help="factor_char_list.csv (default: next to the panel)")
    ap.add_argument("--out", help="write the findings as json")
    a = ap.parse_args(argv)
    panel = pd.read_parquet(a.panel, columns=["permno", "eom", OUTCOME_COL])
    fl = Path(a.factor_list) if a.factor_list else Path(a.panel).with_name("factor_char_list.csv")
    factors = set(pd.read_csv(fl)["variable"]) if fl.exists() else None
    manifest = json.loads(Path(a.manifest).read_text()) if a.manifest else None
    rep = audit(panel, read_table(a.signals), manifest,
                read_table(a.da) if a.da else None, factors)
    rep.print()
    if a.out:
        Path(a.out).write_text(json.dumps(rep.to_json(), indent=2))
    raise SystemExit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
