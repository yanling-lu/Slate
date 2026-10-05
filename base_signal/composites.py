"""Composite-factor base signal: a few theory-backed families, signs fixed in advance.

Why: with returns this noisy, fitting 147 coefficients (or a different 7 features each year)
mostly fits noise. Here nothing is estimated per feature. The only fitted pieces are a handful
of family weights, chosen on VALIDATION months only.

Pipeline, per forecast year (same walk-forward schedule as the rest of the repo):

1. Families (default: value, profitability, momentum). Each is a list of characteristics with a
   sign fixed from theory and never estimated (`FAMILIES`). Extra families (`EXTRA_FAMILIES`)
   can be switched on via the `families=` argument.
2. Composite: for every month, rank each member within the tradable universe, flip the sign,
   average the ranks of the members that are present (at least half must be), re-rank. The
   composite needs no fitting, so it cannot leak.
3. Stability filter, on the VALIDATION window only. A family is kept only if
     a) its rank IC against next-month sector-relative return is positive in at least
        `MIN_BLOCKS_POSITIVE` of `N_BLOCKS` equal sub-periods,
     b) its top-minus-bottom decile spread is still positive after turnover costs
        (cost = 2 x (replaced fraction of each leg) x cost_bps).
   (The universe is already the tradable one: price, dollar volume, size group.)
4. Weights: grid over the simplex (step 0.25) across the kept families, scored by validation
   rank IC; ties go to the weights closest to equal. If no family passes, equal weights across
   all families are used and the manifest says so.
5. Score the test year with the blended composite. `pred_ret` maps the score to a return with a
   one-variable regression fitted on training + validation rows.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from portfolio_agent.splits import FIRST_TEST_YEAR, LAST_TEST_YEAR, split_for_year

from .model import TARGET, _month_groups, mean_rank_ic, test_report, tradable, winsorize

SOURCE = "chars_composite"
N_BLOCKS, MIN_BLOCKS_POSITIVE = 4, 3
WEIGHT_STEP = 4                      # weights are multiples of 1/WEIGHT_STEP
DEFAULT_COST_BPS = 10.0              # per dollar traded; the challenge supplies no cost model

# member -> sign (+1: higher value is better). Fixed from theory, never estimated.
FAMILIES: dict[str, dict[str, int]] = {
    "value": {"be_me": 1, "ni_me": 1, "ocf_me": 1, "fcf_me": 1, "sale_me": 1,
              "ebitda_mev": 1, "bev_mev": 1},
    "profitability": {"gp_at": 1, "op_at": 1, "cop_at": 1, "ope_be": 1, "ni_be": 1,
                      "niq_be": 1, "f_score": 1, "qmj_prof": 1, "ebit_sale": 1, "ocf_at": 1},
    "momentum": {"ret_12_1": 1, "ret_9_1": 1, "ret_6_1": 1, "ret_12_7": 1,
                 "resff3_12_1": 1, "resff3_6_1": 1},
}
EXTRA_FAMILIES: dict[str, dict[str, int]] = {
    "low_risk": {"beta_60m": -1, "betabab_1260d": -1, "ivol_ff3_21d": -1, "rvol_21d": -1,
                 "qmj_safety": 1},
    "investment": {"at_gr1": -1, "noa_gr1a": -1, "capx_gr1": -1, "chcsho_12m": -1,
                   "eqnetis_at": -1, "netis_at": -1},
}


def columns_needed(families) -> list[str]:
    return sorted({c for fam in families.values() for c in fam})


def check_families(families, factor_list) -> None:
    bad = [c for fam in families.values() for c in fam if c not in set(factor_list)]
    if bad:
        raise ValueError(f"family members not in factor_char_list.csv: {bad}")
    for name, fam in families.items():
        if any(s not in (-1, 1) for s in fam.values()):
            raise ValueError(f"family {name}: signs must be +1 or -1")


def composites(df: pd.DataFrame, families) -> dict[str, np.ndarray]:
    """One composite rank in [-0.5, 0.5] per family, row-aligned with df. Missing -> 0."""
    out = {}
    eoms = df["eom"].to_numpy()
    for name, members in families.items():
        cols = list(members)
        ranks = (df.groupby("eom")[cols].rank(pct=True) - 0.5).to_numpy(dtype=np.float64)
        ranks = ranks * np.array([members[c] for c in cols])
        count = np.isfinite(ranks).sum(axis=1)
        total = np.where(np.isfinite(ranks), ranks, 0.0).sum(axis=1)
        comp = np.where(count >= max(1, int(np.ceil(len(cols) / 2))), total / np.maximum(count, 1), np.nan)
        rr = pd.Series(comp).groupby(eoms).rank(pct=True) - 0.5
        out[name] = rr.fillna(0.0).to_numpy()
    return out


def sector_relative(df: pd.DataFrame) -> np.ndarray:
    """Next-month excess return minus its GICS-sector-month median (month median if < 5 peers)."""
    y = df[TARGET]
    sector = df["gics"].astype("string").str.slice(0, 2).fillna("NA")
    g = y.groupby([df["eom"], sector])
    med = np.where(g.transform("count") >= 5, g.transform("median"), y.groupby(df["eom"]).transform("median"))
    return (y - med).to_numpy(dtype=np.float64)


def _decile_net_spread(comp, y_w, permno, eoms, mask, cost_bps):
    """Mean monthly top-minus-bottom decile spread, gross and net of turnover costs."""
    months = np.unique(eoms[mask])
    spreads, costs, t_top, t_bot = [], [], [], []
    prev_top = prev_bot = None
    for e in months:
        idx = np.flatnonzero(mask & (eoms == e))
        c = comp[idx]
        hi, lo = np.quantile(c, [0.9, 0.1])
        top, bot = idx[c >= hi], idx[c <= lo]
        if len(top) == 0 or len(bot) == 0:
            continue
        spreads.append(y_w[top].mean() - y_w[bot].mean())
        s_top, s_bot = set(permno[top]), set(permno[bot])
        if prev_top is not None:
            f_top = 1 - len(s_top & prev_top) / len(s_top)
            f_bot = 1 - len(s_bot & prev_bot) / len(s_bot)
            t_top.append(f_top)
            t_bot.append(f_bot)
            costs.append(2 * (f_top + f_bot) * cost_bps / 1e4)   # sell f and buy f of each leg
        prev_top, prev_bot = s_top, s_bot
    gross = float(np.mean(spreads)) if spreads else np.nan
    cost = float(np.mean(costs)) if costs else 0.0
    return {"gross_spread": gross, "net_spread": gross - cost,
            "turnover_top": float(np.mean(t_top)) if t_top else np.nan,
            "turnover_bottom": float(np.mean(t_bot)) if t_bot else np.nan}


def _weight_grid(k: int):
    for combo in itertools.product(range(WEIGHT_STEP + 1), repeat=k):
        if sum(combo) == WEIGHT_STEP:
            yield np.array(combo, dtype=float) / WEIGHT_STEP


def build_composite_signals(df: pd.DataFrame, families=None, years=range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1),
                            cost_bps: float = DEFAULT_COST_BPS, log=print):
    """df: permno, eom, the family members, gics, prc, dolvol_126d, size_grp, ret_exc_lead1m."""
    families = families or FAMILIES
    df = df[tradable(df).to_numpy()].reset_index(drop=True)
    eoms = df["eom"].to_numpy()
    permno = df["permno"].to_numpy()
    tgt_month = (df["eom"] + pd.offsets.MonthEnd(1)).to_numpy()
    comp = composites(df, families)
    names = list(families)
    y_raw = df[TARGET].to_numpy(dtype=np.float64)
    y_rel = sector_relative(df)
    yrank = np.nan_to_num((pd.Series(y_rel).groupby(eoms).rank(pct=True) - 0.5).to_numpy())
    has_y = ~np.isnan(y_raw)

    out, entries = [], []
    for year in years:
        t0 = time.time()
        s = {k: np.datetime64(v) for k, v in split_for_year(year).items()}
        in_tr = has_y & (tgt_month >= s["train_start"]) & (tgt_month <= s["train_end"])
        in_va = has_y & (tgt_month >= s["val_start"]) & (tgt_month <= s["val_end"])
        in_te = (tgt_month >= s["test_start"]) & (tgt_month <= s["test_end"])
        if not in_te.any() or not in_va.any():
            continue
        lo, hi = np.percentile(y_raw[in_tr], [1, 99])
        y_w = np.clip(np.where(has_y, y_raw, 0.0), lo, hi)
        val_months = np.unique(eoms[in_va])
        blocks = np.array_split(val_months, N_BLOCKS)

        diag, kept = {}, []
        for name in names:
            ic_blocks = [mean_rank_ic(comp[name], yrank, _month_groups(eoms, in_va & np.isin(eoms, b)))
                         for b in blocks]
            n_pos = int(sum(ic > 0 for ic in ic_blocks))
            sp = _decile_net_spread(comp[name], y_w, permno, eoms, in_va, cost_bps)
            reasons = []
            if n_pos < MIN_BLOCKS_POSITIVE:
                reasons.append(f"IC positive in only {n_pos} of {N_BLOCKS} validation sub-periods")
            if not sp["net_spread"] > 0:
                reasons.append(f"decile spread not positive after turnover costs ({sp['net_spread']:+.4%}/month)")
            diag[name] = {"kept": not reasons, "reasons": reasons,
                          "val_ic_by_block": [round(float(x), 4) for x in ic_blocks],
                          "val_ic_mean": round(float(mean_rank_ic(comp[name], yrank, _month_groups(eoms, in_va))), 4),
                          "blocks_positive": n_pos, **{k: round(float(v), 5) for k, v in sp.items()}}
            if not reasons:
                kept.append(name)

        fallback = not kept
        pool = names if fallback else kept
        va_groups = _month_groups(eoms, in_va)
        equal = np.full(len(pool), 1 / len(pool))
        if fallback:                      # nothing passed the stability rule: equal weights, flagged
            w, tried = equal, 1
            val_ic = mean_rank_ic(sum(wi * comp[n] for wi, n in zip(w, pool)), yrank, va_groups)
        else:
            best, tried = (-np.inf, None, np.inf), 0
            for w in _weight_grid(len(pool)):
                blend = sum(wi * comp[n] for wi, n in zip(w, pool))
                ic = mean_rank_ic(blend, yrank, va_groups)
                dist = float(np.abs(w - equal).sum())
                tried += 1
                if ic > best[0] + 1e-9 or (abs(ic - best[0]) <= 1e-9 and dist < best[2]):
                    best = (ic, w, dist)
            val_ic, w, _ = best
        weights = {n: 0.0 for n in names}
        weights.update({n: float(x) for n, x in zip(pool, w)})
        blend_all = sum(weights[n] * comp[n] for n in names)

        fit = np.flatnonzero(in_tr | in_va)                       # score -> return map (2 parameters)
        slope, intercept = np.polyfit(blend_all[fit], y_w[fit], 1)
        te = np.flatnonzero(in_te)
        sig = df.loc[te, ["permno", "eom"]].copy()
        sig["score"] = blend_all[te]
        sig["pred_ret"] = intercept + slope * blend_all[te]
        sig["source"] = SOURCE
        cols = [comp[n][te] for n in names]
        sig["rationale"] = [" | ".join(f"{n} {v:+.2f}" for n, v in zip(names, vals)) for vals in zip(*cols)]
        out.append(sig)

        entries.append({
            "year": year, "train_target_end": str(pd.Timestamp(s["train_end"]).date()),
            "selection_window": "validation", "fit_data_end": str(pd.Timestamp(s["val_end"]).date()),
            "feature_selection_data_end": str(pd.Timestamp(s["val_end"]).date()),
            "feature_set": "composite",
            "characteristics": sorted({c for n in names for c in families[n]}),
            "events": [], "mixtures_tried": tried + len(names),
            "families": {n: {"members": families[n], **diag[n]} for n in names},
            "weights": {k: round(v, 4) for k, v in weights.items()},
            "all_families_failed_stability_equal_weights_used": fallback,
            "cost_bps": cost_bps,
            "val_metric": {"name": "mean_monthly_rank_ic_sector_relative", "blend": round(float(val_ic), 5)},
            "score_to_return": {"slope": float(slope), "intercept": float(intercept)},
        })
        log(f"{year}: kept {kept or 'NONE (equal-weight fallback)'} | weights "
            f"{ {k: round(v, 2) for k, v in weights.items()} } | val IC {val_ic:.3f} | {time.time() - t0:.1f}s")
    signals = pd.concat(out, ignore_index=True)
    return signals, {
        "model": "composite factors (signs fixed from theory), weights chosen on validation, walk-forward",
        "universe": "price >= 5, 126d dollar volume >= 2e6, size_grp in mega/large/small (formation month)",
        "composite": "average of within-month member ranks (sign applied), re-ranked; needs >= half the members",
        "stability_rule": f"IC > 0 in >= {MIN_BLOCKS_POSITIVE} of {N_BLOCKS} validation sub-periods and positive "
                          f"decile spread after turnover costs ({cost_bps:g} bps per dollar traded)",
        "years": entries}


def run(panel_path, factor_path, out_dir, families=None, years=range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1),
        cost_bps=DEFAULT_COST_BPS):
    import json
    from pathlib import Path

    from .model import load_panel
    families = families or FAMILIES
    factors = pd.read_csv(factor_path)["variable"].tolist()
    check_families(families, factors)
    df = load_panel(panel_path, columns_needed(families), extra=("gics",))
    signals, manifest = build_composite_signals(df, families, years, cost_bps)
    manifest["oos_report"] = test_report(signals, df, manifest)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    signals[["permno", "eom", "score", "source", "pred_ret", "rationale"]].to_parquet(out / "signals.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return signals, manifest


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--factors", default="data/factor_char_list.csv")
    ap.add_argument("--out", default="out_composite")
    ap.add_argument("--extra", nargs="*", default=[], choices=sorted(EXTRA_FAMILIES),
                    help="also include these optional families")
    ap.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    a = ap.parse_args()
    fams = {**FAMILIES, **{k: EXTRA_FAMILIES[k] for k in a.extra}}
    _, m = run(a.panel, a.factors, a.out, fams, cost_bps=a.cost_bps)
    print(json.dumps(m["oos_report"], indent=1))
