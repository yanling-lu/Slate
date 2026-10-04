"""Dense base signal: a score for every stock-month from the 147 characteristics.

Walk-forward, one refit per forecast year (schedule in portfolio_agent/splits.py):

  1. Preprocess: each characteristic is rank-transformed within its month to
     [-0.5, 0.5]; missing -> 0 (the median rank). Uses only that month's cross-section.
  2. Stage 1 (Signals-agent role): from the 147, keep the 20 with the largest |mean
     monthly rank IC| against next-month return, on TRAINING months only.
  3. Stage 2 (Devil's-advocate role): cut the 20 to 7, keeping features whose IC has the
     same sign on the VALIDATION window as in training, ranked by |validation IC|.
  4. Mixtures: fit a ridge on every subset of those 7 with at least 3 features on the
     training months; pick the subset with the best mean monthly rank IC on VALIDATION.
  5. Refit the chosen subset on training + validation, score the test year.

Universe: fitting and scoring use only stocks passing the portfolio's own screens
(price, dollar volume, size group; formation-month information only). Fitting on micro and
nano caps inflates the IC (about -0.15 for volatility features across all stocks vs
about -0.05 in the tradable set) with names the portfolio can never hold.

Nothing from the test year's returns is used to choose anything. The target is
ret_exc_lead1m, clipped at the 1st/99th percentile of the fitting rows (fitted
on those rows only). The output score is the predicted return, `pred_ret`.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from scipy.stats import rankdata

from portfolio_agent.config import Policy
from portfolio_agent.splits import FIRST_TEST_YEAR, LAST_TEST_YEAR, split_for_year

TARGET = "ret_exc_lead1m"
SOURCE = "chars_model"
N_STAGE1, N_STAGE2, MIN_MIX = 20, 7, 3
RIDGE = 1e-3          # relative to the number of rows; fixed, not tuned


SCREEN_COLS = ["prc", "dolvol_126d", "size_grp"]


def tradable(df: pd.DataFrame, policy: Policy | None = None) -> pd.Series:
    """The portfolio agent's long-side screen, from formation-month columns only."""
    p = policy or Policy()
    return ((df["prc"] >= p.min_price) & (df["dolvol_126d"] >= p.min_dolvol)
            & df["size_grp"].isin(p.size_groups))


def load_panel(path, factors) -> pd.DataFrame:
    cols = list(dict.fromkeys(["permno", "eom"] + list(factors) + SCREEN_COLS + [TARGET]))
    df = pd.read_parquet(path, columns=cols)
    df["eom"] = pd.to_datetime(df["eom"]) + MonthEnd(0)
    df["permno"] = df["permno"].astype("int64")
    return df.sort_values(["eom", "permno"]).reset_index(drop=True)


def rank_features(df: pd.DataFrame, factors) -> np.ndarray:
    """Within-month rank in [-0.5, 0.5]; missing -> 0. Row-aligned with df."""
    r = df.groupby("eom")[list(factors)].rank(pct=True) - 0.5
    return r.fillna(0.0).to_numpy(dtype=np.float64)


def _month_groups(eoms: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    order = idx[np.argsort(eoms[idx], kind="stable")]
    cut = np.flatnonzero(np.diff(eoms[order].astype("int64")) != 0) + 1
    return [g for g in np.split(order, cut) if g.size > 30]


def feature_ic(X, yrank, groups) -> np.ndarray:
    """Mean over months of corr(feature rank, target rank): shape (n_features,)."""
    out = []
    for g in groups:
        x, y = X[g], yrank[g]
        x = x - x.mean(0)
        y = y - y.mean()
        den = np.sqrt((x ** 2).sum(0) * (y ** 2).sum())
        out.append(np.divide(x.T @ y, den, out=np.zeros(x.shape[1]), where=den > 0))
    return np.mean(out, axis=0) if out else np.zeros(X.shape[1])


def mean_rank_ic(pred, yrank, groups) -> float:
    ics = []
    for g in groups:
        p = rankdata(pred[g]) - (g.size + 1) / 2
        y = yrank[g] - yrank[g].mean()
        den = np.sqrt((p ** 2).sum() * (y ** 2).sum())
        if den > 0:
            ics.append((p @ y) / den)
    return float(np.mean(ics)) if ics else np.nan


def fit_ridge(X, y):
    xm, ym = X.mean(0), y.mean()
    xc = X - xm
    beta = np.linalg.solve(xc.T @ xc + RIDGE * len(y) * np.eye(X.shape[1]), xc.T @ (y - ym))
    return beta, ym - xm @ beta


def winsorize(y, lo_hi=(1, 99)):
    lo, hi = np.percentile(y, lo_hi)
    return np.clip(y, lo, hi)


def build_signals(df: pd.DataFrame, factors, years=range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1)):
    """df: permno, eom, the factors, ret_exc_lead1m. Returns (signals, manifest)."""
    factors = list(factors)
    df = df[tradable(df).to_numpy()].reset_index(drop=True)
    eoms = df["eom"].to_numpy()
    tgt_month = (df["eom"] + MonthEnd(1)).to_numpy()
    y_raw = df[TARGET].to_numpy(dtype=np.float64)
    has_y = ~np.isnan(y_raw)
    X = rank_features(df, factors)
    yrank = (df.groupby("eom")[TARGET].rank(pct=True) - 0.5).to_numpy()
    yrank = np.nan_to_num(yrank)

    out, entries = [], []
    for year in years:
        s = {k: np.datetime64(v) for k, v in split_for_year(year).items()}
        in_train = has_y & (tgt_month >= s["train_start"]) & (tgt_month <= s["train_end"])
        in_val = has_y & (tgt_month >= s["val_start"]) & (tgt_month <= s["val_end"])
        in_test = (tgt_month >= s["test_start"]) & (tgt_month <= s["test_end"])
        if not in_test.any():
            continue
        g_tr, g_va = _month_groups(eoms, in_train), _month_groups(eoms, in_val)
        if not g_tr or not g_va:
            continue

        ic_tr = feature_ic(X, yrank, g_tr)
        stage1 = np.argsort(-np.abs(ic_tr))[:N_STAGE1]
        ic_va = feature_ic(X, yrank, g_va)
        stable = np.where(np.sign(ic_va[stage1]) == np.sign(ic_tr[stage1]), np.abs(ic_va[stage1]), 0.0)
        top = stage1[np.argsort(-stable, kind="stable")[:N_STAGE2]]

        tr, va = np.flatnonzero(in_train), np.flatnonzero(in_val)
        y_tr = winsorize(y_raw[tr])
        best, tried = (-np.inf, None), 0
        for k in range(MIN_MIX, len(top) + 1):
            for sub in itertools.combinations(range(len(top)), k):
                cols = top[list(sub)]
                beta, b0 = fit_ridge(X[np.ix_(tr, cols)], y_tr)
                pred = np.full(len(df), np.nan)
                pred[va] = X[np.ix_(va, cols)] @ beta + b0
                ic = mean_rank_ic(pred, yrank, g_va)
                tried += 1
                if ic > best[0] + 1e-12 or (abs(ic - best[0]) <= 1e-12 and best[1] is not None
                                            and len(cols) > len(best[1])):
                    best = (ic, cols)
        val_ic, cols = best

        fit = np.flatnonzero(in_train | in_val)                     # refit on train + validation
        beta, b0 = fit_ridge(X[np.ix_(fit, cols)], winsorize(y_raw[fit]))
        te = np.flatnonzero(in_test)
        sig = df.loc[te, ["permno", "eom"]].copy()
        sig["score"] = X[np.ix_(te, cols)] @ beta + b0
        sig["pred_ret"] = sig["score"]
        sig["source"] = SOURCE
        sig["rationale"] = ""
        out.append(sig)

        names = [factors[c] for c in cols]
        coef = {n: round(float(b), 6) for n, b in zip(names, beta)}
        entries.append({
            "year": year,
            "train_target_end": str(pd.Timestamp(s["train_end"]).date()),
            "selection_window": "validation",
            "fit_data_end": str(pd.Timestamp(s["val_end"]).date()),
            "feature_selection_data_end": str(pd.Timestamp(s["val_end"]).date()),
            "characteristics": names, "events": [],
            "shortlist_7_ranked": [factors[c] for c in top],
            "shortlist_20": [factors[c] for c in stage1],
            "mixtures_tried": tried,
            "val_metric": {"name": "mean_monthly_rank_ic", "value": round(val_ic, 5)},
            "coefficients": coef,
        })
    signals = pd.concat(out, ignore_index=True)
    return signals, {"model": "ridge on rank-transformed characteristics, walk-forward",
                     "universe": "price >= 5, 126d dollar volume >= 2e6, size_grp in mega/large/small (formation month)",
                     "preprocessing": "within-month rank to [-0.5,0.5]; missing=0; target clipped 1/99 pct of fitting rows",
                     "years": entries}


def test_report(signals: pd.DataFrame, df: pd.DataFrame, manifest: dict) -> dict:
    """Out-of-sample diagnostics. Reporting only: nothing here feeds back into selection."""
    m = signals.merge(df[["permno", "eom", TARGET]], on=["permno", "eom"]).dropna(subset=[TARGET])
    m["tyear"] = (m["eom"] + MonthEnd(1)).dt.year
    yrank = m.groupby("eom")[TARGET].rank(pct=True) - 0.5
    m["yrank"] = yrank
    rep = {}
    for year, g in m.groupby("tyear"):
        groups = [np.flatnonzero(g["eom"].to_numpy() == e) for e in g["eom"].unique()]
        ic = mean_rank_ic(g["score"].to_numpy(), g["yrank"].to_numpy(), groups)
        r, p = g[TARGET].to_numpy(), g["pred_ret"].to_numpy()
        rep[int(year)] = {"rank_ic": round(ic, 4), "oos_r2": round(1 - ((r - p) ** 2).sum() / (r ** 2).sum(), 5),
                          "stock_months": int(len(g))}
        for e in manifest["years"]:
            if e["year"] == year:
                e["test_rank_ic"], e["oos_r2_test"] = rep[int(year)]["rank_ic"], rep[int(year)]["oos_r2"]
    r, p = m[TARGET].to_numpy(), m["pred_ret"].to_numpy()
    groups = [np.flatnonzero(m["eom"].to_numpy() == e) for e in m["eom"].unique()]
    rep["all"] = {"rank_ic": round(mean_rank_ic(m["score"].to_numpy(), m["yrank"].to_numpy(), groups), 4),
                  "oos_r2": round(1 - ((r - p) ** 2).sum() / (r ** 2).sum(), 5), "stock_months": int(len(m))}
    return rep


def run(panel_path, factor_path, out_dir) -> tuple[pd.DataFrame, dict]:
    factors = pd.read_csv(factor_path)["variable"].tolist()
    df = load_panel(panel_path, factors)
    signals, manifest = build_signals(df, factors)
    report = test_report(signals, df, manifest)
    manifest["oos_report"] = report
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    signals[["permno", "eom", "score", "source", "pred_ret", "rationale"]].to_parquet(out / "signals.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return signals, manifest


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--factors", default="data/factor_char_list.csv")
    ap.add_argument("--out", default="out_base")
    a = ap.parse_args()
    _, m = run(a.panel, a.factors, a.out)
    print(json.dumps(m["oos_report"], indent=1))
