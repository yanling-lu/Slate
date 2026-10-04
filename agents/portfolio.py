"""Pipeline-facing portfolio agent.

Adapts the deterministic engine in `portfolio_agent/` to the interfaces the rest of
the pipeline uses: it takes the validated Signal batch and the Devil's Advocate
challenges from the graph, and returns a `PortfolioTarget`-shaped dict.

    run(validated_signals: dict, challenges: list, as_of_date: str) -> dict

How the pieces combine
----------------------
* The LLM Signal Agent is event-driven and sparse (only stocks with a filing that
  month). A 100-500 name market-neutral book needs a dense cross-section, so the
  agent combines two sources: a dense base signal (the characteristics model, loaded
  from BASE_SIGNALS_PATH) and the sparse `signal_agent` source.
* Devil's Advocate challenges become per-stock vetoes / haircuts:
  reject -> veto, revise -> haircut (severity = 1 - resilience_score, 0.25-0.9),
  proceed -> no change. A numeric `objection_score` 0-3 is also understood.
* The agent makes no LLM calls. `model` is accepted for interface compatibility only.
* Month-by-month state: the graph is invoked per stock, so signals for the same month
  accumulate across calls and each call returns the book built from everything seen so
  far. The month is committed (becomes the "previous book" for turnover) when a later
  month is requested. Months must be requested in order with no gaps.

Environment: CHARS_PATH (panel, default data/chars_final_with_names.parquet),
BASE_SIGNALS_PATH (dense signals csv/parquet), PORTFOLIO_DEMO_BASELINE=1 to use the
fixed momentum/quality/value placeholder instead (labelled, not the strategy).
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from portfolio_agent.agent import PortfolioConstructionAgent  # noqa: F401  (re-export)
from portfolio_agent.agent import Rebalance
from portfolio_agent.config import Mandate, Policy
from portfolio_agent.contracts import (OBJECTION_MAP, ContractError, read_table, validate_da,
                                       validate_signals)
from portfolio_agent.data import load_panel, prepare_panel

LLM_SOURCE = "signal_agent"
SIG_COLS = ["permno", "eom", "source", "score", "confidence", "rationale", "asof_date"]
DA_COLS = ["permno", "eom", "action", "severity", "side", "reason"]
ACTION_RANK = {"pass": 0, "haircut": 1, "veto": 2}


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_eom(as_of_date) -> pd.Timestamp:
    """Formation month-end. Rejects mid-month dates: characteristics are month-end
    snapshots, so a mid-month cutoff would let later information in."""
    ts = pd.Timestamp(as_of_date).normalize()
    if ts.day < 28:
        raise ContractError(
            f"as_of_date {ts.date()} is mid-month; the portfolio forms on month-end "
            "information (use the last trading day of the month)")
    return ts + MonthEnd(0)


def signals_to_frame(batch, eom: pd.Timestamp) -> pd.DataFrame:
    """Signal batch -> one row per permno for source `signal_agent`.

    Neutral and insufficient-evidence signals are dropped (no row = no view). Several
    signals for one stock are merged: score = confidence-weighted mean, confidence = max.
    The model's own `ticker` is ignored; labels come from the panel."""
    batch_asof = _get(batch, "as_of_date")
    if batch_asof is not None and pd.Timestamp(batch_asof) > eom:
        raise ContractError(f"signal batch dated {batch_asof} is after formation month-end {eom.date()}")
    rows = []
    for s in _get(batch, "signals", None) or []:
        if _get(s, "insufficient_evidence", False) or _get(s, "direction") == "neutral":
            continue
        asof = pd.Timestamp(_get(s, "as_of_date", eom))
        if asof > eom:
            raise ContractError(f"signal for permno {_get(s, 'permno')} is dated {asof.date()}, "
                                f"after formation month-end {eom.date()}")
        catalyst, why = _get(s, "catalyst", ""), _get(s, "rationale", "")
        rows.append({"permno": int(_get(s, "permno")), "score": float(_get(s, "score")),
                     "confidence": float(_get(s, "confidence", 1.0)),
                     "rationale": f"{catalyst}: {why}".strip(": "), "asof_date": asof})
    if not rows:
        return pd.DataFrame(columns=SIG_COLS)
    merged = []
    for permno, g in pd.DataFrame(rows).groupby("permno"):
        w = g["confidence"].sum()
        score = (g["score"] * g["confidence"]).sum() / w if w > 0 else g["score"].mean()
        merged.append({"permno": permno, "score": score, "confidence": g["confidence"].max(),
                       "rationale": " | ".join(g["rationale"])[:300],
                       "asof_date": g["asof_date"].max()})
    out = pd.DataFrame(merged)
    out["eom"], out["source"] = eom, LLM_SOURCE
    return out[SIG_COLS]


def challenges_to_da(challenges, batch, eom: pd.Timestamp) -> pd.DataFrame:
    """Devil's Advocate challenges -> the portfolio agent's objection table."""
    side_of = {}
    for s in _get(batch, "signals", None) or []:
        d = _get(s, "direction")
        p = int(_get(s, "permno"))
        side = d if d in ("long", "short") else "both"
        side_of[p] = side if side_of.get(p, side) == side else "both"
    best: dict[int, dict] = {}
    for c in challenges or []:
        permno = int(_get(c, "permno"))
        score = _get(c, "objection_score")
        if score is not None:
            if int(score) not in OBJECTION_MAP:
                raise ContractError(f"objection_score must be 0-3, got {score}")
            action, severity = OBJECTION_MAP[int(score)]
        else:
            rec = _get(c, "recommendation", "proceed")
            if rec == "reject":
                action, severity = "veto", 1.0
            elif rec == "revise":
                res = float(_get(c, "resilience_score", 0.5))
                action, severity = "haircut", float(np.clip(1.0 - res, 0.25, 0.9))
            else:
                action, severity = "pass", 0.0
        claims = "; ".join(str(_get(x, "claim", "")) for x in (_get(c, "counterarguments", []) or []))
        row = {"permno": permno, "eom": eom, "action": action, "severity": severity,
               "side": side_of.get(permno, "both"),
               "reason": (f"{_get(c, 'recommendation', '')}: {claims}".strip(": "))[:300]}
        old = best.get(permno)
        if old is None or (ACTION_RANK[action], severity) >= (ACTION_RANK[old["action"]], old["severity"]):
            best[permno] = row                       # most severe objection wins; later calls win ties
    return pd.DataFrame(list(best.values()), columns=DA_COLS)


def _clean(v):
    if isinstance(v, (pd.Timestamp,)):
        return v.date().isoformat()
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


class PortfolioReconstructionAgent:
    def __init__(self, model: str | None = None, panel=None, base_signals=None,
                 mandate: Mandate | None = None, policy: Policy | None = None):
        self.model = model                      # unused: this agent is deterministic
        self.mandate = mandate or Mandate()
        pol = policy or Policy()
        fixed = {LLM_SOURCE: 1.0, **pol.fixed_scale_sources}
        self.policy = replace(pol, fixed_scale_sources=fixed)
        self._panel_arg, self._base_arg = panel, base_signals
        self._engine = None
        self._base_by_month: dict | None = None
        self._cur: pd.Timestamp | None = None        # month being built
        self._prev: pd.Series | None = None          # committed book from the previous month
        self._prev_eom: pd.Timestamp | None = None
        self._last: Rebalance | None = None          # result for the current month
        self._sig: dict[int, dict] = {}
        self._da: dict[int, dict] = {}
        self._history: list[tuple[pd.Timestamp, Rebalance]] = []

    # ------------------------------------------------------------------ setup
    def _load(self):
        if self._engine is not None:
            return
        panel = self._panel_arg
        chars_path = os.environ.get("CHARS_PATH", "data/chars_final_with_names.parquet")
        if panel is None:
            panel = load_panel(chars_path)
        elif isinstance(panel, (str, os.PathLike)):
            panel = load_panel(panel)
        elif "sector" not in panel.columns:
            panel = prepare_panel(panel)
        self._engine = PortfolioConstructionAgent(panel, self.mandate, self.policy)

        base = self._base_arg
        if base is None and os.environ.get("BASE_SIGNALS_PATH"):
            base = read_table(os.environ["BASE_SIGNALS_PATH"])
        if base is None and os.environ.get("PORTFOLIO_DEMO_BASELINE", "").lower() in ("1", "true", "yes"):
            from portfolio_agent.run import demo_signals
            base = demo_signals(chars_path)
        elif isinstance(base, (str, os.PathLike)):
            base = read_table(base)
        if base is not None:
            base = validate_signals(base)
            if (base["source"] == LLM_SOURCE).any():
                raise ContractError(f"base signals may not use the reserved source '{LLM_SOURCE}'")
            self._base_by_month = {t: g for t, g in base.groupby("eom")}

    # ------------------------------------------------------------- month state
    def _roll(self, eom: pd.Timestamp):
        if self._cur is None:
            self._cur = eom
            return
        if eom == self._cur:
            return
        if eom < self._cur:
            raise ContractError(f"months must be processed in order: {eom.date()} after {self._cur.date()}")
        if self._last is not None:                   # commit the finished month
            self._history.append((self._cur, self._last))
            self._prev, self._prev_eom = self._last.weights, self._cur
        if self._prev_eom is not None and eom != self._prev_eom + MonthEnd(1):
            raise ContractError(f"gap in months: last completed {self._prev_eom.date()}, "
                                f"requested {eom.date()}; turnover needs consecutive months")
        self._cur, self._last, self._sig, self._da = eom, None, {}, {}

    # ------------------------------------------------------------------ public
    def run(self, validated_signals, challenges, as_of_date) -> dict:
        eom = to_eom(as_of_date)
        self._roll(eom)
        for _, r in signals_to_frame(validated_signals, eom).iterrows():
            self._sig[int(r["permno"])] = r.to_dict()          # a revision replaces the earlier signal
        for _, r in challenges_to_da(challenges, validated_signals, eom).iterrows():
            self._da[int(r["permno"])] = r.to_dict()
        return self.target(as_of_date)

    def target(self, as_of_date) -> dict:
        """Build the book for this month from the base signal plus everything added so far."""
        self._load()
        eom = to_eom(as_of_date)
        self._roll(eom)
        parts = []
        if self._base_by_month is not None and eom in self._base_by_month:
            parts.append(self._base_by_month[eom])
        else:
            raise ContractError(
                f"no dense base signal for {eom.date()}: set BASE_SIGNALS_PATH (characteristics "
                "model output) or PORTFOLIO_DEMO_BASELINE=1; LLM signals alone cannot fill a "
                "100-500 name book")
        if self._sig:
            parts.append(validate_signals(pd.DataFrame(list(self._sig.values()))[SIG_COLS]))
        sig = pd.concat(parts, ignore_index=True)
        da = validate_da(pd.DataFrame(list(self._da.values()), columns=DA_COLS))
        self._last = self._engine.rebalance(eom, sig, da, self._prev)
        return self._payload(eom, self._last)

    # --------------------------------------------------------------- reporting
    def _payload(self, eom: pd.Timestamp, r: Rebalance) -> dict:
        rows = self._engine.by_month[eom]
        w = r.weights

        def ticker(p):
            t = rows.at[p, "ticker_ff"]
            return str(t) if pd.notna(t) else f"PERMNO{p}"

        d = r.diagnostics
        reasons = []
        if d["relax_level"] > 0:
            reasons.append(f"soft limits relaxed: {d['relaxations']}")
        if d["policy_breaches"]:
            reasons.append(f"policy limits breached: {d['policy_breaches']}")
        if d["beta_imputed_weight"] > 0.25 * d["gross"]:
            reasons.append("over 25% of gross sits in names with imputed beta")
        return {
            "as_of_date": eom.date().isoformat(),
            "holdings_date": (eom + pd.Timedelta(days=1)).date().isoformat(),
            "target_weights": [{"permno": int(p), "ticker": ticker(p), "weight": float(x)}
                               for p, x in w.items()],
            "constraint_report": {
                "checks": [{k: _clean(v) for k, v in c.as_dict().items()} for c in r.checks],
                "diagnostics": {k: _clean(v) for k, v in d.items() if not k.startswith("chk_")},
            },
            "human_approval_required": bool(reasons),
            "approval_reasons": reasons,
            "trade_rationale": [{k: _clean(v) for k, v in row.items()}
                                for row in r.trades.drop(columns="date").to_dict("records")],
            "violations": [],                   # filled by the graph's policy node
            "sectors": {int(p): str(rows.at[p, "sector"]) for p in w.index},
        }

    def _all(self):
        done = list(self._history)
        if self._last is not None:
            done.append((self._cur, self._last))
        return done

    def holdings_frame(self) -> pd.DataFrame:
        """Submission-format holdings for every month built so far."""
        out = []
        for eom, r in self._all():
            rows = self._engine.by_month[eom].loc[r.weights.index]
            out.append(pd.DataFrame({
                "Date": eom + pd.Timedelta(days=1), "PERMNO": r.weights.index,
                "TICKER": rows["ticker_ff"].values, "COMPANY NAME": rows["company_name_ff"].values,
                "WEIGHT": r.weights.values, "eom": eom,
                "label_status": np.where(rows["ticker"].notna(), "verified_in_month",
                                         np.where(rows["ticker_ff"].notna(), "carried_forward",
                                                  "unlabeled"))}))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    def monthly_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.diagnostics for _, r in self._all()])

    def trades_frame(self) -> pd.DataFrame:
        return pd.concat([r.trades for _, r in self._all()], ignore_index=True) if self._all() else pd.DataFrame()
