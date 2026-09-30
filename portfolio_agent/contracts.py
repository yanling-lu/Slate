"""Input contracts for the upstream agents, with leakage checks.

signals (signal / analyst agents), one row per stock-month-source:
    permno      int
    eom         formation month-end t; the score may use information up to t only
    score       float, higher = more bullish for month t+1 (any scale)
    source      optional str, default "signal"
    confidence  optional float in [0, 1], default 1
    rationale   optional str, copied into the trade log
    asof_date   optional date of the latest input used; must be <= eom

devil's advocate, one row per stock-month it objects to:
    permno, eom
    action      "veto" | "haircut" | "pass"
    severity    float in [0, 1], used by "haircut" (alpha and cap scaled by 1 - severity)
    side        optional "long" | "short" | "both" - which thesis is attacked
    reason      optional str, copied into the trade log
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import month_end

FORBIDDEN = ("lead", "fwd", "forward", "future", "next_month", "target")
DA_ACTIONS = {"veto", "haircut", "pass"}
DA_SIDES = {"long", "short", "both"}


class ContractError(ValueError):
    pass


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _common(df: pd.DataFrame, required, name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ContractError(f"{name}: missing columns {missing}")
    leaked = [c for c in df.columns if any(k in c for k in FORBIDDEN)]
    if leaked:
        raise ContractError(f"{name}: forward-looking columns not allowed: {leaked}")
    df["permno"] = df["permno"].astype("int64")
    df["eom"] = month_end(df["eom"])
    if "asof_date" in df.columns:
        late = pd.to_datetime(df["asof_date"]) > df["eom"]
        if late.any():
            raise ContractError(
                f"{name}: {int(late.sum())} rows use information dated after the "
                "formation month-end (asof_date > eom)"
            )
    return df


def validate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = _common(df, ["permno", "eom", "score"], "signals")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df[np.isfinite(df["score"])]
    if "source" not in df.columns:
        df["source"] = "signal"
    df["source"] = df["source"].fillna("signal").astype(str)
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    df["confidence"] = df["confidence"].fillna(1.0).astype(float)
    if not df["confidence"].between(0, 1).all():
        raise ContractError("signals: confidence must be in [0, 1]")
    if "rationale" not in df.columns:
        df["rationale"] = ""
    df["rationale"] = df["rationale"].fillna("").astype(str)
    if df.duplicated(["permno", "eom", "source"]).any():
        raise ContractError("signals: duplicate (permno, eom, source) rows")
    return df[["permno", "eom", "source", "score", "confidence", "rationale"]]


OBJECTION_MAP = {  # objection_score -> (action, severity)
    0: ("pass", 0.0), 1: ("pass", 0.0), 2: ("haircut", 0.5), 3: ("veto", 1.0),
}


def from_objection_score(df: pd.DataFrame) -> pd.DataFrame:
    """Adapter for the devil's-advocate agent's 0-3 objection score:
    3 = drop the name, 2 = halve the bet, 0-1 = keep."""
    df = df.copy()
    score = pd.to_numeric(df["objection_score"], errors="coerce")
    if score.isna().any() or not score.isin(list(OBJECTION_MAP)).all():
        raise ContractError("devils_advocate: objection_score must be an integer 0-3")
    score = score.astype(int)
    df["action"] = score.map(lambda s: OBJECTION_MAP[s][0])
    df["severity"] = score.map(lambda s: OBJECTION_MAP[s][1])
    return df.drop(columns="objection_score")


def validate_da(df: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["permno", "eom", "action", "severity", "side", "reason"]
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    df = _common(df, ["permno", "eom"], "devils_advocate")
    if "objection_score" in df.columns:
        if "action" in df.columns:
            raise ContractError("devils_advocate: give objection_score or action, not both")
        df = from_objection_score(df)
    elif "action" not in df.columns:
        raise ContractError("devils_advocate: need an 'action' or 'objection_score' column")
    df["action"] = df["action"].astype(str).str.lower()
    if not set(df["action"]) <= DA_ACTIONS:
        raise ContractError(f"devils_advocate: action must be one of {sorted(DA_ACTIONS)}")
    if "side" not in df.columns:
        df["side"] = "both"
    df["side"] = df["side"].fillna("both").astype(str).str.lower()
    if not set(df["side"]) <= DA_SIDES:
        raise ContractError(f"devils_advocate: side must be one of {sorted(DA_SIDES)}")
    if "severity" not in df.columns:
        df["severity"] = 0.5
    df["severity"] = df["severity"].fillna(0.5).astype(float)
    if not df["severity"].between(0, 1).all():
        raise ContractError("devils_advocate: severity must be in [0, 1]")
    if "reason" not in df.columns:
        df["reason"] = ""
    df["reason"] = df["reason"].fillna("").astype(str)
    if df.duplicated(["permno", "eom"]).any():
        raise ContractError("devils_advocate: duplicate (permno, eom) rows")
    return df[cols]
