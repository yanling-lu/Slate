"""Panel loading. Formation data and outcomes are loaded by separate functions so
the construction agent never holds a forward return."""
from __future__ import annotations

import pandas as pd
from pandas.tseries.offsets import MonthEnd

from .parquet_io import read_parquet

# Everything here is known at the end of formation month t.
FORMATION_COLS = [
    "permno", "eom", "ticker", "company_name", "me", "prc", "dolvol_126d",
    "size_grp", "gics", "beta_60m", "ret", "ret_exc",
]
OUTCOME_COL = "ret_exc_lead1m"


def month_end(s) -> pd.Series:
    return pd.to_datetime(s) + MonthEnd(0)


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["eom"] = month_end(df["eom"])
    df["permno"] = df["permno"].astype("int64")
    gics = df["gics"].astype("string")
    df["sector"] = gics.str.slice(0, 2).fillna("NA").astype(object)
    df = df.sort_values(["permno", "eom"]).reset_index(drop=True)
    # Labels are carried forward only (never back-filled from a later name).
    for col in ("ticker", "company_name"):
        df[col + "_ff"] = df.groupby("permno")[col].ffill()
    return df


def load_panel(path) -> pd.DataFrame:
    """Formation-safe columns only."""
    return prepare_panel(read_parquet(path, columns=FORMATION_COLS))


def load_outcomes(path) -> pd.DataFrame:
    """Forward returns, for the backtest only. Keyed by formation month."""
    df = read_parquet(path, columns=["permno", "eom", "me", OUTCOME_COL])
    df["eom"] = month_end(df["eom"])
    df["permno"] = df["permno"].astype("int64")
    return df


def rf_by_month(panel: pd.DataFrame) -> pd.Series:
    """Risk-free return the data pipeline used for each calendar month."""
    return (panel["ret"] - panel["ret_exc"]).groupby(panel["eom"]).median()
