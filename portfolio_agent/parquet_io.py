"""Parquet reading that survives old pyarrow.

The supplied data files were written by a newer parquet writer. pyarrow 19 and older cannot read
them ("OSError: Repetition level histogram size mismatch"), and on a date-filtered read of the 8-K
file it can abort the whole Python process. pyarrow 25 (pinned in requirements.txt) reads them fine.

read_parquet() therefore:
  * pyarrow >= 25: reads normally (date filters pushed down to the file).
  * older pyarrow: reads with fastparquet instead when it is installed (it is in requirements.txt),
    applying the date filter exactly in pandas afterwards;
  * older pyarrow and no fastparquet: reads without pushing the filter into pyarrow (that path can
    abort the process) and filters in pandas; if the known error appears, stops with a plain-English
    message saying what to install instead of an opaque C++ error.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

KNOWN_ERROR = "Repetition level histogram"
MIN_GOOD_PYARROW = 25


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _pyarrow_major() -> int | None:
    try:
        import pyarrow
        return int(pyarrow.__version__.split(".")[0])
    except Exception:
        return None


def _help(path) -> str:
    major = _pyarrow_major()
    return (f"Cannot read {path} with pyarrow {major if major is not None else '(not installed)'}: pyarrow older than "
            f"{MIN_GOOD_PYARROW} fails on these files ('Repetition level histogram size mismatch').\n"
            f"Fix, either:  pip install -U 'pyarrow>={MIN_GOOD_PYARROW}'   (what requirements.txt pins)\n"
            "         or:  pip install fastparquet                (used automatically as a fallback)")


def _read(path, columns, between, engine, push_filters=True):
    kwargs = {}
    cols = list(columns) if columns is not None else None
    added = False
    if between is not None:
        col, lo, hi = between
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        if cols is not None and col not in cols:
            cols.append(col)
            added = True
        if engine == "pyarrow" and push_filters:
            kwargs["filters"] = [(col, ">=", lo.date()), (col, "<=", hi.date())]
        elif engine == "fastparquet":                   # fastparquet compares against datetime64 statistics
            kwargs["filters"] = [(col, ">=", np.datetime64(lo)), (col, "<=", np.datetime64(hi))]
    df = pd.read_parquet(path, columns=cols, engine=engine, **kwargs)
    if between is not None:                             # exact: row-group filtering is only coarse
        d = pd.to_datetime(df[col])
        df = df[(d >= lo) & (d <= hi)]
        if added:
            df = df.drop(columns=col)
    return df


def read_parquet(path, columns=None, between=None) -> pd.DataFrame:
    """pd.read_parquet with the fallbacks above. between=(column, first_date, last_date), inclusive."""
    major = _pyarrow_major()
    old = major is None or major < MIN_GOOD_PYARROW
    fast = _have("fastparquet")
    if old and between is not None and fast:
        return _read(path, columns, between, "fastparquet")
    try:
        # old pyarrow: do not push the filter down (that path can abort the process); filter in pandas
        return _read(path, columns, between, "pyarrow", push_filters=not old)
    except OSError as e:
        if KNOWN_ERROR not in str(e):
            raise
        if not fast:
            raise RuntimeError(_help(path)) from e
        return _read(path, columns, between, "fastparquet")
