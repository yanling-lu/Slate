"""Benchmark and market series from FRED (public CSV endpoint, no API key).

TB3MS: 3-month T-bill secondary market rate, annualised percent, monthly.
SP500: S&P 500 index level, daily, PRICE index (excludes dividends).
Files are cached in the output folder so a report can be rebuilt offline.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd={start}"


def fetch_fred(series_id: str, cache: Path, start: str = "2020-06-01", refresh: bool = True) -> Path:
    """Download a FRED series to `cache`; fall back to an existing cached copy if offline."""
    cache = Path(cache)
    if refresh or not cache.exists():
        try:
            req = urllib.request.Request(URL.format(id=series_id, start=start),
                                         headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=30).read()
            if not data.startswith(b"observation_date") and not data.startswith(b"DATE"):
                raise ValueError(f"unexpected FRED response for {series_id}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
        except Exception as e:
            if not cache.exists():
                raise RuntimeError(
                    f"could not download FRED {series_id} ({e}); pass --{series_id.lower()} "
                    "with a downloaded csv") from e
    return cache


def _read(path) -> pd.Series:
    df = pd.read_csv(path)
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    s.index = pd.to_datetime(df.iloc[:, 0])
    return s.dropna()


def tb3ms_monthly(path) -> pd.Series:
    """Annualised percent, indexed by month-end of the observation month."""
    s = _read(path)
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s[~s.index.duplicated(keep="last")]


def sp500_monthly_return(path) -> pd.Series:
    """Month-end to month-end price return, indexed by month-end."""
    s = _read(path)
    eom = s.index + pd.offsets.MonthEnd(0)
    level = s.groupby(eom).last()
    return level.pct_change().dropna()
