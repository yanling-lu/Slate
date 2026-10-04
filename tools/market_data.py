from __future__ import annotations
from datetime import date
from functools import lru_cache
import os
from config import settings
import pandas as pd


@lru_cache(maxsize=1)
def get_tb3ms_monthly() -> pd.Series:
    """FRED TB3MS, annualized percent. Download once, cache to disk."""
    cache = "data/tb3ms.csv"
    if not os.path.exists(cache):
        from fredapi import Fred
        fred = Fred(api_key=settings.fred_api_key())
        s = fred.get_series("TB3MS")
        s.to_csv(cache)
    s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    s.index = s.index.to_period("M")
    return s


def benchmark_return_series() -> pd.Series:
    """Monthly benchmark = TB3MS/100/12 + 0.04/12. Decimal units."""
    tb = get_tb3ms_monthly() / 100.0 / 12.0
    return tb + 0.04 / 12.0


@lru_cache(maxsize=1)
def get_sp500_monthly() -> pd.Series:
    cache = "data/sp500.csv"
    if not os.path.exists(cache):
        from fredapi import Fred
        fred = Fred(api_key=settings.fred_api_key())
        s = fred.get_series("SP500")
        s.to_csv(cache)
    s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    s.index = s.index.to_period("M")
    return s.pct_change().dropna()