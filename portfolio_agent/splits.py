"""The challenge's walk-forward schedule, as dates of the TARGET (holding) month.

A formation row in month t has target month t+1, so a January 2021 forecast uses
December 2020 characteristics. Labels whose target month falls in the test year are
never used for fitting, even if their characteristic month is earlier.

    forecast year 2021: train 2015-02..2018-12, validate 2019-01..2020-12, test 2021
    forecast year 2022: train 2015-02..2019-12, validate 2020-01..2021-12, test 2022
    ...refit annually through 2026 (test months 2026-01..2026-08)
"""
from __future__ import annotations

import pandas as pd

FIRST_TARGET = pd.Timestamp("2015-02-28")      # first target month in the panel
LAST_TARGET = pd.Timestamp("2026-08-31")
FIRST_TEST_YEAR, LAST_TEST_YEAR = 2021, 2026


def split_for_year(year: int) -> dict:
    if not FIRST_TEST_YEAR <= year <= LAST_TEST_YEAR:
        raise ValueError(f"forecast year must be {FIRST_TEST_YEAR}-{LAST_TEST_YEAR}")
    ts = pd.Timestamp
    return {
        "train_start": FIRST_TARGET, "train_end": ts(f"{year - 3}-12-31"),
        "val_start": ts(f"{year - 2}-01-31"), "val_end": ts(f"{year - 1}-12-31"),
        "test_start": ts(f"{year}-01-31"), "test_end": min(ts(f"{year}-12-31"), LAST_TARGET),
    }


def test_year_of_formation(eom) -> int:
    """Forecast (test) year in which the target month of this formation month falls."""
    return (pd.Timestamp(eom) + pd.offsets.MonthEnd(1)).year
