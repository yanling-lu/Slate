from __future__ import annotations
from datetime import date
from tools.data_loader import load_chars, load_factor_list
import numpy as np

def get_characteristics(permno: int, as_of_date: date,
                        top_k: int = 20) -> list[dict]:
    """Return the permno's characteristic snapshot for the month of as_of_date.
    Only characteristics whose month <= as_of_date are returned (no look-ahead).
    The target column ret_exc_lead1m is NEVER returned.
    """
    df = load_chars()
    factor_cols = load_factor_list()

    month_key = as_of_date.replace(day=1)
    m = (df["permno"] == permno) & (df["eom"] == month_key)
    if not m.any():
        return []

    row = df.loc[m].iloc[0]

    # Rank each factor cross-sectionally within the same month to get a
    # "how unusual is this stock" score. top_k = largest absolute z-scores.
    month_slice = df.loc[df["eom"] == month_key, factor_cols]
    mu, sd = month_slice.mean(), month_slice.std(ddof=0).replace(0, np.nan)
    z = (month_slice - mu) / sd
    z_row = z.loc[row.name].dropna()

    picked = z_row.abs().sort_values(ascending=False).head(top_k)
    return [
        {
            "feature_id": f,
            "value": float(row[f]) if not np.isnan(row[f]) else None,
            "z_score_vs_month": float(picked[f]),
            "direction": "high" if picked[f] > 0 else "low",
        }
        for f in picked.index
    ]