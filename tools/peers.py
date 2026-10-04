from __future__ import annotations
from datetime import date
from tools.data_loader import load_chars


def get_peer_context(permno: int, as_of_date: date, sector_col: str = "sic") -> dict:
    df = load_chars()
    month_key = as_of_date.replace(day=1)

    target = df.loc[(df["permno"] == permno) & (df["eom"] == month_key)]
    if target.empty:
        return {}
    if sector_col not in target.columns:
        return {"sector": "UNKNOWN", "n_peers": 0}

    sector = target.iloc[0][sector_col]
    peers = df.loc[(df["eom"] == month_key) & (df[sector_col] == sector)]
    med = peers["market_equity"].median() if "market_equity" in peers.columns else None
    return {
        "sector": str(sector),
        "n_peers": int(len(peers)),
        "median_market_equity": float(med) if med is not None else None,
    }