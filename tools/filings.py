from __future__ import annotations
from datetime import date, timedelta
from tools.data_loader import load_filings

MATERIAL_ITEMS = {"1.01","1.02","2.01","2.02","2.05","2.06",
                  "4.01","4.02","5.02","7.01","8.01"}

def retrieve_filings(permno: int, as_of_date: date,
                     items: list[str] | None = None,
                     max_docs: int = 10,
                     lookback_days: int = 30) -> list[dict]:
    """Return filings for one permno filed in [as_of_date - lookback, as_of_date].
    Enforces the information cutoff: filing_date <= as_of_date.
    """
    df = load_filings()
    window_start = as_of_date - timedelta(days=lookback_days)

    m = ((df["permno"] == permno)
        & (df["filing_date"] <= as_of_date)
        & (df["filing_date"] >= window_start)
    )
    sub = df.loc[m].sort_values("filing_date", ascending=False).head(max_docs)

    if items:
        sub = sub[sub["items"].apply(
            lambda codes: any(c in codes for c in items))]

    return sub[[
        "document_id","permno","filing_date","items","text",
        "ticker","company_name"
    ]].to_dict("records")