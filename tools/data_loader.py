"""
Loads and caches the two provided Parquet datasets. All downstream tools
query this loader; nothing else touches the files directly. This keeps the
as-of filter, memory caching, and column selection in one place.
"""
from __future__ import annotations
import pandas as pd
from config import settings
from functools import lru_cache


@lru_cache(maxsize=1)
def load_chars() -> pd.DataFrame:
    df = pd.read_parquet(settings.CHARS_PATH, columns=None )
    df["eom"] = pd.to_datetime(df["eom"])
    return df


@lru_cache(maxsize=1)
def load_filings() -> pd.DataFrame:
    import pyarrow.parquet as pq
    path = settings.FILINGS_PATH
    available = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in [
            "document_id", "permno",
            "filing_date", "filing_timestamp_utc",
            "items", "text", "ticker", "company_name",
            "provider_ticker", "provider_company_name", "content_company_name",
            "gvkey", "iid", "cik", "cusip", "form_type",
            "period_end_date", "content_report_date",
            "text_chars", "word_count",
        ] if c in available]

    df = pd.read_parquet(path, columns=cols)
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date

    if "items" in df.columns and "items" not in df.columns:
        df = df.rename(columns="items")

    # ---- normalize list columns to plain Python lists ----
    def _to_list(x):
        if x is None:
            return []
        try:
            if x != x:
                return []
        except Exception:
            pass
        try:
            return [str(v) for v in list(x)]
        except TypeError:
            return []

    if "items" in df.columns:
        df["items"] = df["items"].apply(_to_list)

    return df


@lru_cache(maxsize=1)
def load_factor_list() -> list[str]:
    return pd.read_csv(settings.FACTOR_LIST_PATH)["variable"].tolist()