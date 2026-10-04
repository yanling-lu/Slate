from __future__ import annotations
import hashlib, json
from datetime import date
from schemas.models import SignalBatch, FilingEvidence


def validate_signal_batch(
    raw: dict,
    as_of: date,
    retrieved_doc_ids: set[str],
    retrieved_feature_ids: set[str] | None = None,
) -> SignalBatch:
    """Deterministic post-LLM validation. Raises on violation."""
    batch = SignalBatch.model_validate(raw)

    for sig in batch.signals:
        # 1. Look-ahead: no filing dated after as_of
        for e in sig.evidence:
            if isinstance(e, FilingEvidence) and e.filing_date > as_of:
                raise ValueError(f"Look-ahead: {sig.ticker} cites future filing")

        # 2. Fabricated document_id / feature_id
        for e in sig.evidence:
            if isinstance(e, FilingEvidence):
                if e.document_id not in retrieved_doc_ids:
                    raise ValueError(f"Fabricated document_id: {e.document_id}")
            else:
                if retrieved_feature_ids is not None and e.feature_id not in retrieved_feature_ids:
                    raise ValueError(f"Fabricated feature_id: {e.feature_id}")

        # 3. Rationale cannot mention dates > as_of
        import re
        for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", sig.rationale):
            y, mo, d = map(int, m.groups())
            if date(y, mo, d) > as_of:
                raise ValueError(f"Rationale references future date {m.group(0)}")

    return batch


def hash_record(record: dict, prev_hash: str) -> str:
    payload = json.dumps(record, sort_keys=True, default=str).encode()
    return hashlib.sha256(prev_hash.encode() + payload).hexdigest()