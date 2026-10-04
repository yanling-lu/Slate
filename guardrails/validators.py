from __future__ import annotations
import hashlib
import json
import re
from datetime import date

from schemas.models import SignalBatch, Signal, FilingEvidence


def sanitize_signal_batch(raw, as_of: str) -> dict:
    """Repair common LLM output violations before Pydantic validation.

    Fixes:
      - raw is not a dict                        -> empty batch
      - missing top-level as_of_date             -> injected
      - missing per-signal as_of_date            -> injected
      - ticker == None                           -> "UNKNOWN"
      - audit_trail entries that are strings     -> wrapped as dicts
      - signals list missing                     -> []
      - a signal that itself contains a nested
        "signals" array                          -> flattened (drop wrapper)
      - missing optional fields                  -> defaults
    """
    # --- top-level shape ---
    if not isinstance(raw, dict):
        raw = {}
    raw.setdefault("as_of_date", as_of)
    raw.setdefault("signals", [])
    raw.setdefault("audit_trail", [])

    if not isinstance(raw["signals"], list):
        raw["signals"] = []
    if not isinstance(raw["audit_trail"], list):
        raw["audit_trail"] = []

    # --- flatten any nested "signals" inside a signal ---
    flat: list[dict] = []
    for sig in raw["signals"]:
        if not isinstance(sig, dict):
            continue
        nested = sig.get("signals")
        if isinstance(nested, list) and nested:
            # The wrapper is a batch; propagate shared fields to children.
            inherited = {
                k: v for k, v in sig.items()
                if k in ("permno", "ticker", "as_of_date", "document_id")
                and v is not None
            }
            for child in nested:
                if not isinstance(child, dict):
                    continue
                flat.append({**inherited, **child})   # child wins on conflict
            # If wrapper had no children, keep the wrapper as a plain signal
            if not any(isinstance(c, dict) for c in nested):
                flat.append({k: v for k, v in sig.items() if k != "signals"})
        else:
            # Strip a stray empty "signals" key if present
            clean = {k: v for k, v in sig.items() if k != "signals"}
            flat.append(clean)
    raw["signals"] = flat

    # --- normalize each signal ---
    for sig in raw["signals"]:
        if sig.get("ticker") is None:
            sig["ticker"] = "UNKNOWN"
        if not isinstance(sig.get("ticker"), str):
            sig["ticker"] = str(sig.get("ticker"))
        sig.setdefault("as_of_date", as_of)
        sig.setdefault("revision_round", 0)
        sig.setdefault("addressed_revision_requests", [])
        sig.setdefault("alternative_interpretations", [])
        sig.setdefault("injection_attempts", [])
        sig.setdefault("insufficient_evidence", False)
        sig.setdefault("document_id", None)
        if not isinstance(sig.get("evidence"), list):
            sig["evidence"] = []
        sig['source'] = _infer_source(sig)

    # --- audit_trail ---
    fixed_trail: list[dict] = []
    for i, entry in enumerate(raw["audit_trail"]):
        if isinstance(entry, dict):
            if "step" in entry and not isinstance(entry["step"], str):
                entry["step"] = f"step_{entry['step']}"
            entry.setdefault("step", f"step_{i + 1}")
            fixed_trail.append(entry)
        else:
            fixed_trail.append({
                "step": f"wrapped_{i + 1}",
                "tool": "unknown",
                "args": {},
                "note": str(entry)[:240],
            })
    raw["audit_trail"] = fixed_trail

    return raw


def validate_signal_batch(
    raw: dict,
    as_of: date,
    retrieved_doc_ids: set[str],
    retrieved_feature_ids: set[str] | None = None,
) -> SignalBatch:
    """Deterministic post-LLM validation.

    Never raises. Invalid signals are dropped and logged via the returned
    batch's audit_trail. If everything fails, an empty batch is returned
    so the pipeline can continue and the failure is visible in the log.
    """
    raw = sanitize_signal_batch(raw, as_of.isoformat())

    kept_signals: list[dict] = []
    rejection_notes: list[dict] = []

    for idx, sig in enumerate(raw["signals"]):
        try:
            validated = Signal.model_validate(sig)
        except Exception as exc:
            rejection_notes.append({
                "step": f"reject_signal_{idx}",
                "tool": "schema",
                "args": {"permno": sig.get("permno"), "ticker": sig.get("ticker")},
                "note": str(exc)[:400],
            })
            continue

        # Look-ahead: no filing dated after as_of
        lookahead = False
        for e in validated.evidence:
            if isinstance(e, FilingEvidence) and e.filing_date > as_of:
                rejection_notes.append({
                    "step": f"reject_signal_{idx}_lookahead",
                    "tool": "cutoff",
                    "args": {"permno": validated.permno},
                    "note": f"filing_date {e.filing_date} > as_of {as_of}",
                })
                lookahead = True
                break
        if lookahead:
            continue

        # Fabricated IDs
        fabricated = None
        for e in validated.evidence:
            if isinstance(e, FilingEvidence):
                if e.document_id not in retrieved_doc_ids:
                    fabricated = f"document_id {e.document_id}"
                    break
            else:
                if retrieved_feature_ids is not None and e.feature_id not in retrieved_feature_ids:
                    fabricated = f"feature_id {e.feature_id}"
                    break
        if fabricated:
            rejection_notes.append({
                "step": f"reject_signal_{idx}_fabricated",
                "tool": "evidence",
                "args": {"permno": validated.permno},
                "note": fabricated,
            })
            continue

        # Rationale cannot mention dates > as_of
        future_date = None
        for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", validated.rationale):
            y, mo, d = map(int, m.groups())
            try:
                if date(y, mo, d) > as_of:
                    future_date = m.group(0)
                    break
            except ValueError:
                continue
        if future_date:
            rejection_notes.append({
                "step": f"reject_signal_{idx}_rationale_date",
                "tool": "cutoff",
                "args": {"permno": validated.permno},
                "note": f"rationale mentions future date {future_date}",
            })
            continue

        kept_signals.append(validated.model_dump(mode="json"))

    return SignalBatch(
        as_of_date=as_of,
        signals=[Signal.model_validate(s) for s in kept_signals],
        audit_trail=list(raw.get("audit_trail", [])) + rejection_notes,
    )

def hash_record(record: dict, prev_hash: str) -> str:
    payload = json.dumps(record, sort_keys=True, default=str).encode()
    return hashlib.sha256(prev_hash.encode() + payload).hexdigest()

def _infer_source(sig: dict) -> str:
    ev = sig.get("evidence") or []
    has_filing = any(isinstance(e, dict) and e.get("kind") == "filing" for e in ev)
    has_char = any(isinstance(e, dict) and e.get("kind") == "characteristic" for e in ev)
    if has_filing and has_char:
        return "both"
    if has_filing:
        return "filing_8k"
    return "characteristics"