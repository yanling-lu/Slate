from __future__ import annotations
from datetime import date
from config import settings
from agents.base import LLMAgent
from tools.filings import retrieve_filings
from tools.characteristics import get_characteristics
from tools.peers import get_peer_context
from tools.evet_signature import compute_event_signature
from guardrails.injection import scan_injection, redact_for_llm



class SignalAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = LLMAgent(model or settings.MODEL_REASONING,
                            settings.signal_system_prompt(),
                            temperature=settings.TEMP_SIGNAL)

    def run(self, state: dict) -> dict:
        """Build the point-in-time payload for one permno-month and ask the LLM."""
        permno = int(state.get("permno", 0))
        as_of = date.fromisoformat(state["as_of_date"])
        
        revision_requests = state.get('revision_requests') or []
        kept_characteristics = state.get('kept_characteristics')
        kept_event_types = state.get('kept_event_types')
        round_now = int(state.get('revision_round', 0))

        filings = retrieve_filings(permno, as_of, max_docs=10)
        
        if kept_event_types:
            allowed_codes = {str(k.get("item_code")) for k in kept_event_types if k.get("item_code")}
            if allowed_codes:
                filings = [
                    f for f in filings
                    if set(_items_of(f)) & allowed_codes
                ]
        
        for f in filings:
            f["injection_attempts"] = scan_injection(f.get("text") or "")
            f["text_redacted"] = redact_for_llm(
                f.get("text") or "", f.get("ticker") or "", f.get("company_name") or "")
            f["signature"] = compute_event_signature(f)
            f["items"] = _items_of(f)
            f.pop("text", None)  # never pass raw text to LLM

        chars = get_characteristics(permno, as_of, top_k=20)
        # Drops characteristics rejected by the devil's agent
        if kept_characteristics:
            allowed_feats = {str(k.get("feature_id")) for k in kept_characteristics
                             if k.get("feature_id")}
            if allowed_feats:
                chars = [c for c in chars if c["feature_id"] in allowed_feats]

        peers = get_peer_context(permno, as_of)

        payload = {
            "as_of_date": state["as_of_date"],
            "permno": permno,
            "filings": filings,
            "characteristics": chars,
            "peer_context": peers,
            "revision_requests": revision_requests,
            "kept_characteristics": kept_characteristics,
            "kept_event_types": kept_event_types,
        }
        return self.llm.call(payload)
    
def _items_of(f: dict) -> list[str]:
    """Same parser as the Devil's tools. Reads 'items' or falls back to text."""
    import re
    raw = f.get("items") if f.get("items") is not None else f.get("item_codes")
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        try:
            raw = " ".join(map(str, raw))
        except TypeError:
            raw = str(raw)
    codes = set(re.findall(r"\d\.\d{2}", raw))
    if not codes:
        codes = set(re.findall(r"Item\s+(\d\.\d{2})", str(f.get("text") or "")))
    return sorted(codes - {"9.01"})