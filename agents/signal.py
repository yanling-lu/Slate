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

        filings = retrieve_filings(permno, as_of, max_docs=10)
        for f in filings:
            f["injection_attempts"] = scan_injection(f.get("text") or "")
            f["text_redacted"] = redact_for_llm(
                f.get("text") or "", f.get("ticker") or "", f.get("company_name") or "")
            f["signature"] = compute_event_signature(f)
            f.pop("text", None)  # never pass raw text to LLM

        chars = get_characteristics(permno, as_of, top_k=20)
        peers = get_peer_context(permno, as_of)

        payload = {
            "as_of_date": state["as_of_date"],
            "permno": permno,
            "filings": filings,
            "characteristics": chars,
            "peer_context": peers,
        }
        return self.llm.call(payload)