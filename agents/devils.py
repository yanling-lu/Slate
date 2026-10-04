"""Devil's Advocate agent (pipeline version).

Sits between the Signal Agent and the Portfolio Agent in orchestration/graph.py:

    signal -> validate -> DEVIL -> (revise_signal -> signal ...) -> portfolio -> policy

    run(validated_signals: dict, as_of_date: str) -> {"as_of_date": ..., "challenges": [...]}

For every non-neutral signal it:
  1. gathers the stock's 8-K filings up to as_of_date (tools.filings.retrieve_filings,
     which enforces the information cutoff),
  2. cleans the text: team redaction (guardrails.injection.redact_for_llm) plus
     cover page, signature, websites, years and month-year dates removed,
  3. asks the LLM for the strongest reasons the trade could be WRONG, scored 0-3,
  4. narrows the signal's evidence: keeps at most 7 characteristics and 3 event types
     (8-K item codes), each scored 0-1,
  5. checks the answer in code (quotes must really be in the filing, codes must exist,
     scores must be in range) before building a schemas.models.Challenge.

Objection score -> recommendation (read by the graph router and the portfolio agent):
    3 -> "reject"  (portfolio: veto)
    2 -> "revise"  (graph sends it back to the Signal Agent, up to MAX_REVISION_ROUNDS;
                    portfolio: haircut)
    0-1 -> "proceed"
Each challenge also carries `objection_score`, which the portfolio agent reads directly.

Extra keys added for the team (outside the Challenge schema, read with .get()):
    objection_score, revision_requests, kept_characteristics, kept_event_types
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date

from config import settings
from agents.base import LLMAgent
from tools.filings import retrieve_filings
from guardrails.injection import scan_injection, redact_for_llm

try:                                    # validate against the team schema when available
    from schemas.models import Challenge
except Exception:                       # pragma: no cover
    Challenge = None

MAX_TEXT_CHARS = 12000                  # filing text per stock, all filings together (keeps cost down)
PER_DOC_CHARS = 4000                    # filing text per document
KEEP_CHARACTERISTICS = 7
KEEP_EVENT_TYPES = 3
COUNTER_TYPES = {"confound", "crowding", "base_rate", "adversarial_filing", "timing", "structural"}
SEVERITY_BY_SCORE = {0: "low", 1: "low", 2: "medium", 3: "high"}
RECOMMENDATION_BY_SCORE = {0: "proceed", 1: "proceed", 2: "revise", 3: "reject"}

DEFAULT_PROMPT = """You are a skeptical risk analyst (the devil's advocate) on an investment team.
A signal analyst has proposed a stock trade. Your job is to find the strongest reasons
the trade could be WRONG, and to keep only the evidence that really supports it.

You receive JSON with: the proposed signal (direction, score, confidence, catalyst,
rationale, evidence), and the company's recent 8-K filings (cleaned text and item codes).

Rules:
- Use ONLY the JSON you are given. Do not use outside knowledge about any company,
  market event or year. Names and dates are hidden on purpose; do not guess the company.
- Every filing quote must be copied EXACTLY from the filing text (one sentence, short).
- Most filings are routine (standard pay, bonus or severance agreements, bylaw updates,
  routine loan paperwork, normal board changes). Those are score 0 or 1.
- A problem that is already settled or resolved is less serious than an open one.
- If the real news is in an attachment that is not included (for example "Exhibit 99.1",
  "press release", "investor presentation"), do NOT guess what it says.
- Do not invent problems. If nothing matters, give score 0.

What to look for:
- For a LONG: bad news hidden inside good-sounding news (renewed or replaced programs,
  non-binding promises like "may" or "does not commit", unused past authorizations,
  new debt, one-time gains, managers or auditors leaving, lawsuits, investigations,
  restatements, going-concern warnings, vague or lowered guidance).
- For a SHORT: good news the negative view ignores (buybacks, new contracts, debt paid
  off, strong results), or bad news that is small or already old.
- Evidence problems: characteristics that do not support the stated direction,
  a catalyst that is not in the filings, a conclusion stronger than the evidence.

Objection score:
0 = no real problem, or not enough information
1 = small concern
2 = serious concern: the signal should be revised and the position made smaller
3 = the trade is probably wrong

Counterargument type must be one of:
confound, crowding, base_rate, adversarial_filing, timing, structural

Also narrow the evidence:
- kept_characteristics: at most 7 of the signal's characteristic feature_ids that best
  support the trade, each with a score between 0 and 1 (1 = strongest support).
- kept_event_types: at most 3 of the 8-K item codes present in the filings that matter
  most for this trade, each with a score between 0 and 1.

Answer ONLY with this JSON:
{"objection_score": 0,
 "counterarguments": [{"type": "timing", "claim": "one sentence", "severity": "low",
                       "document_id": "id or empty", "quote": "exact sentence or empty"}],
 "revision_requests": ["what the signal analyst should check or fix"],
 "residual_uncertainties": ["short sentence"],
 "kept_characteristics": [{"feature_id": "...", "score": 0.8}],
 "kept_event_types": [{"item_code": "2.02", "score": 0.7}]}"""

MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|"
          "December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")


# ----------------------------------------------------------------- text cleaning
def strip_cover(text: str) -> str:
    """Keep the news: start at the first 'Item x.xx', cut the signature, hide websites."""
    text = str(text or "")
    m = re.search(r"\bItem\s+\d\.\d{2}", text)
    if m:
        text = text[m.start():]
    end = text.rfind("SIGNATURE")
    if end > 0:
        text = text[:end]
    return re.sub(r"(https?://\S+|www\.\S+|\b[\w.-]+\.(com|net|org|io)\b)", "[WEBSITE]", text)


def extra_redaction(text: str) -> str:
    """On top of the team's redact_for_llm: month-year dates, numeric dates, years."""
    text = re.sub(rf"\b({MONTHS})\.?\s+\d{{1,2}},?\s+\d{{4}}", "[DATE]", text)
    text = re.sub(rf"\b({MONTHS})\.?\s+(19|20)\d{{2}}\b", "[DATE]", text)
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", text)
    return re.sub(r"\b(19|20)\d{2}\b", "[YEAR]", text)


def clean_for_llm(text: str, ticker: str, company_name: str) -> str:
    return extra_redaction(redact_for_llm(strip_cover(text), ticker or "", company_name or ""))


def _squash(s) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _items_of(f: dict) -> list[str]:
    raw = f.get("items")
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        try:
            raw = " ".join(map(str, raw))
        except TypeError:
            raw = str(raw)
    codes = set(re.findall(r"\d\.\d{2}", raw))
    if not codes:                       # fall back to the text itself ("Item 5.02 ...")
        codes = set(re.findall(r"Item\s+(\d\.\d{2})", str(f.get("text") or "")))
    return sorted(codes - {"9.01"})     # 9.01 only lists attachments


def _score01(v) -> float:
    try:
        return float(min(max(float(v), 0.0), 1.0))
    except (TypeError, ValueError):
        return 0.0


def signal_id(sig: dict) -> str:
    return hashlib.sha256(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------- agent
class DevilsAdvocate:
    def __init__(self, model: str | None = None, max_docs: int = 10):
        try:
            prompt = settings.devil_system_prompt()     # team prompt from .env, if set
        except Exception:
            prompt = DEFAULT_PROMPT
        self.llm = LLMAgent(model or settings.MODEL_DEVIL, prompt,
                            temperature=settings.TEMP_DEVIL)
        self.max_docs = max_docs

    # ---- public: called by orchestration/graph.py node_devil ----
    def run(self, validated_signals: dict, as_of_date: str) -> dict:
        as_of = date.fromisoformat(str(as_of_date)[:10])
        challenges = []
        for sig in (validated_signals or {}).get("signals", []) or []:
            if sig.get("insufficient_evidence") or sig.get("direction") == "neutral":
                continue                                  # no trade, nothing to challenge
            challenges.append(self.challenge_one(sig, as_of))
        return {"as_of_date": as_of.isoformat(), "challenges": challenges}

    # ---- one signal ----
    def challenge_one(self, sig: dict, as_of: date) -> dict:
        permno = int(sig["permno"])
        filings, by_id, codes, used = [], {}, set(), 0
        for f in retrieve_filings(permno, as_of, max_docs=self.max_docs):
            fd = f.get("filing_date")
            if fd is not None and str(fd)[:10] > as_of.isoformat():
                continue                                  # cutoff, checked again here
            if used >= MAX_TEXT_CHARS:
                break
            text = clean_for_llm(f.get("text"), f.get("ticker"), f.get("company_name"))
            text = text[:min(PER_DOC_CHARS, MAX_TEXT_CHARS - used)]
            used += len(text)
            doc_id = str(f.get("document_id"))
            items = _items_of(f)
            codes.update(items)
            by_id[doc_id] = {"text": text, "filing_date": str(fd)[:10] if fd is not None else None,
                             "items": items}
            filings.append({"document_id": doc_id, "item_codes": items, "text": text,
                            "injection_attempts": scan_injection(f.get("text") or "")})

        feature_ids = {e.get("feature_id") for e in sig.get("evidence", [])
                       if e.get("kind") == "characteristic"}
        payload = {
            "proposed_signal": {
                "direction": sig.get("direction"), "score": sig.get("score"),
                "confidence": sig.get("confidence"),
                "catalyst": extra_redaction(redact_for_llm(str(sig.get("catalyst", "")),
                                                           sig.get("ticker", ""), "")),
                "rationale": extra_redaction(redact_for_llm(str(sig.get("rationale", "")),
                                                            sig.get("ticker", ""), "")),
                "time_horizon_months": sig.get("time_horizon_months"),
                "evidence": [{k: (extra_redaction(str(v)) if k in ("quote", "interpretation") else v)
                              for k, v in e.items() if k != "filing_date"}
                             for e in sig.get("evidence", [])],
                "revision_round": sig.get("revision_round", 0),
            },
            "filings": filings,
        }

        try:
            answer = self.llm.call(payload)
        except Exception as exc:                          # never break the pipeline
            answer = {"objection_score": 0, "counterarguments": [],
                      "residual_uncertainties": [f"devil's advocate call failed: {type(exc).__name__}"]}
        return self._build_challenge(sig, answer, by_id, codes, feature_ids)

    # ---- code-side checks: the AI argues, the rules decide ----
    def _build_challenge(self, sig, answer, by_id, codes, feature_ids) -> dict:
        try:
            score = int(answer.get("objection_score", 0))
        except (TypeError, ValueError):
            score = 0
        score = min(max(score, 0), 3)

        counter, dropped = [], 0
        for c in (answer.get("counterarguments") or [])[:5]:
            ctype = c.get("type") if c.get("type") in COUNTER_TYPES else "structural"
            sev = c.get("severity") if c.get("severity") in ("low", "medium", "high") else SEVERITY_BY_SCORE[score]
            evidence = []
            doc_id, quote = str(c.get("document_id") or ""), str(c.get("quote") or "")
            if quote:
                doc = by_id.get(doc_id)
                if doc is None or len(quote.strip()) < 15 or _squash(quote) not in _squash(doc["text"]):
                    dropped += 1                          # quote not really in the filing
                    continue
                if doc["filing_date"]:
                    evidence.append({"kind": "filing", "document_id": doc_id,
                                     "item_code": doc["items"][0] if doc["items"] else None,
                                     "quote": quote[:240], "filing_date": doc["filing_date"]})
            counter.append({"type": ctype, "claim": str(c.get("claim", ""))[:500],
                            "evidence": evidence, "severity": sev})

        if dropped and not counter and score >= 2:
            score = 1                                     # objection had no real proof
        uncertainties = [str(u)[:300] for u in (answer.get("residual_uncertainties") or [])][:5]
        if dropped:
            uncertainties.append(f"{dropped} counterargument(s) removed: quote not found in the filing")

        kept_chars = []
        for k in answer.get("kept_characteristics") or []:
            fid = k.get("feature_id")
            if fid in feature_ids and fid not in {x["feature_id"] for x in kept_chars}:
                kept_chars.append({"feature_id": fid, "score": _score01(k.get("score"))})
        kept_chars = sorted(kept_chars, key=lambda x: -x["score"])[:KEEP_CHARACTERISTICS]

        kept_events = []
        for k in answer.get("kept_event_types") or []:
            code = str(k.get("item_code", ""))
            if code in codes and code not in {x["item_code"] for x in kept_events}:
                kept_events.append({"item_code": code, "score": _score01(k.get("score"))})
        kept_events = sorted(kept_events, key=lambda x: -x["score"])[:KEEP_EVENT_TYPES]

        challenge = {
            "permno": int(sig["permno"]),
            "ticker": str(sig.get("ticker", "")),
            "signal_id": signal_id(sig),
            "resilience_score": round(1 - score / 3, 3),
            "recommendation": RECOMMENDATION_BY_SCORE[score],
            "counterarguments": counter,
            "residual_uncertainties": uncertainties,
            "round": int(sig.get("revision_round", 0) or 0),
        }
        if Challenge is not None:                        # check the shape the team agreed on
            try:
                challenge = Challenge(**challenge).model_dump(mode="json")
            except Exception as exc:
                challenge["residual_uncertainties"].append(f"schema check failed: {exc}"[:300])

        challenge.update({
            "objection_score": score,
            "revision_requests": [str(r)[:300] for r in (answer.get("revision_requests") or [])][:5]
                                 if score == 2 else [],
            "kept_characteristics": kept_chars,
            "kept_event_types": kept_events,
        })
        return challenge
