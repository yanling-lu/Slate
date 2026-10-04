import re

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(the\s+)?system\s+prompt",
    r"you\s+are\s+now",
    r"new\s+instructions\s*:",
    r"</?(system|assistant|user)>",
    r"forget\s+(everything|all)\s+(above|before)",
]


def scan_injection(text: str) -> list[str]:
    hits = []
    for p in INJECTION_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            hits.append(p)
    return hits


def redact_for_llm(text: str, ticker: str, company_name: str) -> str:
    """Model-recall firewall: strip names/dates before the LLM sees the text."""
    out = text
    if company_name:
        out = re.sub(re.escape(company_name), "[COMPANY]", out, flags=re.IGNORECASE)
    if ticker:
        out = re.sub(rf"\b{re.escape(ticker)}\b", "[TICKER]", out)
    # Absolute dates
    out = re.sub(
        r"\b(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        "[DATE]", out)
    out = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "[DATE]", out)
    return out