def compute_event_signature(filing: dict) -> dict:
    """Cheap deterministic pre-classifier. No LLM. Used to save tokens on
    routine filings before they hit the reasoning model.
    """
    codes = set(filing.get("items", []))
    text = (filing.get("text") or "").lower()

    if codes & {"4.01","4.02"}:
        return {"item": sorted(codes), "sentiment": "distress", "novelty": "high"}
    if "2.02" in codes:
        sentiment = "unknown"
        if any(w in text for w in ["exceeded","above","raised","outperform"]):
            sentiment = "positive"
        elif any(w in text for w in ["below","missed","lowered","shortfall"]):
            sentiment = "negative"
        return {"item": sorted(codes), "sentiment": sentiment, "novelty": "medium"}
    if codes & {"2.05","2.06"}:
        return {"item": sorted(codes), "sentiment": "negative", "novelty": "high"}
    if codes & {"1.01","1.02"}:
        return {"item": sorted(codes), "sentiment": "unknown", "novelty": "medium"}
    if codes & {"5.02"}:
        return {"item": sorted(codes), "sentiment": "unknown", "novelty": "medium"}
    return {"item": sorted(codes), "sentiment": "neutral", "novelty": "low"}