from datetime import date
from guardrails.validators import validate_signal_batch


def test_validator_rejects_lookahead():
    raw = {
        "as_of_date": "2024-06-30",
        "signals": [{
            "permno": 1, "ticker": "X", "as_of_date": "2024-06-30",
            "direction": "long", "score": 1.0, "confidence": 0.6,
            "catalyst": "x", "time_horizon_months": 2, "rationale": "r",
            "evidence": [{
                "kind": "filing", "document_id": "d1", "item_code": "2.02",
                "quote": "q", "filing_date": "2024-07-15",   # future!
            }],
        }],
    }
    try:
        validate_signal_batch(raw, date(2024, 6, 30), {"d1"})
        assert False, "should have raised"
    except ValueError as e:
        assert "Look-ahead" in str(e)