import pytest
from datetime import date
from schemas.models import Signal, FilingEvidence, CharacteristicEvidence


def _filing_ev():
    return FilingEvidence(document_id="d1", item_code="2.02",
                          quote="q", filing_date=date(2024, 6, 1))


def test_signal_requires_evidence_if_not_neutral():
    with pytest.raises(ValueError):
        Signal(permno=1, ticker="X", as_of_date=date(2024, 6, 30),
               direction="long", score=1.0, confidence=0.6,
               catalyst="x", time_horizon_months=2, rationale="r", evidence=[])


def test_score_sign_must_match_direction():
    with pytest.raises(ValueError):
        Signal(permno=1, ticker="X", as_of_date=date(2024, 6, 30),
               direction="long", score=-1.0, confidence=0.6,
               catalyst="x", time_horizon_months=2, rationale="r",
               evidence=[_filing_ev()])


def test_characteristic_only_signal_ok():
    ev = CharacteristicEvidence(feature_id="be_me", value=0.5,
                                z_score_vs_month=1.8, direction="high",
                                interpretation="high be_me historically positive")
    s = Signal(permno=1, ticker="X", as_of_date=date(2024, 6, 30),
               direction="long", score=0.7, confidence=0.45,
               catalyst="value anomaly", time_horizon_months=3,
               rationale="characteristic-only", evidence=[ev])
    assert s.direction == "long"