from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from typing import Literal, Optional, Union
from datetime import date


class FilingEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["filing"] = "filing"
    document_id: str
    item_code: Optional[str] = None
    quote: str = Field(..., max_length=240)
    filing_date: date


class CharacteristicEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["characteristic"] = "characteristic"
    feature_id: str
    value: Optional[float] = None
    z_score_vs_month: Optional[float] = None
    direction: Literal["high", "low"] = "high"
    interpretation: str = Field(..., max_length=240)


Evidence = Union[FilingEvidence, CharacteristicEvidence]


class Signal(BaseModel):
    """
    One signal per (permno, as_of_date). Sentiment-driven.
    Sentiment scale: -3 (very bearish) to +3 (very bullish), continuous.
    Long/short buckets are derived from sentiment:
      - sentiment >= +0.5  -> direction = "long"
      - sentiment <= -0.5  -> direction = "short"
      - otherwise          -> direction = "neutral"
    """
    model_config = ConfigDict(extra="forbid")
    permno: int
    ticker: str
    as_of_date: date
    sentiment: float = Field(..., ge=-3.0, le=3.0)
    document_id: Optional[str] = None
    direction: Literal["long", "short", "neutral"]
    source: Literal["filing_8k", "characteristics", "both"] = "characteristics"
    score: float = Field(..., ge=-2.0, le=2.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    catalyst: str
    time_horizon_months: int = Field(..., ge=1, le=24)
    rationale: str
    evidence: list[Evidence]
    alternative_interpretations: list[str] = []
    injection_attempts: list[str] = []
    insufficient_evidence: bool = False
    revision_round: int = 0
    addressed_revision_requests: list[str] = []

    @field_validator("evidence")
    @classmethod
    def at_least_one_if_not_neutral(cls, v, info):
        if info.data.get("direction") != "neutral" and len(v) == 0:
            raise ValueError("non-neutral signals require ≥1 evidence item")
        return v

    @model_validator(mode="after")
    def check_sign_and_magnitude(self):
        if self.direction == "long" and self.score < 0:
            raise ValueError("long signal with negative score")
        if self.direction == "short" and self.score > 0:
            raise ValueError("short signal with positive score")
        if self.direction == "neutral" and abs(self.score) > 0.5:
            raise ValueError("neutral signal with |score| > 0.5")
        if self.direction == "neutral" and self.confidence > 0.5:
            raise ValueError("neutral signal with confidence > 0.5")
        return self


class SignalBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    as_of_date: date
    signals: list[Signal] = []
    audit_trail: list[dict] = []


class Counterargument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal[
        "confound", "crowding", "base_rate",
        "adversarial_filing", "timing", "structural",
    ]
    claim: str
    evidence: list[Evidence] = []
    severity: Literal["low", "medium", "high"]


class Challenge(BaseModel):
    model_config = ConfigDict(extra="allow")   # Devil adds extra keys post-validation
    permno: int
    ticker: str
    signal_id: str
    resilience_score: float = Field(..., ge=0.0, le=1.0)
    recommendation: Literal["proceed", "revise", "reject"]
    counterarguments: list[Counterargument]
    residual_uncertainties: list[str] = []
    round: int = 0
    objection_score: int = Field(0, ge=0, le=3)
    revision_requests: list[str] = []
    kept_characteristics: list[dict] = []
    kept_event_types: list[dict] = []


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")
    permno: int
    ticker: str
    weight: float = Field(..., ge=-0.05, le=0.05)


class PortfolioTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    as_of_date: date
    target_weights: list[Position]
    constraint_report: dict = {}
    human_approval_required: bool = False
    approval_reasons: list[str] = []
    trade_rationale: list[dict] = []
    violations: list[str] = []
    sectors: dict[int, str] = {}