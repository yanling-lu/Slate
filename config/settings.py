"""
Central settings loader. Reads .env once, exposes typed accessors, and
parses the prompt strings (which are stored with literal \n sequences).
"""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _require(key: str) -> str:
    v = os.environ.get(key)
    if v is None or v == "":
        raise RuntimeError(f"Missing required env var: {key}")
    return v


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_float(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v not in (None, "") else default


def _get_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v not in (None, "") else default


def _get_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key, "").lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def _unfold(raw: str) -> str:
    """Prompts are stored with literal \\n in .env; convert to real newlines."""
    return raw.replace("\\n", "\n")


# ---------- Secrets ----------
@lru_cache(maxsize=1)
def openai_api_key() -> str:
    return _require("OPENAI_API_KEY")


@lru_cache(maxsize=1)
def fred_api_key() -> str:
    return _require("FRED_API_KEY")


# ---------- Model config ----------
MODEL_REASONING = _get("MODEL_REASONING", "gpt-4o")
MODEL_TRIAGE = _get("MODEL_TRIAGE", "gpt-4o-mini")
MODEL_DEVIL = _get("MODEL_DEVIL", "gpt-4o")
TEMP_SIGNAL = _get_float("TEMP_SIGNAL", 0.0)
TEMP_DEVIL = _get_float("TEMP_DEVIL", 0.3)
TEMP_PORTFOLIO = _get_float("TEMP_PORTFOLIO", 0.0)


# ---------- Pipeline limits ----------
MAX_TOKENS_PER_DOC = _get_int("MAX_TOKENS_PER_DOC", 8000)
MAX_STEPS_SIGNAL = _get_int("MAX_STEPS_SIGNAL", 5)
MAX_STEPS_DEVIL = _get_int("MAX_STEPS_DEVIL", 8)
MAX_STEPS_PORTFOLIO = _get_int("MAX_STEPS_PORTFOLIO", 6)
MAX_REVISION_ROUNDS = _get_int("MAX_REVISION_ROUNDS", 2)
SIGNAL_LOOKBACK_DAYS = _get_int("SIGNAL_LOOKBACK_DAYS", 30)
TOP_K_CHARACTERISTICS = _get_int("TOP_K_CHARACTERISTICS", 20)


# ---------- Paths ----------
CHARS_PATH = Path(_get("CHARS_PATH", "data/chars_final_with_names.parquet"))
FILINGS_PATH = Path(_get("FILINGS_PATH", "data/8k_20150101_20260831_identified.parquet"))
FACTOR_LIST_PATH = Path(_get("FACTOR_LIST_PATH", "data/factor_char_list.csv"))
AUDIT_DB = _get("AUDIT_DB", "audit.db")


# ---------- Kill switch ----------
SYSTEM_HALT = _get_bool("SYSTEM_HALT", False)


# ---------- Prompts ----------
@lru_cache(maxsize=1)
def signal_system_prompt() -> str:
    return _unfold(_require("SIGNAL_SYSTEM_PROMPT"))


@lru_cache(maxsize=1)
def devil_system_prompt() -> str:
    return _unfold(_require("DEVIL_SYSTEM_PROMPT"))


@lru_cache(maxsize=1)
def portfolio_system_prompt() -> str:
    return _unfold(_require("PORTFOLIO_SYSTEM_PROMPT"))

@lru_cache(maxsize=1)
def instruction_first_pass() -> str:
    return _get("INSTRUCTION_FIRST_PASS", "FIRST PASS. Produce the initial signal.")


@lru_cache(maxsize=1)
def instruction_revision_pass() -> str:
    return _get(
        "INSTRUCTION_REVISION_PASS",
        "REVISION PASS. Address every item in revision_requests with concrete "
        "evidence from the filings shown. Use only the characteristics listed "
        "in kept_characteristics and only filings whose item_codes appear in "
        "kept_event_types. Fill addressed_revision_requests[] with the "
        "requests you addressed. If a request cannot be addressed with "
        "available evidence, downgrade to neutral with insufficient_evidence=true.",
    )