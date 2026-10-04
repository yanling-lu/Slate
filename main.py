"""
Full agent pipeline.

Signal Agent <-> Devil's Advocate (revision loop, up to MAX_REVISION_ROUNDS)
    -> Portfolio Reconstruction Agent
    -> Policy check.

Usage:
    python main.py
    python main.py --as-of 2024-06-30 --n 10
    python main.py --as-of 2024-06-30 --permnos 14593 88234 55102

Env (in .env):
    PORTFOLIO_DEMO_BASELINE=1       synthesize a base signal (dev)
    BASE_SIGNALS_PATH=...           path to base_signals.parquet (prod)
    CHARS_PATH=...                  panel (shared by both agents)
    DEBUG_VALIDATION=true           dump raw LLM output before validation
    LANGCHAIN_TRACING_V2=true       enable LangSmith (default: disabled)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent))

# if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() != "true":
#     os.environ["LANGCHAIN_TRACING_V2"] = "false"
#     os.environ.pop("LANGCHAIN_API_KEY", None)
#     os.environ.pop("LANGCHAIN_ENDPOINT", None)

from pandas import Timestamp
from pandas.tseries.offsets import MonthEnd

from config import settings
from audit.logger import ImmutableAuditLog
from schemas.models import Position
from guardrails.validators import validate_signal_batch
from guardrails.policy import check_constraints
from agents.signal import SignalAgent
from agents.devils import DevilsAdvocate
from agents.portfolio import PortfolioReconstructionAgent


DEBUG = os.environ.get("DEBUG_VALIDATION", "false").lower() == "true"
MAX_IMBALANCE = 3.0   # longs may not exceed 3x shorts, and vice versa


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _validate_as_of(as_of: str) -> Timestamp:
    ts = Timestamp(as_of).normalize()
    if ts.day < 28:
        raise SystemExit(
            f"as_of_date {as_of} is mid-month. The portfolio agent forms on "
            f"month-end information. Use a month-end such as 2024-06-30."
        )
    return ts + MonthEnd(0)


def _pick_test_permnos(n: int = 5) -> list[int]:
    from tools.data_loader import load_filings
    MATERIAL = {"4.01", "4.02", "2.02", "2.05", "2.06", "1.01"}
    df = load_filings()
    mask = (
        df["items"].explode().isin(MATERIAL)
          .groupby(level=0).any()
          .reindex(df.index, fill_value=False)
    )
    return list(df[mask]["permno"].drop_duplicates().head(n))


def _retrieved_ids(permno: int, as_of: str) -> tuple[set[str], set[str]]:
    from tools.filings import retrieve_filings
    from tools.characteristics import get_characteristics
    as_of_d = date.fromisoformat(as_of[:10])
    doc_ids = {f["document_id"] for f in retrieve_filings(permno, as_of_d)}
    feat_ids = {c["feature_id"] for c in get_characteristics(permno, as_of_d)}
    return doc_ids, feat_ids


def _has_base_signal() -> tuple[bool, str]:
    if os.environ.get("BASE_SIGNALS_PATH"):
        return True, f"BASE_SIGNALS_PATH={os.environ['BASE_SIGNALS_PATH']}"
    if os.environ.get("PORTFOLIO_DEMO_BASELINE", "").lower() in ("1", "true", "yes"):
        return True, "PORTFOLIO_DEMO_BASELINE"
    return False, ""


def _run_signal_round(
    signal_agent: SignalAgent,
    audit: ImmutableAuditLog,
    as_of: str,
    permnos: list[int],
    round_no: int,
    revision_requests: list[str],
    kept_characteristics: list[dict] | None,
    kept_event_types: list[dict] | None,
) -> list[dict]:
    """One pass of the Signal Agent across all permnos. Returns flat signals."""
    out: list[dict] = []
    for permno in permnos:
        raw = signal_agent.run({
            "permno": permno,
            "as_of_date": as_of,
            "revision_round": round_no,
            "revision_requests": revision_requests,
            "kept_characteristics": kept_characteristics,
            "kept_event_types": kept_event_types,
        })
        audit.log("signal_agent", f"emit_round_{round_no}", raw)

        if DEBUG:
            print(f"\n[DEBUG] Raw LLM output permno={permno} round={round_no}:")
            print(json.dumps(raw, indent=2, default=str)[:3000])

        doc_ids, feat_ids = _retrieved_ids(permno, as_of)
        batch = validate_signal_batch(
            raw, date.fromisoformat(as_of[:10]), doc_ids, feat_ids,
        )
        audit.log("guardrail", f"validate_round_{round_no}",
                  batch.model_dump(mode="json"))
        out.extend(batch.model_dump(mode="json")["signals"])
    return out


def _bundle_revision(challenges: list[dict]) -> tuple[list[str], list[dict], list[dict]]:
    """Aggregate Devil's revise verdicts into the next round's context."""
    requests: list[str] = []
    kept_chars: dict[str, float] = {}
    kept_events: dict[str, float] = {}
    for c in challenges:
        if c.get("recommendation") != "revise":
            continue
        requests.extend(c.get("revision_requests") or [])
        for k in c.get("kept_characteristics") or []:
            fid = str(k.get("feature_id"))
            kept_chars[fid] = max(kept_chars.get(fid, 0.0), float(k.get("score", 0.0)))
        for k in c.get("kept_event_types") or []:
            code = str(k.get("item_code"))
            kept_events[code] = max(kept_events.get(code, 0.0), float(k.get("score", 0.0)))
    kc = [{"feature_id": f, "score": s}
          for f, s in sorted(kept_chars.items(), key=lambda kv: -kv[1])][:7]
    ke = [{"item_code": c, "score": s}
          for c, s in sorted(kept_events.items(), key=lambda kv: -kv[1])][:3]
    return requests, kc, ke


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
def run_pipeline(as_of: str, permnos: list[int]) -> dict:
    audit = ImmutableAuditLog(settings.AUDIT_DB)

    ok, source = _has_base_signal()
    if not ok:
        raise SystemExit(
            "No base signal configured. Set PORTFOLIO_DEMO_BASELINE=1 (dev) or "
            "BASE_SIGNALS_PATH=data/base_signals.parquet (prod) in .env"
        )
    print(f"base signal source: {source}")

    signal_agent = SignalAgent()
    devil_agent = DevilsAdvocate()
    portfolio_agent = PortfolioReconstructionAgent()

    # ---- 1+2. Signal <-> Devil revision loop ----
    print(f"\n[1+2/4] Signal Agent <-> Devil's Advocate "
          f"(up to {settings.MAX_REVISION_ROUNDS} revision rounds)")

    revision_requests: list[str] = []
    kept_characteristics: list[dict] | None = None
    kept_event_types: list[dict] | None = None
    round_no = 0
    all_signals: list[dict] = []
    challenges: list[dict] = []

    while True:
        print(f"\n  --- round {round_no} ---")
        round_signals = _run_signal_round(
            signal_agent, audit, as_of, permnos, round_no,
            revision_requests, kept_characteristics, kept_event_types,
        )
        validated = {"as_of_date": as_of, "signals": round_signals, "audit_trail": []}
        da_out = devil_agent.run(validated, as_of)
        audit.log("devil_agent", f"challenge_round_{round_no}", da_out)
        round_challenges = da_out.get("challenges", [])

        n_proceed = sum(1 for c in round_challenges if c.get("recommendation") == "proceed")
        n_revise = sum(1 for c in round_challenges if c.get("recommendation") == "revise")
        n_reject = sum(1 for c in round_challenges if c.get("recommendation") == "reject")
        print(f"    signals={len(round_signals)}  challenges={len(round_challenges)}  "
              f"proceed={n_proceed} revise={n_revise} reject={n_reject}")

        all_signals = round_signals
        challenges = round_challenges

        if n_revise == 0 or round_no >= settings.MAX_REVISION_ROUNDS:
            break

        revision_requests, kept_characteristics, kept_event_types = \
            _bundle_revision(round_challenges)
        print(f"    revision requests: {len(revision_requests)}  "
              f"kept_chars={len(kept_characteristics)}  "
              f"kept_events={len(kept_event_types)}")
        round_no += 1

    validated = {"as_of_date": as_of, "signals": all_signals, "audit_trail": []}

    # ---- signal diagnostics ----
    if all_signals:
        src_counts = Counter(s["source"] for s in all_signals)
        print(f"\n  source breakdown: {dict(src_counts)}")

    vetoed = {c["permno"] for c in challenges if c.get("recommendation") == "reject"}
    long_signals = [s for s in all_signals
                    if s["direction"] == "long" and s["permno"] not in vetoed]
    short_signals = [s for s in all_signals
                     if s["direction"] == "short" and s["permno"] not in vetoed]
    neutral_signals = [s for s in all_signals if s["direction"] == "neutral"]

    print(f"\n  post-veto signal balance:")
    print(f"    longs:   {len(long_signals)}")
    print(f"    shorts:  {len(short_signals)}")
    print(f"    neutral: {len(neutral_signals)}")
    print(f"    vetoed:  {len(vetoed)}")

    if len(long_signals) + len(short_signals) < 20:
        print("    WARNING: very few LLM signals survive; base signal will dominate.")

    if long_signals and short_signals:
        ratio = max(len(long_signals), len(short_signals)) / min(len(long_signals), len(short_signals))
        if ratio > MAX_IMBALANCE:
            print(f"    WARNING: long/short imbalance ratio = {ratio:.1f}x")
            # trim the over-represented side to the cap
            if len(long_signals) > len(short_signals):
                keep = int(MAX_IMBALANCE * len(short_signals))
                long_signals = sorted(long_signals, key=lambda s: -abs(s["score"]))[:keep]
                print(f"    trimmed longs to {keep}")
            else:
                keep = int(MAX_IMBALANCE * len(long_signals))
                short_signals = sorted(short_signals, key=lambda s: -abs(s["score"]))[:keep]
                print(f"    trimmed shorts to {keep}")
            kept = {s["permno"] for s in long_signals} | {s["permno"] for s in short_signals}
            all_signals = [s for s in all_signals if s["permno"] in kept]
            validated = {"as_of_date": as_of, "signals": all_signals, "audit_trail": []}

    # ---- 3. Portfolio Agent ----
    print(f"\n[3/4] Portfolio Reconstruction Agent")
    target = portfolio_agent.run(
        validated_signals=validated,
        challenges=challenges,
        as_of_date=as_of,
    )
    audit.log("portfolio_agent", "target", target)

    weights = target.get("target_weights", [])
    if not weights:
        print("  portfolio agent returned no weights.")
        return {"portfolio_target": target, "signals": all_signals,
                "challenges": challenges, "violations": []}

    longs = [w for w in weights if w["weight"] > 0]
    shorts = [w for w in weights if w["weight"] < 0]
    gross = sum(abs(w["weight"]) for w in weights)
    net = sum(w["weight"] for w in weights)
    print(f"  positions: {len(weights)}  "
          f"(longs={len(longs)} shorts={len(shorts)})")
    print(f"  gross={gross:.3f}  net={net:+.3f}")

    diag = target.get("constraint_report", {}).get("diagnostics", {})
    if diag:
        print(f"  diagnostics: {json.dumps(diag, default=str)[:600]}")

    # ---- 4. Policy check ----
    print(f"\n[4/4] Policy check")
    try:
        positions = [
            Position(permno=w["permno"], ticker=w["ticker"], weight=w["weight"])
            for w in weights
        ]
        violations = check_constraints(positions, target.get("sectors", {}))
    except Exception as exc:
        violations = [f"policy check error: {exc}"]
    audit.log("policy", "check", {"violations": violations})
    print(f"  violations: {violations or 'none'}")

    if target.get("human_approval_required"):
        print("\n  HUMAN APPROVAL REQUIRED:")
        for r in target.get("approval_reasons", []):
            print(f"    - {r}")

    print("\ntop 5 longs:")
    for w in sorted(longs, key=lambda x: -x["weight"])[:5]:
        print(f"  {w['ticker']:<8} {w['weight']:+.4f}")
    print("top 5 shorts:")
    for w in sorted(shorts, key=lambda x: x["weight"])[:5]:
        print(f"  {w['ticker']:<8} {w['weight']:+.4f}")

    return {
        "portfolio_target": target,
        "signals": all_signals,
        "challenges": challenges,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full agent pipeline")
    parser.add_argument("--as-of", default="2024-06-30",
                        help="month-end date (day >= 28); default 2024-06-30")
    parser.add_argument("--permnos", type=int, nargs="*", default=None,
                        help="explicit permnos; default auto-picks")
    parser.add_argument("--n", type=int, default=5,
                        help="how many permnos to auto-pick (default 5)")
    args = parser.parse_args()

    eom = _validate_as_of(args.as_of)
    permnos = args.permnos or _pick_test_permnos(args.n)
    if not permnos:
        raise SystemExit("No permnos to run. Check that data files are present.")

    print(f"as_of={eom.date()}  permnos={permnos}")

    run_pipeline(args.as_of, permnos)

    print(f"\naudit log: {settings.AUDIT_DB}")
    print('inspect: sqlite3 audit.db "SELECT seq, agent, step FROM audit ORDER BY seq;"')


if __name__ == "__main__":
    main()