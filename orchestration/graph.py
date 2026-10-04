from __future__ import annotations
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from config import settings

MAX_REVISION_ROUNDS = settings.MAX_REVISION_ROUNDS


class PipelineState(TypedDict, total=False):
    as_of_date: str
    permno: int
    retrieved_doc_ids: Annotated[set, operator.or_]
    retrieved_feature_ids: Annotated[set, operator.or_]
    raw_signals: dict
    validated_signals: dict
    challenges: list
    portfolio_target: dict
    audit_hashes: Annotated[list, operator.add]
    revision_round: int
    revision_requests: list
    kept_characteristics: list
    kept_event_types: list
    halted: bool


def build_graph(signal_agent, devil_agent, portfolio_agent, audit, kill_switch):
    def node_signal(state):
        if kill_switch.is_halted():
            return {"halted": True}
        out = signal_agent.run(state)
        audit.log("signal_agent", "emit", out)
        return {"raw_signals": out}

    def node_validate_signals(state):
        from guardrails.validators import validate_signal_batch
        from schemas.models import SignalBatch
        from datetime import date

        try:
            as_of = date.fromisoformat(state["as_of_date"])
        except Exception as exc:
            audit.log("guardrail", "bad_as_of", {"error": str(exc)[:200]})
            return {"halted": True}

        retrieved_doc_ids = set(state.get("retrieved_doc_ids") or [])
        retrieved_feature_ids = set(state.get("retrieved_feature_ids") or [])

        try:
            validated = validate_signal_batch(
                state.get("raw_signals") or {},
                as_of,
                retrieved_doc_ids,
                retrieved_feature_ids,
            )
            audit.log("guardrail", "validate_signals",
                      {"n_signals": len(validated.signals)})
            return {"validated_signals": validated.model_dump(mode="json")}
        except Exception as exc:
            # Should not happen — validate_signal_batch is fail-soft — but
            # belt-and-suspenders so the pipeline never dies here.
            audit.log("guardrail", "validate_signals_crashed",
                      {"error": str(exc)[:400]})
            empty = SignalBatch(as_of_date=as_of, signals=[], audit_trail=[])
            return {"validated_signals": empty.model_dump(mode="json")}

    def node_devil(state):
        if state.get("halted"):
            return {}
        out = devil_agent.run(state["validated_signals"], state["as_of_date"])
        audit.log("devil_agent", "challenge",
                  {"n_challenges": len(out.get("challenges", []))})
        return {"challenges": out["challenges"]}

    def node_collect_revision_requests(state):
        """Aggregate Devil's per-signal bundles for the next Signal pass."""
        requests: list[str] = []
        kept_chars: dict[str, float] = {}
        kept_events: dict[str, float] = {}

        for c in state.get("challenges") or []:
            if c.get("recommendation") != "revise":
                continue
            requests.extend(c.get("revision_requests") or [])
            for k in c.get("kept_characteristics") or []:
                fid = str(k.get("feature_id"))
                kept_chars[fid] = max(kept_chars.get(fid, 0.0),
                                      float(k.get("score", 0.0)))
            for k in c.get("kept_event_types") or []:
                code = str(k.get("item_code"))
                kept_events[code] = max(kept_events.get(code, 0.0),
                                        float(k.get("score", 0.0)))

        kept_chars_list = [
            {"feature_id": fid, "score": s}
            for fid, s in sorted(kept_chars.items(), key=lambda kv: -kv[1])
        ][:7]
        kept_events_list = [
            {"item_code": code, "score": s}
            for code, s in sorted(kept_events.items(), key=lambda kv: -kv[1])
        ][:3]

        audit.log("orchestrator", "revision_bundle", {
            "round": state.get("revision_round", 0),
            "n_requests": len(requests),
            "kept_characteristics": kept_chars_list,
            "kept_event_types": kept_events_list,
        })

        return {
            "revision_requests": requests,
            "kept_characteristics": kept_chars_list,
            "kept_event_types": kept_events_list,
            "revision_round": state.get("revision_round", 0) + 1,
        }

    def route_after_devil(state) -> str:
        round_now = int(state.get("revision_round", 0))
        if round_now >= MAX_REVISION_ROUNDS:
            return "portfolio"
        if any(c.get("recommendation") == "revise"
               for c in state.get("challenges") or []):
            return "collect_revision"
        return "portfolio"

    def node_portfolio(state):
        if state.get("halted"):
            return {}
        out = portfolio_agent.run(state["validated_signals"],
                                  state.get("challenges") or [],
                                  state["as_of_date"])
        audit.log("portfolio_agent", "target", out)
        return {"portfolio_target": out}

    def node_policy(state):
        from guardrails.policy import check_constraints
        from schemas.models import Position

        target = state.get("portfolio_target") or {}
        positions = [Position(**p) for p in target.get("target_weights", [])]
        sectors = target.get("sectors", {})
        violations = check_constraints(positions, sectors)
        audit.log("policy", "check", {"violations": violations})
        return {"portfolio_target": {**target, "violations": violations}}

    # ---------------- graph ----------------
    g = StateGraph(PipelineState)
    g.add_node("signal", node_signal)
    g.add_node("validate", node_validate_signals)
    g.add_node("devil", node_devil)
    g.add_node("revise_signal", node_collect_revision_requests)
    g.add_node("portfolio", node_portfolio)
    g.add_node("policy", node_policy)

    g.set_entry_point("signal")
    g.add_edge("signal", "validate")
    g.add_edge("validate", "devil")
    g.add_conditional_edges("devil", route_after_devil, {
        "revise_signal": "revise_signal",
        "portfolio": "portfolio",
    })
    g.add_edge("revise_signal", "signal")
    g.add_edge("portfolio", "policy")
    g.add_edge("policy", END)

    return g.compile()