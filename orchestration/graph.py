from __future__ import annotations
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from config import settings

MAX_REVISION_ROUNDS = settings.MAX_REVISION_ROUNDS


class PipelineState(TypedDict):
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
    halted: bool


def build_graph(signal_agent, devil_agent, portfolio_agent, audit, kill_switch):

    # ---------------- nodes ----------------
    def node_signal(state):
        if kill_switch.is_halted():
            return {"halted": True}
        # merge revision requests into the payload on re-entry
        out = signal_agent.run({
            **state,
            "revision_requests": state.get("revision_requests", []),
            "revision_round": state.get("revision_round", 0),
        })
        audit.log("signal_agent", "emit", out)
        return {"raw_signals": out}

    def node_validate_signals(state):
        from guardrails.validators import validate_signal_batch
        from datetime import date
        # build the retrieved-ID sets from the payload the agent saw
        retrieved_doc_ids = {
            f["document_id"] for f in state["raw_signals"].get("_retrieved_filings", [])
        } if "_retrieved_filings" in state["raw_signals"] else set(state["retrieved_doc_ids"])
        retrieved_feature_ids = {
            c["feature_id"] for c in state["raw_signals"].get("_retrieved_chars", [])
        } if "_retrieved_chars" in state["raw_signals"] else set(state["retrieved_feature_ids"])

        validated = validate_signal_batch(
            state["raw_signals"],
            date.fromisoformat(state["as_of_date"]),
            retrieved_doc_ids,
            retrieved_feature_ids,
        )
        audit.log("guardrail", "validate_signals", validated.model_dump())
        return {
            "validated_signals": validated.model_dump(),
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_feature_ids": retrieved_feature_ids,
        }

    def node_devil(state):
        if state.get("halted"):
            return {}
        out = devil_agent.run(state["validated_signals"], state["as_of_date"])
        audit.log("devil_agent", "challenge", out)
        return {"challenges": out["challenges"]}

    # ---- the argumentation router ----------------------------------------
    def route_after_devil(state) -> str:
        """If any challenge says 'revise' and we haven't exhausted rounds,
        bounce back to signal_agent with the revision requests."""
        round_now = state.get("revision_round", 0)
        if round_now >= MAX_REVISION_ROUNDS:
            return "portfolio"
        needs_revision = any(
            c.get("recommendation") == "revise" for c in state["challenges"]
        )
        if needs_revision:
            return "revise_signal"
        return "portfolio"

    def node_collect_revision_requests(state):
        """Extract Devil's requests and pass them back to the Signal Agent."""
        requests = []
        for c in state["challenges"]:
            if c.get("recommendation") == "revise":
                requests.append({
                    "permno": c["permno"],
                    "signal_id": c["signal_id"],
                    "points": c.get("revision_requests", []),
                })
        audit.log("orchestrator", "revision_requested",
                  {"round": state.get("revision_round", 0), "requests": requests})
        return {
            "revision_requests": requests,
            "revision_round": state.get("revision_round", 0) + 1,
        }
    # ----------------------------------------------------------------------

    def node_portfolio(state):
        if state.get("halted"):
            return {}
        out = portfolio_agent.run(state["validated_signals"], state["challenges"],
                                  state["as_of_date"])
        audit.log("portfolio_agent", "target", out)
        return {"portfolio_target": out}

    def node_policy(state):
        from guardrails.policy import check_constraints
        from schemas.models import Position
        positions = [Position(**p) for p in state["portfolio_target"]["target_weights"]]
        sectors = state["portfolio_target"].get("sectors", {})
        violations = check_constraints(positions, sectors)
        audit.log("policy", "check", {"violations": violations})
        return {"portfolio_target": {**state["portfolio_target"], "violations": violations}}

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