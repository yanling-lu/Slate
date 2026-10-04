from config import settings
from audit.logger import ImmutableAuditLog
from orchestration.graph import build_graph
from agents.signal import SignalAgent
from agents.devils import DevilsAdvocate
from agents.portfolio import PortfolioReconstructionAgent


class KillSwitch:
    def __init__(self, audit: ImmutableAuditLog):
        self._halted = settings.SYSTEM_HALT
        self._audit = audit

    def is_halted(self) -> bool:
        return self._halted

    def halt(self, reason: str):
        self._halted = True
        self._audit.log("kill_switch", "halt", {"reason": reason})


audit = ImmutableAuditLog(settings.AUDIT_DB)
kill = KillSwitch(audit)

signal_agent = SignalAgent(model="gpt-4o")
devil_agent = DevilsAdvocate(model="gpt-4o")
portfolio_agent = PortfolioReconstructionAgent(model="gpt-4o")

graph = build_graph(signal_agent, devil_agent, portfolio_agent, audit, kill)

if __name__ == "__main__":
    state = {
        "as_of_date": "2024-06-30",
        "permno": 14593,
        "retrieved_doc_ids": set(),
        "retrieved_feature_ids": set(),
        "revision_round": 0,
        "audit_hashes": [],
    }
    result = graph.invoke(state)
    print(result["portfolio_target"])
