from schemas.models import Position

HARD = dict(
    n_min=100, n_max=500,
    gross_max=2.00, net_min=-0.50, net_max=0.50,
    per_name_max=0.05, sector_net_max=0.10,
)


def check_constraints(weights: list[Position], sectors: dict[int, str]) -> list[str]:
    violations = []
    n = len(weights)
    if not (HARD["n_min"] <= n <= HARD["n_max"]):
        violations.append(f"position count {n} outside [100,500]")

    gross = sum(abs(w.weight) for w in weights)
    if gross > HARD["gross_max"] + 1e-9:
        violations.append(f"gross {gross:.3f} > 2.00")

    net = sum(w.weight for w in weights)
    if not (HARD["net_min"] - 1e-9 <= net <= HARD["net_max"] + 1e-9):
        violations.append(f"net {net:.3f} outside [-0.5,0.5]")

    for w in weights:
        if abs(w.weight) > HARD["per_name_max"] + 1e-9:
            violations.append(f"{w.ticker} weight {w.weight:.3f} > 5%")

    sector_net: dict[str, float] = {}
    for w in weights:
        s = sectors.get(w.permno, "UNKNOWN")
        sector_net[s] = sector_net.get(s, 0.0) + w.weight
    for s, v in sector_net.items():
        if abs(v) > HARD["sector_net_max"] + 1e-9:
            violations.append(f"sector {s} net {v:.3f} > 10%")

    return violations