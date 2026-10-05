# Portfolio-construction agent

Turns forecasts from the signal agent(s) and the devil's-advocate agent into monthly
long/short weights that satisfy the McGill-FIAM 2026 market-neutral mandate.

```bash
python -m portfolio_agent.run --panel "Provided Documents/chars_final_with_names.parquet" \
    --signals inputs/signals.csv --da inputs/devils_advocate.csv \
    --tb3ms TB3MS.csv --sp500 sp500_monthly.csv --out output
python -m unittest discover -s portfolio_agent/tests -t .
```

`--demo-baseline` replaces `--signals` with a fixed momentum/quality/value rank so the
pipeline runs before the other agents deliver. It is a placeholder, not the strategy.

## Inputs (see `contracts.py`)

| File | Columns |
|---|---|
| signals | `permno, eom, score` + optional `source, confidence, rationale, asof_date` |
| devil's advocate | `permno, eom, action (veto/haircut/pass)` + optional `severity, side, reason` |

`eom` is the formation month-end t. Holdings are dated the 1st of month t+1.
The devil's-advocate file may instead carry an integer `objection_score` (0-3) in place of
`action`/`severity`: 3 = veto, 2 = haircut (severity 0.5), 0-1 = pass. Giving both is an error.
Any column containing `lead/fwd/forward/future/target`, or `asof_date > eom`, is rejected.
Multiple signal sources are z-scored per month, clipped, confidence-weighted and averaged.

## Guardrails

| Limit | Mandate | Enforced by the agent |
|---|---|---|
| Names (long+short) | 100-500 | hard: run aborts if violated |
| Gross | <= 200% | hard; target 190% |
| Net | -50%..+50% | hard; target 0 (dollar neutral) |
| Ex-ante beta (`beta_60m`) | - | policy: \|beta\| <= 0.05 |
| Sector (GICS 2-digit) net | - | policy: <= 5% each |
| Per-name weight | - | policy: <= 0.75% (so >= 253 names at full gross) |
| One-way monthly turnover | - | policy: <= 30% of NAV |
| Universe | - | price >= $5, 126d $ volume >= $2M; shorts >= $5M, no micro/nano |

Soft limits relax in a logged order (sector, beta, turnover) only if a month is
infeasible; the hard mandate never relaxes. `guardrails.py` re-checks the finished book
independently of the optimizer. Devil's-advocate `veto` blocks a name on that side;
`haircut` shrinks its alpha and position cap by `1 - severity`.

## Pipeline wrapper

`agents/portfolio.py` adapts this engine to the LangGraph pipeline (Signal and Challenge
schemas in, `PortfolioTarget` dict out). See the root README, "Pipeline integration".
`Policy.fixed_scale_sources` marks sparse sources, such as the LLM signal agent, that are
used as given instead of being re-standardised across the few names they cover.

## Upstream spec and audit

`SIGNALS_SPEC.md` defines the walk-forward schedule and the signals/manifest formats.
`python -m portfolio_agent.audit` checks them for look-ahead (see the spec, section 7).

## Outputs

`holdings.csv` (submission format), `portfolio_returns.csv`, `trade_log.csv` (reason for
every trade), `guardrail_report.csv` / `guardrail_checks.csv` (per-month exposure and pass/fail),
`labels_to_verify.csv` (holdings whose ticker/name are not verified in-month; the challenge
requires checking these against an authoritative source), `summary.json`.

## Assumptions to state in the deck

- Returns are gross of trading costs unless `--tcost-bps` is set; NAV earns rf, short rebate = rf.
- Without `--tb3ms` / `--sp500`, benchmark and market are proxies and labelled as such in the outputs. Supply FRED TB3MS and S&P 500 returns for the final numbers.
- Held names with no next-month return are assumed to earn 0 excess return (58 stock-months in the demo).
- No LLM is called by this agent, and outcomes are only joined in `backtest.py` after all weights are final.

## AI disclosure

This package was written with Claude Code (Anthropic's Claude models) and reviewed by the component owner.
It makes no LLM calls at run time. Demo results use a placeholder signal and are not the team's results.
