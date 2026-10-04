# Team Slate: Agentic Market-Neutral U.S. Equity Strategy

A multi-agent AI system that predicts next-month U.S. stock returns and builds a market-neutral long/short portfolio, tested strictly out-of-sample over 01/2021 to 08/2026.

## Objective

1. Identify the stock-level factors that best predict future returns.
2. Go long the stocks with the highest expected returns and short those with the lowest. Trades are placed at the start of each month and evaluated at month-end.
3. Backtest rigorously with no data leakage, and document every step (agent plans, tools, prompts) so a third party can reproduce the results.

## Portfolio Constraints

| Constraint | Requirement |
|---|---|
| Universe | U.S. equities |
| Positions | 100 to 500 names (long + short combined) |
| Gross exposure | ≤ 200% of capital (max 2x leverage) |
| Net exposure | −50% to +50% at every rebalance |
| Rebalancing | Monthly or quarterly |
| Neutrality | Dollar-, beta-, or sector/factor-neutral |

**Benchmark:** 3-month U.S. T-bill + 4% (~7.7% annually). Evaluated primarily on information ratio, plus alpha, drawdowns, turnover, and realised beta to the S&P 500 (target ≈ 0).

## Architecture

| Agent | Role | Owner |
|---|---|---|
| Signals Agent | Builds predictive signals from fundamentals and 8-K filings (text as data, via LLMs) | Lawrence |
| Devil's Advocate Agent | Challenges signals and positions; stress-tests for bias, leakage, and weak reasoning | Maria |
| Portfolio Construction Agent | Turns signals into long/short holdings that satisfy all constraints | Yan-Ling |

Every agent includes guardrails to enforce constraints and prevent look-ahead bias.

**Candidate fundamental factors:** value, profitability, growth and growth rate, price trend, volatility.

### Status

| Component | Status |
|---|---|
| Portfolio Construction Agent (`portfolio_agent/`) | Built and unit-tested (9 tests). Runs end to end on the provided panel with a placeholder signal. |
| Signals Agent | In progress |
| Devil's Advocate Agent | In progress. The portfolio agent already accepts its output, including the 0-3 objection score. |

### How the agents connect

Agents pass CSV or parquet files keyed on `permno` + `eom`, where `eom` is the month-end of the formation month t (information dated on or before `eom` only). Holdings are dated the 1st of month t+1.

| File | Columns |
|---|---|
| Signals | `permno, eom, score` (+ optional `source, confidence, rationale, asof_date`) |
| Devil's advocate | `permno, eom` + either `action` (`veto`/`haircut`/`pass`) and `severity`, or `objection_score` 0-3 (3 = veto, 2 = halve the bet, 0-1 = keep); optional `side, reason` |

### Pipeline integration (`agents/portfolio.py`)

`PortfolioReconstructionAgent` is the graph-facing wrapper around the engine in `portfolio_agent/`.
`run(validated_signals, challenges, as_of_date)` returns a `PortfolioTarget`-shaped dict
(`target_weights`, `sectors`, `constraint_report`, `trade_rationale`, `human_approval_required`).

| Pipeline input | How the portfolio agent uses it |
|---|---|
| `Signal` (direction, score in [-2, 2], confidence) | Becomes source `signal_agent`. Neutral and `insufficient_evidence` signals are dropped (no row = no view). Several signals for one stock merge by confidence-weighted mean. Scores are used as given, not re-standardised across the few names with a filing. The model's own `ticker` is ignored; labels come from the panel. |
| `Challenge` (`recommendation`, `resilience_score`) | `reject` = veto. `revise` = haircut of the alpha and position cap by `1 - resilience_score` (bounded 0.25-0.9). `proceed` = no change. A numeric `objection_score` 0-3 is also accepted. |
| Dense base signal (`BASE_SIGNALS_PATH`) | Required. The 8-K signal covers only stocks with a filing that month, so a 100-500 name book needs a dense cross-section from the characteristics model. `PORTFOLIO_DEMO_BASELINE=1` uses a labelled placeholder instead. |

The graph runs per stock, so signals for the same month accumulate across calls and each call returns the book built from everything seen so far. The month commits (becomes the previous book for the turnover budget) when a later month is requested. Months must be requested in order, with no gaps. `as_of_date` must be a month-end (day 28 or later), because characteristics are month-end snapshots. The agent makes no LLM calls; `model` is accepted only for interface compatibility. After the last month, `holdings_frame()`, `monthly_frame()` and `trades_frame()` give the submission CSVs.

### Portfolio construction guardrails

| Limit | Type | Value |
|---|---|---|
| Names (long + short) | Hard (mandate) | 100-500 |
| Gross exposure | Hard (mandate) | ≤ 200%; internal target 190% |
| Net exposure | Hard (mandate) | −50% to +50%; internal target 0% (dollar neutral) |
| Ex-ante beta | Policy | within ±0.05 |
| Net weight per GICS sector | Policy | within ±5% |
| Position size | Policy | ≤ 0.75% of NAV |
| One-way monthly turnover | Policy | ≤ 30% of NAV |
| Universe | Policy | price ≥ $5 and daily dollar volume ≥ $2M; shorts ≥ $5M and no micro/nano caps |

A hard-limit breach aborts the run and writes nothing. Policy limits relax one step at a time, and each step is logged, only when a month is otherwise infeasible. Inputs with forward-looking columns or information dated after `eom` are rejected. Details are in [`portfolio_agent/README.md`](portfolio_agent/README.md).

## Quick start

```bash
pip install pandas numpy scipy pyarrow
python -m unittest discover -s portfolio_agent/tests -t .
python -m portfolio_agent.run --panel "Provided Documents/chars_final_with_names.parquet" \
    --signals inputs/signals.parquet --da inputs/devils_advocate.csv \
    --tb3ms TB3MS.csv --sp500 sp500_monthly.csv --out output
```

The provided data files are not stored in this repo. Place them in `Provided Documents/`. `--demo-baseline` runs with a fixed momentum/quality/value placeholder instead of `--signals`.

## Outputs

- `holdings.csv`: monthly long and short stock holdings (Date, PERMNO, TICKER, COMPANY NAME, WEIGHT), 01/2021 to 08/2026
- `portfolio_returns.csv`: monthly total portfolio returns over the test period, with benchmark, active return and market excess return
- `guardrail_report.csv` and `guardrail_checks.csv`: per-month gross, net, names, turnover, ex-ante beta and pass/fail for every check
- `trade_log.csv`: the reason for every trade
- `summary.json`: return, risk, alpha/beta, exposure and turnover statistics
- `labels_to_verify.csv`: holdings whose ticker or company name is not verified in-month and must be checked against an authoritative source

## Data

- Provided challenge datasets (baseline predictors from the supplied CSV list)
- 8-K filings (optional text data)
- WRDS (supplementary, documented where used)

Preprocessing steps are implemented and documented in the code.

## Team

- Lawrence: Signals agent
- Maria: Devil's advocate agent
- Yan-Ling: Portfolio construction agent, subject matter expert
- Diana: Compliance and results review
- Cathy: Presentation deck and CVs

## Submission (due Oct 4, 11:59 pm)

Deck with appendix (PDF), holdings and returns CSVs, Python code, licence, team CVs.

## AI Use Disclosure

- **Code:** The portfolio construction agent (`portfolio_agent/`), its tests and its documentation were written with Claude Code (Anthropic's Claude models), directed and reviewed by the team member who owns that component. Other components: owners should add their own AI-tool use here.
- **No LLM at run time in the portfolio agent:** It is deterministic (a linear program plus rule-based checks). It makes no model calls and never reads forward returns while forming weights, so it has no model-side look-ahead. Any LLM use in the Signals and Devil's Advocate agents is documented by their owners, with prompts and tool definitions in the deck appendix.
- **Verification:** Every AI-written claim about performance is traceable to the backtest outputs. Backtest results are not evidence of future performance, and nothing in this repo is investment advice.
- **Placeholders:** Results produced with `--demo-baseline` or without `--tb3ms` / `--sp500` use a placeholder signal and proxy benchmark or market series, and the output files label them as such. They are not the team's reported results.

## License

Per the challenge rules, the submission is public domain / free to use. No separate `LICENSE` file is required.
