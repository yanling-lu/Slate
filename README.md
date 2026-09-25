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

## Outputs

- `holdings.csv`: monthly long and short stock holdings, 01/2021 to 08/2026
- `returns.csv`: monthly (or daily) total portfolio returns over the test period
- Exposure report: average and maximum gross exposure, average and extreme net exposure, realised beta to the S&P 500

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

## License

See `LICENSE`.
