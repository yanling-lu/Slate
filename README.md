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

The hurdle is a time series (FRED TB3MS/100/12 + 0.04/12 each month), not a constant. Fetched from FRED and checked complete for all 68 holding months:

| Year | Avg T-bill rate | Avg annual hurdle | Compounded over the year |
|---|---|---|---|
| 2021 | 0.04% | 4.04% | 4.12% |
| 2022 | 2.02% | 6.02% | 6.19% |
| 2023 | 5.07% | 9.07% | 9.45% |
| 2024 | 4.97% | 8.97% | 9.34% |
| 2025 | 4.07% | 8.07% | 8.37% |
| 2026 (Jan-Aug) | 3.64% | 7.64% | 5.21% (7.91% annualised) |

Full period: 7.28% average annual hurdle, 7.53% compounded annual rate, 50.9% cumulative.

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
| Portfolio Construction Agent (`portfolio_agent/`, `agents/portfolio.py`) | Built and tested (19 engine and 16 adapter tests). Deterministic, no LLM. Accepts the pipeline's Signal and Challenge schemas. |
| Dense base signal (`base_signal/`) | Built. Walk-forward ridge on the characteristics (shortlist 20, then 7, mixtures chosen on validation), tradable universe only. Passes the look-ahead audit and truncation test. Weak out of sample: test rank IC 0.037, OOS R2 -0.11%. |
| Look-ahead audit and walk-forward spec (`portfolio_agent/audit.py`, `SIGNALS_SPEC.md`) | Built. Checks contract, coverage, target-filtered universe, rank-IC leak detection, manifest schedule, and a truncation test. |
| Deck report pack (`report/`) | Built. 10 charts and the required tables, with live FRED benchmark and S&P 500 data. |
| Composite factor model (`base_signal/composites.py`) | New. Value, profitability and momentum composites with signs fixed from theory; family weights chosen on validation only; a family is kept only if it is stable across validation sub-periods and positive after turnover costs. Passes the audit and truncation test. |
| 8-K sentiment agent (`agents/eightk_signal.py`) | New. An OpenAI model scores the sentiment of each material 8-K filed in the month; the scores become a long / short / neutral signal per stock in the repo's `Signal` schema. Tested with a fake LLM; not yet run against the API (needs a key and a call budget). |
| Signals Agent (`agents/signal.py`, `tools/`) | Code on `main` (Lawrence). Needs an OpenAI key and the prompts in `.env`. Not yet run end to end. The graph handles one stock at a time, so a loop over stocks and months is still needed. |
| Devil's Advocate Agent (`agents/devils.py`) | Maria's PR #4 is open (adds the agent, objection score 0-3, top-7 characteristics and top-3 event types). Not on `main` yet; `agents/devils.py` on `main` is a stub. |

### Current results

Base signal only (no LLM signals, no Devil's Advocate), 01/2021 to 08/2026, gross of trading costs, real TB3MS and S&P 500 from FRED. These are not final results.

| Metric | Strategy | Cash + 4% | S&P 500 (price, context only) |
|---|---|---|---|
| CAGR | 0.88% | 7.53% | 13.47% |
| Max drawdown | -34.3% | 0% | -24.8% |
| Information ratio vs cash + 4% | -0.29 | | |
| Sharpe | -0.06 | | |
| Alpha / t-stat (Newey-West) | 1.4% / 0.2 | | |
| Beta / SE (Newey-West) | -0.23 / 0.16 | | |

Gross exposure 190% and net 0% every month; one-way turnover at the 30% cap. The long book contributed about +43% of NAV and the short book about -48%: the shorts (speculative names that rallied) are the problem. A 147-feature ensemble was tried and was not stronger (IC 0.015), so the 7-feature model is the current base.

Composite model, same conditions (gross of costs, real TB3MS and S&P 500; not final results):

| Metric | Composite model | 7-feature model |
|---|---|---|
| Rank IC (68 months) | 0.053 | 0.037 |
| CAGR | 11.1% | 0.9% |
| Sharpe | 0.47 | -0.06 |
| Information ratio vs cash + 4% | +0.26 | -0.29 |
| Max drawdown | -26.5% | -34.3% |
| Beta / SE (Newey-West) | **-0.34** / 0.17 | -0.23 / 0.16 |

Calendar years, composite vs hurdle: 2021 +63.8% vs 4.1%; 2022 +38.7% vs 6.2%; 2023 -6.7% vs 9.5%; 2024 +1.2% vs 9.3%; 2025 -2.3% vs 8.4%; 2026 (8 months) -13.2% vs 5.2%. **The gain is concentrated in 2021-2022; the model is negative after that.** The realised beta (-0.34, t about -2) is well away from zero even though the optimiser holds ex-ante beta within +-0.05, which the challenge reads as a neutrality failure.

#### Why the beta is negative

It is a style effect, not stock betas. The long stocks' realised beta is +1.02 and the short stocks' +1.38, so the dollar-neutral book is net short about 0.34 of market beta. The short side is the more volatile one (1.6x the volatility of the long side), 29% of it has no `beta_60m` (long side: 6%) and is filled with a median, and the slow `beta_60m` understates how these names move with the market. Adding a high-minus-low volatility factor (built from the panel) to the regression takes the market beta to +0.12 (SE 0.12) and the book loads -0.73 on the factor (monthly correlation -0.74). That factor fell 21% and 33% in 2021-2022, when the book gained 64% and 36%, and rose in 2023, 2025 and 2026, when the book lost. The composite is in effect a short-volatility bet that paid in 2021-2022 and has lost since, and about 60% of its headline alpha (12.7% to 5.3% a year) is explained by that exposure. Untested fixes: a volatility-neutral constraint in the optimiser, a faster beta estimate, and imputing missing betas by volatility bucket instead of the overall median.

### Composite factor model (feature selection)

`python -m base_signal.composites --panel "Provided Documents/chars_final_with_names.parquet" --out out_composite`

1. **Families:** value (`be_me`, `ni_me`, `ocf_me`, `fcf_me`, `sale_me`, `ebitda_mev`, `bev_mev`), profitability (`gp_at`, `op_at`, `cop_at`, `ope_be`, `ni_be`, `niq_be`, `f_score`, `qmj_prof`, `ebit_sale`, `ocf_at`) and momentum (`ret_12_1`, `ret_9_1`, `ret_6_1`, `ret_12_7`, `resff3_12_1`, `resff3_6_1`). Optional `--extra low_risk investment`.
2. **Signs fixed from theory** (higher profitability is better, higher volatility worse). Nothing is estimated per feature, so there is no yearly sign flipping.
3. **One composite per family:** average of the members' within-month ranks, re-ranked.
4. **Weights across families** from a coarse grid (step 0.25), scored on validation months only, then the test year is scored once.
5. **Stability filter on validation:** keep a family only if its rank IC is positive in at least 3 of 4 sub-periods and its top-minus-bottom decile spread is positive after turnover costs (10 bps per dollar traded, an assumption: the challenge supplies no cost model). If no family passes, equal weights are used and the manifest says so.

Each stock's rationale in the signal file lists its family composites. The manifest records every decision per year.

### 8-K sentiment agent

`python -m agents.eightk_signal --filings "Provided Documents/8k_20150101_20260831_identified.parquet" --estimate` sizes a run without any API call. A real run needs `OPENAI_API_KEY`, `--max-calls`, and writes `signals_8k.csv` (directional signals only, source `signal_agent`) plus `signals_8k_all.csv` and `eightk_manifest.json`. Feed it to the portfolio next to the base signal: `--signals out_composite/signals.csv out_8k/signals_8k.csv`.

- **Inputs and cutoff:** filings dated in the formation month and on or before month-end only (the challenge: accumulate the month's news to forecast the next month). Only material items (1.01, 1.02, 2.01, 2.02, 2.05, 2.06, 4.01, 4.02, 5.02) are sent; the rest are skipped before any LLM call. `--include-other-items` adds 3.01, 7.01, 8.01.
- **Signal:** `sentiment` in [-3, +3]; direction long at >= +0.5, short at <= -0.5, otherwise neutral; `score` = sentiment x 2/3; `source = "filing_8k"`. Several filings in a month are combined by confidence-weighted mean, and conflicting filings cancel to neutral.
- **Guardrails:** company name, ticker and dates redacted before the model sees the text; extraction-only prompt; the quote must appear verbatim in the redacted filing or the output is discarded; prompt-injection patterns cap the confidence; one call per filing at temperature 0, cached, with a hard call budget; parallel workers (`--workers`).
- **Recall check:** `recall_split_ic` compares the signal's IC before and after the model's training cutoff. A signal that is strong before and fades after is recalling, not predicting.
- **Size of a real run** (material items, formation months 2020-12 to 2026-07): 135,778 filings, about 275M input and 16M output tokens, 107,559 stock-months. Multiply by your model's price. Choosing the 8-K signal's weight on validation would need the 2015-2020 filings scored as well.

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

Run everything from the repo root, in this order. **Each step needs the file the step before it creates:**
generated files (the `out_*/` and `output*/` folders) are not stored in git, so a fresh clone has none of them.

```bash
pip install -r requirements.txt
python -m pytest tests                # and: python -m unittest discover -s portfolio_agent/tests -t .

# 1. create the signal file  ->  out_composite/signals.csv + manifest.json   (about 15 seconds)
python -m base_signal.composites --panel "Provided Documents/chars_final_with_names.parquet" --out out_composite

# 2. look-ahead audit of the signal and its manifest
python -m portfolio_agent.audit --panel "Provided Documents/chars_final_with_names.parquet" \
    --signals out_composite/signals.csv --manifest out_composite/manifest.json

# 3. build the portfolio and backtest  ->  output_run/ (holdings.csv, portfolio_returns.csv, ...)
python -m portfolio_agent.run --panel "Provided Documents/chars_final_with_names.parquet" \
    --signals out_composite/signals.csv --out output_run

# 4. write the charts and tables  ->  results/composite/ (prints a one-line headline)
python -m report.build --run-dir output_run --out results/composite
```

Everything the code generates is CSV or JSON. Only the challenge's own data files (`Provided Documents/*.parquet`) are parquet, and they are not stored in this repo: place them in `Provided Documents/`. The older 7-feature model is `python -m base_signal.model --panel ... --out out_base`. `--demo-baseline` runs with a fixed momentum/quality/value placeholder instead of `--signals`; add `--da` for Devil's Advocate objections, and `--tb3ms` / `--sp500` to use your own benchmark files. To combine signals, list several files after `--signals`.

**Troubleshooting**

| Message | Cause and fix |
|---|---|
| `Signals file not found: out_composite/signals.csv` | Step 1 has not been run on this machine. Run it first. |
| `OSError: Repetition level histogram size mismatch` or `Cannot read ... with pyarrow 19` | An old `pyarrow` cannot read the challenge's parquet files. `pip install -U "pyarrow>=25"` (what `requirements.txt` pins). If `fastparquet` is installed the code falls back to it automatically. |
| `Data file not found: Provided Documents/...` | The challenge data is not in git. Put it in `Provided Documents/` or pass the full path with `--panel`. |

## Deck report pack (`report/`)

```bash
python -m portfolio_agent.run --panel "Provided Documents/chars_final_with_names.parquet" --signals out_base/signals.csv --out output_base
python -m report.build --run-dir output_base --out report_out     # downloads TB3MS and S&P 500 from FRED (no key)
```

Writes 10 charts, the required statistics tables, the top-10 holdings and P&L contributors (ticker and company name, with a flag
for labels that need verification) and `methodology.md`. The S&P 500 appears only as grey context, as the challenge requires; the
benchmark is TB3MS + 4%. Needs only pandas, numpy and matplotlib: it reads the run's CSVs, not the parquet files.

## Outputs

- `holdings.csv`: monthly long and short stock holdings (Date, PERMNO, TICKER, COMPANY NAME, WEIGHT), 01/2021 to 08/2026
- `portfolio_returns.csv`: monthly total portfolio returns over the test period, with benchmark, active return and market excess return
- `guardrail_report.csv` and `guardrail_checks.csv`: per-month gross, net, names, turnover, ex-ante beta and pass/fail for every check
- `trade_log.csv`: the reason for every trade
- `position_pnl.csv`: every held position with weight, realised return, P&L and formation-month size and liquidity (used by the report)
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

| Item | Status |
|---|---|
| Deck with appendix, one PDF (8 pages plus up to 10 appendix pages) | Cathy. Charts and tables come from `report/`. |
| Monthly holdings CSV (Date, PERMNO, TICKER, COMPANY NAME, WEIGHT), 01/2021-08/2026 | Produced by `portfolio_agent.run` as `holdings.csv`. Regenerate from the final signal. |
| Portfolio returns CSV for the test period | Produced as `portfolio_returns.csv`. |
| One Python file called `MAIN.py` describing the process (do not submit the data) | **Not yet written.** `main.py` is the LangGraph demo, not the full pipeline. |
| Licence note, team CVs | Public domain per the challenge rules; CVs: Cathy. |
| Hard-to-verify labels | `labels_to_verify.csv` lists holdings whose ticker or name is not verified in-month. Check these before they appear in the deck. |

### Known gaps

- The 8-K sentiment agent has not been run against the API; its effect on results is unmeasured.
- The composite model's realised beta (-0.34 vs the S&P 500) and its fade after 2022 are a style effect (see "Why the beta is negative"); neutralising volatility explicitly is untested.
- Two tests fail on `main` since the `Signal` schema change (`sentiment` is now required): `tests/test_pipeline.py::test_validator_rejects_lookahead` (the validator no longer raises) and `tests/test_schemas.py::test_characteristic_only_signal_ok`.
- No LLM signal has been run: the Signals Agent needs keys and prompts, and a loop over stocks and months.
- The Devil's Advocate is in PR #4 and not yet wired to produce results.
- The final numbers in the deck currently come from the weak base signal only.
- Daily-frequency measurement is not done; all figures are monthly.

## AI Use Disclosure

- **Code:** The portfolio construction agent (`portfolio_agent/`, `agents/portfolio.py`), the dense base signal (`base_signal/`), the look-ahead audit, the report module (`report/`), the tool month-key fix, their tests and documentation were written with Claude Code (Anthropic's Claude models), directed and reviewed by the team member who owns them. The composite factor model (`base_signal/composites.py`) and the 8-K sentiment agent (`agents/eightk_signal.py`) were also written with Claude Code. Other components: owners should add their own AI-tool use here.
- **LLM at run time:** only the 8-K sentiment agent (default `gpt-4o-mini`, temperature 0, one call per material filing, prompt in `agents/eightk_signal.py`) and the Signals / Devil's Advocate agents call an OpenAI model. The composite model is rule-based.
- **No LLM at run time in the portfolio agent:** It is deterministic (a linear program plus rule-based checks). It makes no model calls and never reads forward returns while forming weights, so it has no model-side look-ahead. Any LLM use in the Signals and Devil's Advocate agents is documented by their owners, with prompts and tool definitions in the deck appendix.
- **Verification:** Every AI-written claim about performance is traceable to the backtest outputs. Backtest results are not evidence of future performance, and nothing in this repo is investment advice.
- **Placeholders:** Results produced with `--demo-baseline` or without `--tb3ms` / `--sp500` use a placeholder signal and proxy benchmark or market series, and the output files label them as such. They are not the team's reported results.

## License

Per the challenge rules, the submission is public domain / free to use. No separate `LICENSE` file is required.
