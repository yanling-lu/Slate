# Signals interface and walk-forward spec

What the Signals Agent and Devil's Advocate hand to the portfolio agent, and the rules
that keep it free of look-ahead. `python -m portfolio_agent.audit` enforces these.

## 1. Pipeline

```
Signals Agent: proposes up to 20 characteristics + top 5 8-K event drivers (training data only)
Devil's Advocate: cuts to 7 of the 147 characteristics + 3 8-K events, ranked 1-10
Fitting code: builds feature mixtures, fits on train, picks the best on VALIDATION, scores TEST
Portfolio agent: signals.parquet (+ optional objections) -> long/short weights
```

The portfolio agent never fits a model and never reads a forward return.

## 2. Walk-forward schedule (dates are TARGET months; from the challenge)

A formation row in month t has target `ret_exc_lead1m` in month t+1. Do not shift it again.
A January 2021 forecast uses December 2020 characteristics. `portfolio_agent/splits.py`
has `split_for_year(year)`.

| Forecast (test) year | Train targets | Validation targets | Test targets |
|---|---|---|---|
| 2021 | 2015-02 .. 2018-12 | 2019-01 .. 2020-12 | 2021-01 .. 2021-12 |
| 2022 | 2015-02 .. 2019-12 | 2020-01 .. 2021-12 | 2022-01 .. 2022-12 |
| ... | expanding | rolling 2 years | one year |
| 2026 | 2015-02 .. 2023-12 | 2024-01 .. 2025-12 | 2026-01 .. 2026-08 |

Refit once a year, forecast monthly. Labels whose target month is in the test year are
never used for fitting, even if their characteristic month is earlier.

## 3. Rules

1. **Select on validation, test once.** Feature lists and mixtures are ranked on the
   validation window only. Test-period returns are never used to choose, drop, tune or
   re-rank anything. Trying many mixtures is fine; ranking them on test data is not.
2. **Re-select each forecast year.** The 20/7/3 shortlists are rebuilt every year from
   data with target month <= that year's validation end. A single list picked once on
   2015-2026 data is look-ahead.
3. **Preprocessing is fitted on training data** (winsorising cut-offs, medians for
   missing values, standardisation, scalers). Cross-sectional ranking within a month is
   allowed because it uses only that month.
4. **Target availability must not define the universe.** Score every stock-month with
   usable characteristics, including those with no next-month return.
5. **8-K text:** use filings with `filing_date` on or before the formation month-end.
   Provider timestamps are date placeholders, so a filing dated the 3rd cannot form a
   portfolio on the 1st. `content_report_date` is not a release date.
6. **LLMs:** the model's knowledge cutoff is after 2021, so asking it about a 2021
   filing is recall, not forecasting. State the model, its cutoff, and how the agent was
   restricted to documents available at t (see `llm` in the manifest).
7. **Record the multiple-testing load** (`mixtures_tried`). It goes in the deck.

## 4. `signals.parquet` (or csv)

One row per stock-month-source.

| Column | Required | Notes |
|---|---|---|
| `permno` | yes | integer |
| `eom` | yes | formation month-end t |
| `score` | yes | higher = more bullish for month t+1; any scale (z-scored per month per source by the agent) |
| `source` | no | e.g. `ml_chars`, `8k_events`; default `signal` |
| `confidence` | no | 0-1, default 1 |
| `rationale` | no | text copied into the trade log, e.g. top 3 drivers |
| `asof_date` | no | latest input date; must be <= `eom` |
| `pred_ret` | no | predicted excess return; used for OOS R2 |

Column names containing `lead`, `fwd`, `forward`, `future`, `next_month` or `target` are rejected.
Cover 2020-12 .. 2026-07 formation months (holding months 2021-01 .. 2026-08), at least
~400 scored names a month (the agent needs 100-500 positions after screens).

## 5. `manifest.json`

```json
{
  "years": [
    {
      "year": 2021,
      "train_target_end": "2018-12-31",
      "selection_window": "validation",
      "fit_data_end": "2020-12-31",
      "feature_selection_data_end": "2018-12-31",
      "characteristics": ["ret_12_1", "be_me", "..."],
      "events": ["2.02", "5.02", "4.02"],
      "ranking": ["ret_12_1", "2.02", "be_me", "..."],
      "mixtures_tried": 12,
      "val_metric": {"name": "rank_ic", "value": 0.041},
      "oos_r2_test": 0.0012
    }
  ],
  "llm": {"model": "...", "knowledge_cutoff": "...", "lookahead_control": "...", "max_steps": 5}
}
```

One entry for each year 2021-2026. The audit fails if `selection_window` is not
`validation`, if any data end date is after that year's validation end, if more than 7
characteristics or 3 events are listed, if a characteristic is not in
`factor_char_list.csv`, or if `mixtures_tried` is missing. `llm` is required when any
LLM touches the signals.

### Declared feature sets

`manifest.years[].feature_set` defaults to a 7-feature shortlist (at most 7 characteristics and 3 events). A model that
deliberately departs from that declares it: `"all"` (all 147; `characteristics` must equal `factor_char_list.csv`) or
`"composite"` (fixed-sign family composites; every member must be in `factor_char_list.csv`). The audit then skips the
7-feature cap but still checks the schedule, the data end dates and `mixtures_tried`. The deck must say which was used.

## 6. `devils_advocate.csv` (optional per-stock objections)

`permno, eom` plus either `action` (`veto`/`haircut`/`pass`) and `severity` 0-1, or
`objection_score` 0-3 (3 = drop, 2 = halve, 0-1 = keep). Optional `side` (`long`/`short`/`both`)
and `reason`. List only stocks being objected to.

## 7. Audit

```bash
python -m portfolio_agent.audit --panel "Provided Documents/chars_final_with_names.parquet" \
    --signals inputs/signals.parquet --manifest inputs/manifest.json --da inputs/devils_advocate.csv
```

| Check | FAIL when |
|---|---|
| contract | missing columns, forbidden column names, `asof_date` > `eom`, duplicates |
| coverage | a formation month 2020-12..2026-07 has no scores, or 2026-08 is scored |
| universe_vs_target | stock-months with no next-month return are scored at under half the rate of the rest |
| ic | mean monthly rank IC against next-month excess return >= 0.20 (WARN at 0.10; real signals sit at 0.02-0.08) |
| oos_r2 | WARN if > 5% or negative (needs `pred_ret`) |
| manifest | any schedule or shortlist violation in section 5 |

**Truncation test (strongest check).** In Python:

```python
from portfolio_agent.audit import truncation_test
raw = pd.read_parquet("Provided Documents/chars_final_with_names.parquet")
truncation_test(generate_signals, raw).print()   # generate_signals(panel) -> signals df
```

It re-runs the generator on the panel cut at 2022-06 and 2024-12 (the cutoff row's
target is blanked) and requires identical scores for every month up to the cutoff. A
score that changes when future rows disappear used the future. Run it on the final
pipeline before submitting, and note any WARN with its explanation in the deck.

## 8. Limits of the audit

It catches target leakage, full-sample statistics, schedule violations and
target-filtered universes. It cannot see model-side look-ahead in an LLM or tell
whether the manifest dates are true. Those need the generator's code and a human read.
