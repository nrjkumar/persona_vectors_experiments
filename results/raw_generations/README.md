# Raw generation dumps

Model outputs kept for reference. **These are not benchmark results** — they are
unscored generations, and nothing here should be read as a score.

## `forecast_bench_unscored_pre_fix.csv`

A ForecastBench generation run that predates the current evaluation code. Kept only
as a record of what was generated; it cannot be scored and should not be used as a
baseline.

Audited state (2026-09-02):

| Check | Value |
|---|---|
| rows | 2500 (500 question_ids x 5 samples) |
| `gold` non-null | 0 / 2500 |
| `score` non-null | 0 / 2500 |
| scoring columns | none — no `predicted_probability`, `brier_score`, `forecasting_score`, `log_score` |
| structured prompts | 0 / 2500 rows contain the calibrated forecasting prompt |
| unfilled `{...}` placeholders in question text | 1250 / 2500 |
| question_ids with a single distinct answer across 5 samples | 314 / 500 |

Why none of it is usable:

- It was generated before `build_forecast_bench_prompt`, so the model saw a bare chat
  prompt rather than the structured request for a calibrated probability.
- It was generated before any probability parsing or Brier scoring existed, so the
  four scoring columns the current code emits are absent.
- No resolution set was joined, so there are no gold outcomes — the questions were
  2026-horizon and unresolved at generation time.
- Half the rows carry unsubstituted date templates, so those prompts were malformed.
- 314 of 500 questions got the same answer five times over, from greedy decoding with
  `n_per_question=5`.

Each of these is fixed in the current pipeline. See
`tests/smoke/forecastbench-eval-review.md` (local, gitignored) for the review that
produced this list. To generate something scorable, pass both a question set and its
resolution set:

```bash
bash scripts/eval_forecast_bench.sh 0 Qwen/Qwen2.5-7B-Instruct \
    <question_set>.json <resolution_set>.json results/my_run.csv
```
