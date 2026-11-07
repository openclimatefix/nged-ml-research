---
id: "00615"
slug: bakeoff-wind-and-solar-development-month-backtests
title: >-
  How do the candidate forecasts for the 3 metered wind farms and the 6 metered solar farms score on
  the 5 development months, and which should be the default for each technology?
theme: production-reference-and-best-guess
also_relevant_to:
  - generator-forecasting
status: planned
kind: other
protocol: wind-v1
date: 2026-11-07
size: >-
  Medium. Answers to the five triggers: (1) it stores Parquet exports and wind plant summaries in the
  private data folder; (2) one defensible design for the solar part (stored forecasts, re-scored) and
  one for the wind part (experiment 609's design, run on daily origins); (3) it spans code whose
  callers cannot be named without searching (experiments 259, 303, 310, 347, 430, 461, and 493);
  (4) it touches no serving path and no degradation rule; (5) it changes no contract. The final
  test is not run.
hypothesis: >-
  On the 5 development months, one candidate per technology (wind default 2 for the wind farms,
  `a2_member` for the solar farms) scores better than the other candidates of its technology in
  the headline band, and the stored and rebuilt forecasts can be scored by the one set of rules
  that experiment 609 fixed for the final test.
prediction: >-
  Written before any new forecast was scored. P1: wind default 2 beats wind default 1 and
  `climatology` over 24 to 240 h by about 0.2 points of capacity, as in experiment 347. P2: solar
  `a2_member` beats the GAM, XGBoost, `climatology`, and `manual_heuristic` references over 12 to
  36 h. P3: scores on rows with the exact-zero readings dropped are higher than scores on all rows.
decision_rule: >-
  Descriptive. Rank the candidates of each technology on the quantile score over 24 to 240 h (wind)
  and 12 to 36 h (solar). The Bonferroni count is the number of candidates compared with each
  technology's default, times the 3 lead bands, and the minimum detectable effect is printed
  beside every contrast. A recommendation is a candidate for the maintainer's decision, not an
  adoption.
benchmark: >-
  Wind: wind default 1 and `climatology`. Solar: `a2_member` quarterly refits against the GAM,
  XGBoost, `climatology`, and `manual_heuristic` references.
row_set_id: development_months_daily_origins_v1
metric: >-
  Quantile score over 11 levels in points of capacity, CRPS, q05-q95 coverage, tail shares,
  threshold-weighted CRPS.
production_status: not-for-production
adopted_in: none
---
**Planned.** This page is the plan of experiment 615. The write-up replaces it when the fits and the
scores finish.
