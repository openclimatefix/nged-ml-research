---
id: "00605"
slug: bakeoff-final-test-data-check
title: >-
  Is the data for the one final bake-off test (forecast origins 2026-07-01 to 2026-10-31) complete
  and clean?
theme: production-reference-and-best-guess
also_relevant_to: []
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. A data check on stored tables: no fit, no GPU, no new data, no forecast error read. One
  defensible design. It reads the power tables, the ECMWF ensemble tables, and the feature code
  that other experiments own, and edits none of them. Serving path and degradation rule: not
  touched (this repository holds none).
hypothesis: >-
  The power and weather tables reach 2026-10-31 with few enough gaps and flagged half-hours that
  every release candidate can be scored on forecast origins 2026-07-01 to 2026-10-31.
prediction: >-
  Written before any table was read. P1: every Primary, grid supply point, bulk supply point, wind
  farm, and solar farm has readings to 2026-10-31 with under 2% of half-hours missing in each month
  from 2026-05 to 2026-10. P2: every daily ensemble run from 2026-04-01 to 2026-10-31 exists. P3:
  the derived ensemble inputs and the pretraining inputs also stop before 2026-10-31. The
  prediction fails if P1 or P2 fails. P3 is a guess about what is missing, not a pass condition.
decision_rule: >-
  Descriptive. The check adopts nothing, reads no forecast error, and has no contrast, so no
  Bonferroni correction or minimum detectable effect applies. A gap counts as blocking for a
  candidate if the candidate cannot be scored on at least 95% of the test origins without it.
benchmark: >-
  The five development months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), on the same tables.
row_set_id: not_applicable_data_check
metric: >-
  Missing half-hours, flagged half-hours by drop reason, and missing ensemble runs, per series and
  month.
---

Planned. The write-up replaces this text.
