---
id: "00607"
slug: bakeoff-extend-cached-inputs-to-2026-10-31
title: >-
  Can the cached inputs be extended to 2026-10-31 so the final bake-off test can run?
theme: production-reference-and-best-guess
also_relevant_to: []
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. A data build from tables already on disk and from public weather services: no fit, no
  GPU, no forecast error read. One defensible design. It calls the builders that experiments 397
  and 561 own and edits none of them. Serving path and degradation rule: not touched (this
  repository holds none).
hypothesis: >-
  Each cached input that experiment 605 found stopping early can be rebuilt or re-fetched to
  2026-10-31 from tables on disk or free services, and the rebuilt tables obey the origin and lag
  rules row by row.
prediction: >-
  Written before the build was checked. P1: the ensemble summary, wind-component, cloud-cover, and
  observed-power tables reach 2026-10-31 with every run and every Primary present. P2: no row
  breaks the rule that weather comes from the run issued at or before the origin minus 9 hours, and
  no lag is stamped less than 24 hours before the origin. P3: the ERA5 and CAMS site tables reach
  2026-10-31 and agree with the stored tables on every shared hour. The prediction fails if any row
  breaks P2.
decision_rule: >-
  Descriptive. The experiment adopts nothing, scores nothing, and reads no forecast error, so no
  Bonferroni correction or minimum detectable effect applies.
benchmark: >-
  The stored tables on the hours they share with the rebuilt tables.
row_set_id: not_applicable_data_build
metric: >-
  Newest timestamp of each input, rows breaking each assertion, and the largest difference between
  rebuilt and stored values on shared rows.
---

Planned. The write-up replaces this page when the build finishes.
