---
id: '00498'
slug: solar-e-annual-refit-schedule-only-read
title: >-
  Is the loss from refitting the Solar E network once a year instead of every quarter larger than 1%
  once the network design is held fixed?
theme: metered-solar-farms
also_relevant_to:
- generator-forecasting
status: planned
date: 2026-11-02
kind: other
size: >-
  simple. Stored: seeds 6 to 9 of one arm in this experiment's private folder only. Designs: one,
  experiment 493's own contrast C2. Callers: this folder imports experiment 310's, 430's, and 493's
  modules. Serving path and degradation rule: not touched (this repository holds none). Asset
  graph: not touched.
hypothesis: >-
  At Solar E the 1.28% loss that experiment 493 reported (its contrast C3) comes mostly from the
  network design (one pass against the 51-member encoder), and the schedule-only loss (annual against
  quarterly refits of the encoder, contrast C2) is smaller than 1%.
prediction: >-
  Written before any new fit. C2 at Solar E over all bands is +0.5% to +0.6% (experiment 493 stored
  +0.53% at six seeds), with a 97.5% month-block interval whose upper end is below +1.5%. The
  per-seed C2 values at six seeds range from −1.2% to +2.2%, so the six-seed mean has a seed spread
  of about 1 percentage point. Ten seeds narrow that spread by about a quarter. The prediction can
  fail: C2's interval at ten seeds could exclude +1%, or the loss could sit in one quarter.
decision_rule: >-
  The annual schedule is called acceptable at Solar E when the upper end of the 97.5% month-block
  interval of C2 over all bands (Bonferroni count 2: all bands and the 12 to 36 hour band, family
  level 5%, 10,000 draws) lies below 1%, the equivalence limit, and the limit is at or above the
  minimum detectable effect from the seed spread (printed before any score is read). A point
  estimate below 1% with an interval that crosses 1% is inconclusive. If the upper end exceeds 1%,
  Solar E keeps quarterly refits and the serving table states the extra refits. The six-seed read
  of experiment 493's stored forecasts comes first; seeds 6 to 9 are fitted only because C2's
  per-seed range at six seeds contains the C3 loss.
benchmark: >-
  Experiment 430's stored quarterly `a2_member` forecasts (seeds 0 to 5), and for seeds 6 to 9 three
  new later quarterly refits per seed, with the first quarter taken from the annual fit (experiment
  493's control showed the two are the same fit).
row_set_id: >-
  Experiment 310's rows, Solar E only, scored year 2025-09 to 2026-08, rows that every compared arm
  holds with finite quantiles. Solar E alone, so the signed-MW six and the eight do not apply.
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), percentage
  change of annual against quarterly, minimum detectable effect from the per-seed paired contrasts,
  tails at 0.01, 0.05, 0.95, and 0.99, the stored-lead band that starts at 12 hours
code: ml-experiments/00498-solar-e-annual-refit-schedule-only-read/
---

# Solar E annual-refit check: the schedule-only read

Planned. Standing practices followed: cleaned data (experiment 493's stored forecasts), no D15 or D18
(demand only, so not applicable), models saved by experiment 493 and the new fits as state files in
the private folder, July 2026 forecasts are the stored scored year of experiment 310 (the scored
year ends August 2026, as in experiments 430 and 493, and is selection-biased). The refit schedule is
the question here, so monthly refits do not apply.
