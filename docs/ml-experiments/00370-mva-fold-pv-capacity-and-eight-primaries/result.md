---
id: "00370"
slug: mva-fold-pv-capacity-and-eight-primaries
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
kind: other
size: >-
  medium. Stored: small result files in the private folder only. Production serving path: not
  touched. Degradation rule: not touched. Designs: more than one would test the hypothesis (a
  pinned capacity inside the fit through the meter, which needs a change to `studies.dp_meter`, or
  a comparison of the fitted capacity with the pin and with the experiment 269 estimate, which
  needs none); the comparison is chosen because experiments 349 and 350 found no capacity source
  better than the pin, and because a `packages/studies` change needs its own reviewed PR. Callers:
  reuses the scripts of experiments 358 and 366 and the helpers of experiment 110, whose callers are
  all in those folders.
hypothesis: >-
  When the registered rule of experiment 366 chooses between the fit through the meter (L0) and the
  naive fit (N), the solar capacity of the chosen fit stays close to the capacity of the fit to the
  signed series at the six Primaries with signed MW readings, and at the eight MVA-only Primaries
  the fit through the meter finds reverse flow at the substations NGED marks "Yes" or "Maybe", not
  only at Stickney (D12).
prediction: >-
  P1. At the scored cut with reactive power zero, for the six signed-MW Primaries (D5, D9, D10, D11,
  D14, D16), the chosen fit's capacity is within 15% of the signed fit's at 5 or more of 6 at the
  natural level and at injection level 1.0. P2. The pinned capacity (experiment 45's vintage at the
  cut) is within 25% of the signed fit's at 4 or more of 6 at the natural level; this is the
  contrast most likely to fail, because experiments 349 and 350 found the pin hard to beat but not
  that it matches the signed fit. P3. At the eight MVA-only Primaries, L0's share of training
  half-hours with fitted reverse flow exceeds 10% at D12 and D8 (NGED "Yes") and is below 10% at the
  other six, which include Langrick (D22, "Maybe"). P4. At the eight MVA-only Primaries the chosen
  arm's mean absolute error on the reading is no higher than N's, and at D12 it is lower by at least
  1 point of capacity.
decision_rule: >-
  Passes (and the all-16-Primaries step is planned) when P1 holds in both cases and P3 holds at D12
  and at 6 of the other 7 substations. P2 and P4 are reported and do not gate. Fails otherwise, and
  the sequence stops for a decision. The rule of experiment 366 is used as registered, with its
  thresholds unchanged and not tuned on the census. No cost-based threshold, because the experiment
  ports nothing.
benchmark: >-
  The signed fit (S), the naive fit (N), and the pinned capacity, at the six signed-MW Primaries;
  N at the eight MVA-only Primaries
row_set_id: >-
  dp_mw_v1 sites D5, D9, D10, D11, D14, D16 (scored) and the 8 MVA-only Primaries D2, D3, D4, D8,
  D12, D19, D21, D22 (census, reading error only); cuts 2024-07-01 and 2025-07-01
metric: >-
  Ratio of the chosen fit's last-season-year solar capacity to the signed fit's; the fit's share of
  training half-hours with reverse flow; mean absolute error on the reading in points of capacity
verdict: ""
production_status: not-for-production
adopted_in: none
date: 2026-10-28
---

# MVA fold: solar capacity and the eight MVA-only Primaries

**Planned. Written before any score of this experiment is read.**

This is steps 2 and 3 of the MVA sequence. Experiment 366 registered a rule that chooses between
the fit through the meter and the naive fit, and it fails at power factor 0.97: real MVA-only meters
have unknown reactive power, so every result here holds only for reactive power zero. The rule is
used exactly as experiment 366 registered it and is not tuned on the census.

## Plan

- **Capacity source.** Experiments 349 and 350 found no source better than the pinned capacity
  (349's estimate scored 0.047 points worse, 350's capacity-free curve was not supported), so the
  pin is the reference. The eight MVA-only Primaries have no pin, so there the fitted capacity is
  reported without a reference.
- **Signed-MW Primaries.** D9 is new; D5, D10, D11, D14 and D16 reuse the cells of experiment 358.
  The cells are the three injection levels, at both cuts, with reactive power zero.
- **Census.** L0 and N are fitted at the eight MVA-only Primaries at both cuts, frozen on past
  weather as in experiment 358, and scored on the following year's readings. No signed truth
  exists, so only reading error and fitted reverse flow are reported.
- **Not covered.** Power factor 0.97; the all-16-Primaries forecast stack (a separate experiment,
  planned only if this one passes); D15 and D18 (not MVA-only census members, handled there).
