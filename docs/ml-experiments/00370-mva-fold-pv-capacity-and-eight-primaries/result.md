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
  P1. At the scored cut with reactive power zero, at injection levels 0.0 and 1.0, the capacity of
  the fit the rule of experiment 366 chooses (`capacity_l0` or `capacity_n`, last season-year at
  the cut) is within 15% of the signed fit's at 5 or more of the 6 signed-MW Primaries (D5, D9, D10,
  D11, D14, D16). Experiment 358 already found this for L0 at 5 of 5, so what is new is D9 and the
  rule's choice. P2. The pinned capacity (experiment 45's `frozen` vintage at the cut; D9 has one,
  the MVA-only Primaries have none) is within 25% of the signed fit's at 4 or more of the 6 at level
  0.0; this is the contrast most likely to fail. P3. L0 does not separate reverse flow from none by
  itself: its fitted share of sunny-midday training half-hours with reverse flow exceeds 50% at 6 or
  more of the 8 MVA-only Primaries, including some NGED marks "No" (experiment 365 found 96 to 97%
  at D5, which has none), so the rule is needed. P4. At D12 the chosen arm (L0) has a mean absolute
  error on the reading that is lower than N's by at least 1 point of capacity at both cuts.
decision_rule: >-
  D12 is its own criterion: the rule chooses L0 at D12 at both cuts and P4 holds there. The other
  criterion is P1, which must hold in both injection cases. The experiment passes when both hold,
  and the all-16-Primaries step is then planned. P2 and P3 are reported and do not gate. Fails
  otherwise, and the sequence stops for a decision. A pass holds for reactive power zero only and
  does not license the all-16 step on real meters, whose reactive power is unknown. The rule of
  experiment 366 is used as registered, with its thresholds unchanged and not tuned on the census.
  The capacity is read at the cut, and the 2025-07-01 cut's last season-year holds no post-cut
  rows. No cost-based threshold, because the experiment ports nothing.
benchmark: >-
  The signed fit (S), the naive fit (N), and the pinned capacity, at the six signed-MW Primaries;
  N at the eight MVA-only Primaries
row_set_id: >-
  dp_mw_v1 sites D5, D9, D10, D11, D14, D16 (scored) and the 8 MVA-only Primaries D2, D3, D4, D8,
  D12, D19, D21, D22 (census, reading error only); cuts 2024-07-01 and 2025-07-01
metric: >-
  Ratio of the chosen fit's last-season-year solar capacity to the signed fit's; the fit's share of
  sunny-midday training half-hours with reverse flow; mean absolute error on the reading in points of capacity
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
  exists, so the raw reading is passed as the folded reading, and only reading error and fitted
  reverse flow are reported. This needs a new script: experiment 358's cells cover only D5, D10,
  D11, D14 and D16, so D9 and the eight MVA-only Primaries are new runs.
- **Not covered.** Power factor 0.97; the all-16-Primaries forecast stack (a separate experiment,
  planned only if this one passes); D15 and D18 (not MVA-only census members, handled there).
