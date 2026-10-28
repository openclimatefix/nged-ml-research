---
id: "00366"
slug: mva-fold-irradiance-correlation-reverse-flow-rule
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
kind: other
size: >-
  simple. Stored: small result files in the private folder only. Production serving path: not
  touched. Degradation rule: not touched. Designs: more than one statistic on the readings would test
  the hypothesis; the Pearson correlation with irradiance is the one experiment 365 proposed, and the
  low-percentile test is kept as the second condition. Callers: reads the stored results of
  experiment 358 and the feature cache, so no new model fit is needed.
hypothesis: >-
  Reverse flow can be told apart from invented reverse flow using the meter readings and the
  irradiance forecast alone: where a substation has reverse flow, its sunny-midday meter readings
  rise with irradiance (the magnitude of a negative flow grows), and where it has none, they fall.
prediction: >-
  At the scored cut with reactive power zero, the rule picks the naive fit at D5 with no injection
  (plain direction accuracy within 0.02 of the signed fit's), picks the fit through the meter (L0) at
  every injected case (levels 1.0 and 2.0), and picks L0 at Stickney (D12) in the census of the 8
  MVA-only Primaries. The correlation at D12 is about +0.75 to +0.83.
decision_rule: >-
  The rule uses L0 if the Pearson correlation of the training sunny-midday readings with irradiance
  exceeds +0.3, or the 1st percentile of those readings is below 1.0 percentage point of capacity;
  otherwise it uses the naive fit (N). Passes when, at the scored cut with reactive power zero, the
  chosen arm is within 0.02 of the signed fit's plain sunny-half-hour direction accuracy at D5 with
  no injection (its own criterion), and at 4 of 5 substations in each injected case at both cuts,
  and the rule picks L0 at D12. Fails otherwise, and the sequence stops for a decision. The 1.0
  threshold on the 1st percentile is informed by the numbers of experiment 365, so it is not a clean
  out-of-sample test; the correlation threshold of +0.3 comes from the design brief's measurements.
benchmark: >-
  The signed fit, the ungated fit through the meter, and the naive fit, all from experiment 358
row_set_id: >-
  dp_mw_v1 sites D5, D10, D11, D14, D16 (scored); the 8 MVA-only Primaries (census only); cuts
  2024-07-01 and 2025-07-01
metric: >-
  Plain share of sunny half-hours with the flow direction right, for the arm the rule chooses; the
  rule's two statistics
---

# MVA fold: reverse-flow rule from the correlation with irradiance

Planned. The rule and its thresholds are fixed here before any statistic is computed.
