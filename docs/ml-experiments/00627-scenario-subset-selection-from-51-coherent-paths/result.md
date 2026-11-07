---
id: "00627"
slug: scenario-subset-selection-from-51-coherent-paths
title: >-
  Which 3, 5, or 10 of the 51 coherent member paths should NGED's power-flow studies use, with no
  label attached to any path?
theme: calibration-and-uncertainty
also_relevant_to:
  - hierarchy-and-pooling
  - generator-forecasting
status: finished
kind: other
protocol: demand-v2
date: 2026-11-07
size: >-
  medium. Stored: tables and figures in this experiment's private folder only; no forecast is
  fitted. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  defensible (summary choice, clustering distance, selection window), so the plan registers each.
  Callers: this folder's scripts only, which import experiments 434, 436, and 625 and `studies.*`.
  One Sonnet plan review, one Sonnet code review before the first run, one Sonnet results review,
  and two Sonnet write-up reviews.
hypothesis: >-
  NGED will run a handful of power-flow studies by hand, so they need K of the 51 coherent member
  paths (K = 3, 5, 10) that cover the distribution of the 51, without labelling any path. A path
  that is low wind on day 1 need not be low wind on day 2, so a label such as "the low-wind
  scenario" does not carry over. We test whether picking K paths by stratifying on one summary of
  the run (S) or by clustering the 51 joint paths (C) covers the 51-path distribution better than a
  simple random sample (R).
prediction: >-
  Written before any score is read. P1. At K = 5 the random sample R has a relative gap above 10%
  at the 5th or 95th percentile of the 51-path distribution on at least one summary, and holds each
  tail (a path of rank 6 or lower, or 46 or higher, of the 51 on a summary) in about 1 - C(45,5) /
  C(51,5) = 48% of draws, so its tail coverage is near 48%; the best of 5,000 random K = 5 subsets
  (the oracle floor, printed beside every arm) has a mean relative gap below 10%, so the bar is
  reachable only by choosing on the gap itself, which no rule can do. P2. A stratified sample S
  has a smaller gap than R on the summary it is ranked on and a
  tail coverage of 50% to 65%, above R's 48%, but its mean relative gap stays above 10% at the 5th
  and 95th percentiles, because the smallest of five stratified draws still sits several ranks
  inside the tail. P3. The cluster medoids C are central paths
  and rarely include a tail path, so C has lower tail coverage than R. Therefore the expected
  verdict is `partly supported` (S passes the tail condition, nothing passes the coverage
  condition) at K = 5, and K = 10 narrows the gap without closing it. The tail test has 10 month
  blocks, so a failure to clear the interval is read as low power. P4. Within a run, the Spearman
  correlation of the 51 paths' wind summary between days 1 to 3 and days 4 to 10 is below 0.5, so
  a path's label does not carry over between bands. P5. Across weekly runs, the share of members
  that keep the same stratum is within 3 points of the chance level 1 / K for the Schaake-shuffled
  members (a tautology, since their index is arbitrary), and for the ensemble-copula members, whose
  index is the weather member.
decision_rule: >-
  Fixed before any score is read. Rows: the Tuesday runs of 2025-09 to 2026-06 stored by experiment
  436 (44 expected; the count is printed first), grouped into calendar months for a month-block
  bootstrap (20,000 resamples). Paths: experiment 436's Schaake-shuffled scenarios (J3), 51 per run
  over 25 nodes (16 Primaries, 3 wind farms, 6 solar farms) and 702 half-hours, in points of
  capacity; the ensemble-copula scenarios (J2) are a sensitivity. Four summaries per path and lead
  band: the mean over the wind farms, the mean over the solar farms, the peak over the band of
  the net demand (mean over the Primaries minus the wind-farm mean minus the solar-farm mean; an
  unweighted proxy, because MW weights are not in the run files), and the peak over the band of the
  Primaries' mean demand. Lead bands: 12 to 36 hours, days 1 to
  3, days 4 to 10, and days 1 to 10 (the headline). The selection is made once per run on the days 1
  to 10 window; a selection made inside each band is a secondary arm. Arms: R (1,000 random draws
  of K paths per run), S-wind, S-solar, S-net, and S-demand (rank the 51 paths by that summary,
  cut into K
  equal-count strata, draw one path from each; 1,000 draws per run), C (k-medoids with k-medoids++
  starting points and seed 627, on the 51 joint paths with each node and half-hour centred across
  the 51 paths, scaled by one standard deviation per node, every 4th half-hour of the window, all
  25 nodes equally weighted, best of 10 starts, each medoid weighted by its cluster
  size), Oracle (the best of 5,000 random
  K-subsets on the gap itself, a floor and not a rule), and All (the 51). Every arm is scored on all
  four summaries. Coverage: per summary and band, the Wasserstein-1 distance of the
  weighted K-path distribution to the 51-path distribution, and the gap at the 5th, 50th, and 95th
  percentiles (lower inverse CDF), each divided by the spread (95th minus 5th percentile) of the
  51-path distribution of the same run, then averaged over summaries and runs. Tail coverage: the
  share of draws (for C, of runs) whose K paths include a path at or beyond the 10th percentile
  (lower tail) or the 90th percentile (upper tail) of the 51 on a summary, averaged over summaries
  and both tails. Joint score: energy score and variogram score (order 0.5) of the K weighted paths
  against the observed vector over the 25 nodes at a coarse step (about 25 half-hours per band),
  reported as the relative difference to All. Stability: within each run, the Spearman correlation
  and the same-stratum share of the 51
  paths between lead bands, per summary; across runs, the share of members in the same one of K
  strata on consecutive weekly runs (the stored runs are Tuesdays, so weekly replaces the brief's
  consecutive days), against the chance level 1 / K. Absolute scores of every arm (and R's 5th to
  95th percentile across draws) are reported beside each contrast. The joint vector is built member
  first, `(51, 25, T)` flattened to `(51, 25 * T)`, which avoids the reshape error experiment 625
  found in 436.
  The verdict is judged at K = 5, days 1 to 10, J3. A label-free arm passes the coverage condition
  if its mean relative gap is at most 0.10 at each of the three percentiles (point estimate), and
  passes the tail condition if its tail coverage exceeds R's, with a month-block 99% interval (a
  Bonferroni correction over the five label-free arms S-wind, S-solar, S-net, S-demand, and C, so
  99% two-sided) that lies wholly
  above zero. Verdict: `supported` if at least one label-free arm passes both conditions; `partly
  supported` if at least one passes the tail condition but none passes the coverage condition;
  `not supported` otherwise. K = 3 and K = 10, the other bands, the other summaries, the joint
  scores, the stability check, and J2 are descriptive and carry no verdict. Part 2, added after
  the part 1 scores were read and before any BSP-group score was read, on the
  manager's instruction: the same selection, with the peak export (minus the sum, in MW) of a bulk
  supply point (BSP) group as the summary. The groups are experiment 161's covered children, D13 =
  D9 + D10 + D11 and D17 = D5 + D14 + D16, which omit unmetered load and the Primaries that report
  apparent power. Arms: R, S-bsp (strata of that BSP's peak export), C (the part 1 medoids), Q
  (K = 4 only: the paths at the 50th, 90th, 95th, and 99th percentile of the 51 peak exports), and
  All. The gate score is the share of observed daily peak exports (days with at least 44 of 48
  half-hours observed on all three children, before 2026-07-01) above the largest daily peak export
  of the chosen paths, at K = 5 and K = 4, days 1 to 10, J3, per BSP; an arm passes the gate if its
  share is at most the share of All plus 0.05 (point estimate, month-block mean). Part 2 carries
  its own words, `gate passed` and `gate failed`, and no verdict. The day-to-day stability score is
  dropped; the within-run band-to-band check stays. Demand is the stored XGBoost demand of
  experiment 436, not the neural networks of the bake-off. No rescue tuning after
  a failed bar. Stop rule: experiment 436's `work/paths/J3` is missing for any of the stored runs.
benchmark: >-
  R (a simple random sample of K paths) for the verdict; All (the 51 paths) as the reference for
  every score.
row_set_id: joint_tuesdays_v1
metric: >-
  Relative gap at the 5th, 50th, and 95th percentiles and Wasserstein-1 distance of the K-path
  distribution to the 51-path distribution; tail coverage at the 10th and 90th percentiles; energy
  and variogram score relative to All; stratum stability across weekly runs.
code: ml-experiments/00627-scenario-subset-selection-from-51-coherent-paths/
verdict: partly supported
best_arm: >-
  Ranking the 51 paths by the peak Primary demand, cutting them into K = 5 equal groups, and drawing
  one path from each group (S-demand): tail coverage 52.9% against 48.0% for a random sample (+4.9
  points, 99% interval +4.3 to +5.6); the mean gap at the 5th, 50th, and 95th percentiles is 11.4%,
  9.9%, and 13.3% of the range of the 51 paths (random sample 14.5%, 15.2%, 17.7%).
headline: >-
  None of the rules we tested for picking 5 of the 51 paths (random, four stratified, and
  cluster-based) holds the 5th and 95th percentile within 10% of the range of all 51 (the best reach
  11% to 14%, and a search on the gap itself reaches 4%). Stratifying on a summary beat a random
  sample on tail coverage in all 10 months, and at K = 10 the mean gap is under 10% for every rule.
  Under the Schaake shuffle a path keeps its wind rank between days 1 to 3 and days 4 to 10 only
  weakly (rank correlation 0.19), and keeps its solar and demand rank more strongly (0.62 and 0.74).
lessons: >-
  A stratified pick of 5 paths is a modest and consistent improvement on a random pick, not a
  solution: none of the five-path rules we tested reached the tails of the 51, though a search on
  the gap itself found five-path subsets that did. Ranking by the bulk supply point's own peak
  export
  does not make the pick safe for a limit on that substation: an observed daily peak export beat the
  largest of 5 chosen paths on 12% of days, against 1.6% for all 51.
take_home_message: >-
  Picking 5 of the 51 forecast paths by cutting them into 5 equal groups on one summary and taking
  one path from each covers their spread better than picking 5 at random, in every month, but still
  misses the tails: the stratified rules leave a gap of 11% to 12% of the range at the 5th
  percentile and 13% to 14% at the 95th. For a bulk supply point's limit, an observed daily peak
  export beat the largest of 5 chosen paths on 12% of days (1.6% for all 51); 10 paths chosen by
  that substation's own peak export came closest (6.3%, against 7.3% for a random 10). NGED should
  treat 10 paths as a minimum when the tail matters.
advice:
  - >-
    Offer NGED at least 10 paths, not 5, when the tail of a bulk supply point matters: at K = 10 the
    stratified pick keeps the gap under 10% and an observed daily peak export beats the largest
    chosen path on 6.3% of days, but the margin over the gate is 0.3 points, so treat 10 as a
    minimum (Part 2 gate).
  - >-
    Choose a small set by stratifying the 51 paths on one summary and drawing one path per stratum,
    not at random: the stratified pick had higher tail coverage in all 10 months (verdict
    contrasts, Figures 1 and 2).
  - >-
    Do not expect k-medoids clusters to beat stratifying for tail studies: their tail coverage
    minus a random sample's had a 99% interval of -3.2 to +11.9 points, so the experiment could not
    separate them from a random sample (Figure 1).
  - >-
    Do not label a scenario "the low-wind path" for a whole run: under the Schaake shuffle a path's
    wind rank between days 1 to 3 and days 4 to 10 has a rank correlation of only 0.19 (0.62 for
    solar, 0.74 for demand), because each path takes a different past date for each lead (Figure 5).
  - >-
    If a fixed set of extreme paths is wanted, include scenarios from the far tail as well as the
    middle: choosing the paths at the 50th, 90th, 95th, and 99th percentile of the substation's peak
    export (Q) always included an export-tail path, but its gap at the 5th percentile was 49% of the
    range, so the set misrepresents the low end (Part 2).
production_status: not-for-production
adopted_in: none
---

**We asked which 3, 5, or 10 of 51 forecast scenarios NGED should run by hand, and found that
choosing 5 by rank on one measure beats choosing 5 at random but still misses the extremes.** At 10
scenarios the mismatch fell below 10% of the range for every rule we tried; we did not test 6 to 9.
National Grid Electricity Distribution (NGED), the distribution network operator, will run a handful
of power-flow studies by hand (calculations of how electricity flows through its network under one
forecast scenario). We took the 51 equally likely forecast scenarios, called paths, that experiments
[436](../00436-joint-scenarios-across-demand-wind-solar-and-leads/result.md) and
[625](../00625-coherent-members-from-quantiles-k-sweep-and-path-checks/result.md) stored for each of
44 weekly runs. Each path is one joint forecast of demand, wind, and solar for 25 sites (16 Primary
substations, 3 wind farms, 6 solar farms) over the next 10 days. We chose K = 3, 5, or 10 of the 51
paths in several ways. For each way, we measured how far the chosen paths' 5th, 50th, and 95th
percentiles sit from those of all 51, as a share of the range between the 5th and 95th percentile of
all 51 (the gap). Part 2 repeats the choice for the summed demand of the three Primaries under each
of two bulk supply points (BSPs). Part 1 and Part 2 reuse [experiment
161](../00161-dp-bsp-top-down-against-bottom-up/result.md)'s choice of covered Primaries.

Terms: K is the number of paths chosen. **R** random sample of K paths (1,000 draws per run);
**S-wind, S-solar, S-net, S-demand** rank the 51 paths by one summary, cut them into K equal groups,
and draw one path from each; **C** k-medoids clustering (each cluster is represented by its most
central path), each medoid weighted by its cluster size; **Oracle** the best of 5,000 random
K-subsets chosen on the gap itself, a floor that no rule can reach; **gap** the difference between a
percentile of the chosen paths and the same percentile of the 51, divided by the spread (95th minus
5th percentile) of the 51; **tail coverage** the share of draws that include a path in the lowest or
highest 10% (ranks 1 to 6, or 46 to 51) of the 51 on a summary. The summaries are the mean wind-farm
output, the mean solar-farm output, the peak net demand (an unweighted proxy, in points of
capacity), and the peak Primary demand, over days 1 to 10. Intervals are month-block bootstrap
intervals over 10 months. Generators are anonymised.

![Tail coverage by selection rule](figure_1_tail_coverage.svg)

Figure 1 shows the share of draws that include a path in the lowest or highest 10% of the 51, with
95% month-block intervals; higher is better.

![Gap to the 51-path distribution](figure_2_percentile_gap.svg)

Figure 2 shows the mean gap at the 5th, 50th, and 95th percentiles as a share of the range of the
51; the dashed line is 10%.

![Variogram score of K paths relative to all 51](figure_3_joint_variogram.svg)

Figure 3 shows how much worse than all 51 paths each rule scores against the observed vector of all
25 sites; lower is better.

![Gate for BSP group peak export](figure_4_bsp_gate.svg)

Figure 4 shows the share of observed days on which the daily peak export beat the largest chosen
path; the dashed line is the gate (all 51 plus 5 points).

![Rank correlation of a path between lead bands](figure_5_label_carry_over.svg)

Figure 5 shows the rank correlation of the 51 paths between days 1 to 3 and days 4 to 10 within a
run, for two ways of coupling the paths.

## Results

- **Verdict (K = 5, days 1 to 10, Schaake-shuffled paths, J3: each path takes the day-to-day pattern
  of a different past date for each lead).** `partly supported`: all four stratified rules pass the
  tail condition and none passes the 10% coverage condition. The decision rule's 99% intervals for
  tail coverage minus R were +4.2 points (+3.7 to +4.8) for S-wind, +4.5 (+3.9 to +5.2) for S-solar,
  +4.2 (+3.7 to +4.7) for S-net, and +4.9 (+4.3 to +5.6) for S-demand, positive in 10 of 10 months
  each. C: +3.9 points (-3.2 to +11.9), positive in 5 of 10 months.
- **Absolute scores at K = 5 (mean of the 5th, 50th, and 95th percentile gaps; tail coverage).** R
  0.158 and 48.0%; S-wind 0.121 and 52.2%; S-solar 0.119 and 52.5%; S-net 0.123 and 52.2%; S-demand
  0.116 and 52.9%; C 0.122 and 51.9%; Oracle 0.039 and 86.7%. The 5th, 50th, and 95th percentile
  gaps are in the table below. R's gap at the 95th percentile varies by 0.156 (mean per-run standard
  deviation across draws), so one random draw can be much better or worse.
- **K = 3 and K = 10.** Mean gap at K = 3: R 0.226, S 0.174 to 0.185, C 0.182. At K = 10: R 0.098, S
  0.072 to 0.077, C 0.085, so the mean gap of every rule is under 0.10 at K = 10. Per percentile, R
  is 0.101 at the 95th and C 0.123, so only the stratified rules (at most 0.083 at each percentile)
  pass the 10% coverage condition at K = 10. The plan predicted that K = 10 would narrow the gap
  without closing it; that prediction failed. Tail coverage at K = 10: R 75%, S 83% to 85%, C 78%;
  at K = 3: R 32%, S 34%, C 24%.
- **Each stratified rule matches best the summary it ranks.** S-wind has a wind gap of 0.058 at the
  50th percentile and 0.085 at the 95th; on the other summaries its gaps are 0.11 to 0.19, close to
  R. A stratified pick on one summary therefore helps little on the others.
- **Joint score (variogram, 25 nodes, relative to all 51; absolute 2.161).** K = 5: R +16.4% (+14.7%
  to +18.4%), S +14.4% to +14.9%, C +14.9% (+11.4% to +18.9%), Oracle +15.1%. K = 10: R +7.2%, S
  +6.2% to +6.5%, C +7.5%. K = 3: R +28.6%, S +25.6% to +26.5%, C +22.4%. At K = 5 and K = 10 no
  rule differs from the others by more than about 2 points; at K = 3 the rules span 6 points (C
  +22.4%, R +28.6%). The loss comes mostly from using fewer paths.
- **Labels do not carry over.** Within a run, the Spearman correlation between the 51 paths' ranks
  on days 1 to 3 and on days 4 to 10 is 0.19 for wind, 0.41 for net demand, 0.62 for solar, and 0.74
  for Primary demand under the Schaake shuffle (J3), and 0.18, 0.01, 0.10, and 0.11 under ensemble
  copula coupling (J2). Prediction P4 (a path's wind rank correlates below 0.5 between lead bands)
  held for wind and failed for solar and demand under J3, because the Schaake shuffle gives each
  path one past date for every lead.
- **Coupling.** J2 (ensemble copula coupling: each path follows one weather-forecast member) at K =
  5 keeps S ahead of R (mean gap 0.146 to 0.149 against 0.167; tail coverage 50.5% to 50.9% against
  48.0%), but C is worse than R (gap 0.195, tail coverage 51.1%), unlike under J3.
- **Selection inside each band** (secondary): selecting inside the 12-to-36-hour or days 1 to 3 band
  instead of on days 1 to 10 improves that band's gap by about 0.01 to 0.016 for the stratified
  rules (the days 4 to 10 band has no gap for the run of 2026-06-30, so it is not compared). Within
  days 1 to 10 the two selections are the same.

| Arm (K = 5) | Gap at 5th percentile | Gap at 50th | Gap at 95th | Tail coverage |
|---|---|---|---|---|
| R | 0.145 | 0.152 | 0.177 | 48.0% |
| S-wind | 0.119 | 0.105 | 0.139 | 52.2% |
| S-solar | 0.113 | 0.105 | 0.139 | 52.5% |
| S-net | 0.119 | 0.108 | 0.141 | 52.2% |
| S-demand | 0.114 | 0.099 | 0.133 | 52.9% |
| C | 0.086 | 0.104 | 0.173 | 51.9% |
| Oracle | 0.041 | 0.041 | 0.034 | 86.7% |

- **Part 2: BSP group peak export (D13 and D17, K = 5 unless stated).** The gate is the share of
  observed days on which the daily peak export exceeded the largest daily peak export among the
  chosen paths, over up to 9 full days per run (381 days in 43 runs). All 51 paths: 1.6%. K = 5: R
  14.3%, S-bsp 12.1%, C 12.6%; all fail the gate (at most 1.6% + 5 points). K = 4: Q (paths at the
  50th, 90th, 95th, and 99th percentile) 8.0% (the difference to All is +6.3 points, +2.1 to +11.8),
  against S-bsp 14.7% and R 17.4%. K = 10: S-bsp 6.3% (+4.7 points, +2.4 to +7.1) passes the gate; R
  7.3% and C 7.0% fail by 0.7 and 0.4 points. The stratified pick on the BSP's own peak export has
  the lowest 50th percentile gap (0.045 at K = 5) and a 95th percentile gap of 0.114. Q hits the
  export tail on every run but its 5th percentile gap is 0.49 of the spread, by construction.
- **Tails.** The summaries are forecasts of the stored marginals of experiments 425, 429, and 430;
  no new quantile scores are computed. The tail results above are the experiment's tail results.
- **Checks.** No forecast was fitted, so the median-minus-mean check and audit triggers 3 to 6 do
  not apply. Audit trigger 1 (a control fails) did not fire. Audit trigger 2 (a result contradicts
  its own prediction) fired: prediction P3, that C would have lower tail coverage than R, and the
  prediction that K = 10 would not close the gap both failed (P1 and P2 held, and the verdict
  `partly supported` was predicted). For P3, (C was not below R on tail coverage; C's interval is
  wide, +3.9 points, -3.2 to +11.9); the explanation, that medoids carry the cluster's weight but
  sit near the centre, was not tested. Audit trigger 7 (a later design depends on this result): none
  does. Part 1 scored all 44 runs; the last run, 2026-06-30, has an identical wind summary on all 51
  paths in days 4 to 10 (spread zero), so its gaps for that band and summary are missing and the
  means use the other runs. Means in the table weight each month equally.
- **Caveats.** Demand is the stored XGBoost demand of experiment 436, not the neural networks of the
  bake-off. The net-demand summary is an unweighted proxy in points of capacity. The BSP groups are
  3 Primaries each (experiment 161's covered children) and omit unmetered load and Primaries that
  report apparent power; the groups are in MW from each Primary's capacity. Ten months give wide
  intervals; the gate for K = 10 passes by 0.3 points and should not be read as a safe margin.
  Forecast paths for valid times after 2026-07-01 are used in the selection of the last runs; only
  the observed values are held out. The gate takes an observed daily peak over at least 44 of 48
  half-hours, which can miss a peak, and so understates the exceedance for all arms, `All` included.
  The 51 member values are those of experiments 425, 429, and 430; experiment 626 widens their tails
  and was not used. The day-to-day stability score was dropped, so prediction P5 (stratum
  persistence across weekly runs) was not tested.
- **Reused and new.** Reused: experiment 436's `work/runs` and `work/paths/J3` and `J2`, and the
  folder of experiment 625 (read only; nothing written to either). New: every selection rule, the
  summaries, the gap, tail, joint-score, and gate tables, and the figures. The folders of
  experiments 434, 436, and 377 are on `PYTHONPATH` for the series table and capacities.
- **Options for later.** Test whether NGED's power-flow answer changes between K = 5 and K = 10
  paths; repeat Part 2 with the neural-network demand exports and the topology of every BSP; and
  choose the paths on the gate itself with a rule that trades the median gap against the tail.
- **Reviews.** A Sonnet plan review (12 findings: accepted a fourth demand summary, a bound for the
  coverage condition from an oracle floor, a within-run label check, absolute scores and R's spread
  beside each contrast, the corrected tail rank, 5 arms in the Bonferroni correction, one scale per
  node in the clustering; rejected a toy test of the reshape, since the vector is built member
  first, and a fractional-day variant). A Sonnet code review before the first run (the last run
  crashed the oracle on a zero spread, fixed; the k-medoids seed was made independent of the other
  arms; per-draw 5th and 95th percentiles were added). A Sonnet code review of the BSP script found
  the gate covered 14 days instead of days 1 to 10, fixed before the reported run. The write-up
  reviews are recorded in the pull request.
