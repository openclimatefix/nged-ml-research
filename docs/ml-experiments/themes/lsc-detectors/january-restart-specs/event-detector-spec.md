# Spec: next steps for the hindsight load-shift detector

**This is a copy of the specification that the Discuss session of 2026-11-10 wrote for the January
restart, trimmed of conversation.** The page [state at the January
pause](../state-at-january-pause.md) says which parts ran since. A0 ran as [experiment
686](../../../00686-event-table-transfer-timing-and-false-solar-growth-events/result.md), whose
result replaces the conflict that the take-home below describes: the two ends of the verified
transfers step together within the noise. Part D's cheap test ran as [experiment
687](../../../00687-causal-detector-features-for-the-prevailing-forecast-heads/result.md). Parts A1
to A4, B, and C have not run. Two statements below are out of date. Step 1 of part B1 (a
register-based solar capacity trajectory) cannot be built, because the registers do not record
recent domestic PV. Part C now uses one Sonnet reviewer, not two. The maintainer's amendment to part
D was not re-reviewed by the spec's author. Probabilities and costs are the spec author's estimates,
not commitments.

**Terms.**

- Cells and the found and paired definitions follow experiment 665.
- "False-transfer share" is the share of decoded transfers that match no injected end.

**Baselines.**

| Method | Long and deep paired | Medium and deep paired | Long and deep donor found | Decoded transfers in the 40 scored worlds |
|---|---|---|---|---|
| E1-J-sym (experiment 680) | 0.38 | 0.32 | 0.48 | 1,230, with 283 correct ends |
| E1-J | 0.29 | 0.25 | 0.44 | 788 |
| F-hyb | 0.36 | about 0.33 | 0.52 | not given |
| R-hyb | 0.35 | not given | not given | not given |
| Logistic donor probe of experiment 675 (alone) | not given | not given | 0.63 | not given |

The flow model of experiment 682 accepted 2,990 transfers for 329 correct ends. 88% to 90% of its
transfers had no correct end at every change cost tried.

## Take-home

- **The detector's main problem is precision, not recall.** Experiment 682 showed that a cost per
  change cannot tell a true pair from a coincident one.
- **Two pieces of information could:** the timing of the two ends, and the shape of what moved. Part
  A turns them into features of a small supervised scorer over candidate pairs.
- **The timing evidence conflicts with the maintainer's statement.** The maintainer says real
  switches are simultaneous. Experiment 660 measured the recipient's step 14.5 and 15 hours after
  the donor's at both verified transfers. A0 settles this by measurement before the window is
  shrunk. If switches are simultaneous, matching both ends in time should cut coincident pairs by
  about 5 to 10 times.
- **The convex residual was a convenience.** Arm C was the stored whole-history residual on CPU, not
  a choice on merit. Part B builds the neural cross-fit, buildable from 2019, and compares the two
  fairly.
- **A Claude reviewer can cut false positives only once its own precision is measured** on blinded
  injected and verified cases (part C).
- **Part D's features can be tested cheaply** on the stored banks of experiments 650 and 652 before
  any refit.

**Candid probabilities.**

| Part | Outcome | Probability |
|---|---|---|
| A | False-transfer share cut by at least a third | 45% |
| A | Paired share E1-J-sym + 0.10 | 30% |
| B | The residual improves recall or false events by the bars | 40% |
| C | Reviewer precision of 0.80 or more on the blinded set | 50% |
| D | A gain on the prevailing strata | 30% |

## A. A learnt scorer over candidate pairs

**A0. Measure end timing first (CPU, about 2 hours).**

- **Data.** The verified transfers D5 and D16 (2025-12) and D3 and D22 (2024-10).
- **Two models,** fitted at half-hour resolution to the residual within ±4 days of each end:
    - a shared onset (and a shared end) for both Primaries;
    - separate onsets.
- **Each model allows two step shapes:** constant MW, and a share of the donor's profile. A
  profile-shaped transfer at 01:00 moves little MW at night, so its onset can look late.
- **Report** the likelihood ratio, each onset's profile-likelihood interval, and the residual of
  each fit.
- **Rules.**
    - If the shared onset sits within the noise (likelihood ratio under 4) at both transfers, adopt
      simultaneity with a tolerance of the joint onset interval, about ±2 to 4 half-hours.
    - Otherwise keep a window of 0 to 36 half-hours. Record that the data do not support
      simultaneity, and ask the maintainer whether the switching could have run in stages.
- **Inject both kinds.** The generator then injects simultaneous transfers by default and keeps
  lagged ones as a 20% sensitivity arm.

**A1. Candidate generation.** Generation is tuned for recall; the scorer supplies the precision.

- Every opposite-sign pair of steps of 2 noise units or more, at either end, on the 25 edges.
- Ends refined to the half-hour, onset gap within 48 half-hours.
- Every candidate already proposed by E1-J-sym.
- An all-pairs copy, over any two Primaries, as a placebo.

**A2. Features per candidate pair.** These replace the fixed thresholds of 4 and 2 noise units, the
36-half-hour window, the share range of 0.3 to 2.0, the change costs, the 1.5-noise-unit mask, and
the 6-hour blocks.

- **Size:** the step size at each end in noise units, at 6-hour, 1-day, and 3-day scales; log ρ (the
  recipient-to-donor MW ratio).
- **Timing:**
    - the onset gap and the end gap, each divided by its joint interval;
    - the likelihood ratio of a shared onset, and of a shared end (from A0);
    - whether both ends are seen;
    - the duration;
    - the return to the earlier level at each end, as a fraction.
- **Shape match:** the correlation between the daily profile the donor lost and the daily profile
  the recipient gained, over the event's days. "Behaviour moves", so a coincident pair has no reason
  to share a profile.
- **Context:**
    - the all-Primaries median step at the onset;
    - the number of other Primaries stepping within ±3 hours, in each sign;
    - ERA5 temperature change over ±6 hours;
    - CAMS clear-sky index and the PV-high flag;
    - hour of day and day of week of the onset.
- **Competition:** the number of other candidate partners for each end, the best competing pair's
  score, and this pair's rank among them.
- **Per-Primary statistics:** noise unit, routine-step rate (60 days), MW or MVA flag at each end,
  E1's confidence for each end, and the logistic donor probe's score (experiment 675).
- **Excluded:** Primary identifiers, so the scorer can carry over to the fleet.

**A3. Labels and training.**

- **Labels.** A candidate is positive when both ends match one injected transfer, with its donor and
  recipient. Every other candidate is negative: coincident pairs, placebo pairs, routine steps,
  ramps, and PV runs.
- **Worlds.** Experiment 660's generator, injected into the power, with:
    - default worlds, plus 200 each of `short_heavy()`, `long_heavy()`, and `constant_mw_square()`;
    - events from 2022-01 to 2023-06 for training and 2023-07 to 2023-12 for validation;
    - the 40 scored worlds of experiment 680 (seeds 900 to 939), with events from 2024.
- **Models.**
    - Logistic regression first, because experiment 675 found it matched every neural encoder.
    - Then gradient-boosted trees (XGBoost, depth 4, 300 rounds), with monotone constraints on the
      timing likelihood ratios and the step sizes.
- **Decoding.** A greedy assignment by score: each end is used at most once.
- **Threshold.** Chosen on validation worlds: the highest paired share with a false-transfer share
  of 0.45 or less. The full precision-recall curve is also reported.

**A4. Checks.**

- **Realness.** Experiment 660's classifier, on the pair features, comparing injected true pairs
  with real high-score pairs. The area under the curve must be under 0.65. If it is higher, list the
  features that give the generator away and fix the generator.
- **Cross-injector.** Train on `constant_mw_square()` and score on shaped transfers, and the
  reverse. The drop must be 0.05 or less.
- **Placebos.**
    - The rewired graph: the draw out of 300 that keeps the fewest real links. It must pair 0.05 or
      less on injected worlds, and call a third or fewer of the real record's transfers.
    - The all-pairs candidates: the real-edge scorer must have a false-transfer share 0.10 or more
      below the all-pairs scorer's.
- **Feature ablations:** without the timing features, without the shape match, and without the
  context features.

**Predictions that could fail.**

| Measure | Long and deep | Medium and deep | Other cells |
|---|---|---|---|
| False-transfer share | 0.45 or less (E1-J-sym about 0.69 to 0.77) | 0.45 or less | not stated |
| Paired share | 0.38 or more, so no loss | 0.32 or more, so no loss | no worse than E1-J-sym − 0.03 |
| Paired share at E1-J-sym's false-transfer share | 0.48 or more (+0.10) | 0.42 or more (+0.10) | not stated |

- If A0 supports simultaneity, removing the timing features raises the false-transfer share by 0.10
  or more.
- On the real record, D5 and D16 are paired with D5 as the donor.

**Stop rule.** If the false-transfer share falls by less than 0.10 at no loss of paired share, the
residual cannot separate true from coincident pairs. Stop part A, and keep E1-J-sym in that case.

**Cost.** CPU only: about 4 to 8 hours for the worlds and the candidates, under 1 hour for the
scorers.

**Dependencies.** Experiment 660's generator, experiment 680's code and scored worlds, and
experiment 675's probe.

## B. The residual

**Why arm C was used.**

- Experiment 653 read the stored whole-history residual of experiments 258 and 327, which is the
  lag-free convex model of experiment 130, refitted monthly on a 24-month window. That residual was
  the only one on disk and runs on CPU.
- Its 24-month window contains earlier events, so the residual absorbs some of them.
- The X4 worker never built the neural cross-fit, and the panel starts in 2022.

**B1. The neural lag-free cross-fit.**

- **Model.** Use the K3 recipe of the roster: the lag-free class 3 network, with its weather
  encoder, date-time encoder, Primary embedding, and a mixer with a 3-component mixture.
    - Weather: ERA5 means and CAMS shortwave at the valid time, not ECMWF ENS, because the labeller
      works in hindsight.
    - Date-time features include holidays.
- **Folds.** For each calendar year Y from 2019 to 2026 (2026 ending on 2026-06-30), train on all
  other years and predict year Y. That gives 8 folds, at 15 Primaries.
    - The early-calibration cuts of experiment 591 apply.
    - D15 and D18 start on 2025-03-01.
    - The final-test window from 2026-07-01 is never read.
- **Buildable from 2019.** Experiment 591's step 1 trained on ERA5 and CAMS from 2019-09 on this
  power. Check that the cleaned power is on disk at the 15 Primaries before starting.
- **Seasonal bias** is learned by the date-time and weather encoders from the other years. A change
  specific to year Y, such as a new connection, stays in that year's residual, which is the change
  the detector should see.
- **Unmetered PV, at the six MW Primaries.**
    - Add an explicit term to the expected power: capacity × the PV curve from CAMS.
    - The capacity is experiment 649's calibrated table: p50 as the centre, p10 to p90 as the width.
      The table already applies the 0.68 read-back calibration, so use the calibrated values, never
      the raw ones.
    - Its resolution is a season-year, so a step inside a season is not believed from the table
      alone.
    - Traps (a) to (e) of the PV session reply apply:
        - (a) never subtract the PV term as a single shift without also removing the all-Primaries
          median step;
        - (b) a sunny spring or any run of very sunny days can be read as a capacity step;
        - (c) a load that follows cloud looks like PV;
        - (d) never score the term on forecast weather and call that skill;
        - (e) use calibrated values, because of the 0.68 read-back.
- **Unmetered PV, at the MVA Primaries.** No usable table exists, so use a soft PV-high prior from
  the CAMS clear-sky index. Reverse flow at D8 and D22 can have causes other than sun (a generator
  at D8; unexplained at D22).
- **A solar capacity trajectory is a composite, not a measured series**
  (`briefs/pv_session_reply_solar_drift.md`).
    - No yearly capacity trajectory exists before 2024, beyond season-year capacities from
      experiments 129, 474, and 479 at five MW Primaries from 2021. Experiment 649's table covers
      the summers of 2024 and 2025 only and carries no trajectory.
    - B1 therefore builds the trajectory on CPU, in under 2 hours, in four steps:
        1. a cumulative curve per Primary from the Feed-in Tariff, Embedded Capacity Register, and
           REPD registers, using the service-area polygons (`area_wkt` in
           `data/NGED/metadata.parquet`);
        2. anchored to experiment 649's calibrated levels in summer 2024 and summer 2025;
        3. plus an unregistered-share term, because the probe sits above the register floor at 8 of
           12 Primaries (experiment 673);
        4. with an interval for each year that includes the spread of the 0.68 read-back factor and
           the farm-to-farm spread.
    - Capacity changes only at season-year boundaries, never inside a sunny run (experiment 634).
    - The trajectory removes only the slow drift, about 0.1 to 0.7 MW a year, which is already below
      the detector's step size. It does not remove a sunny-spring residual or a load that follows
      cloud. B relies on the context features of part A (clear-sky index, PV-high flag) and on part
      C's rubric item R8 for those two.
- **Cheap check before any of this: where do the false events under solar growth come from?** CPU
  only, about 1 hour.
    - **Data.** Experiment 653's stored injected worlds with solar growth, where the masked arm
      decoded 0.93 false events per injected Primary-year.
    - **Classify each false event** by:
        - the decoded edge width: a step of 1 hour or less, against a ramp of more than 1 day;
        - whether it lies within 14 days of a season-year boundary;
        - the mean CAMS clear-sky index over its first 3 days, as a high-clear-sky run (index 0.7 or
          more) against ordinary days;
        - its daily shape, as the share of its MW in 10:00 to 15:00 UTC;
        - the slow level's misfit to the injected growth at its onset.
    - **Prediction that could fail.** Half or more of the 0.93 are slow ramps or slow-level misfits,
      which the composite trajectory would remove, and under a third fall in high-clear-sky runs.
    - **If the prediction fails** (most are sunny-run steps), the trajectory is not worth building
      for false events. Keep it only for B1's level, and rely on the part A features.
- **Slow level.** Keep experiment 653's masked, frozen slow level, but cut it to a yearly linear
  trend with no seasonal harmonics, because the encoders now hold the seasonal shape. It stays as an
  arm, not as the default.

**B2. Noise from the model.**

- The model outputs a half-hourly mixture.
- Each Primary's block-scale noise unit is κ_i × the model's standard deviation of a 6-hour block
  mean. Compute that standard deviation from the mixture variances, with an order-1 autocorrelation
  fitted per Primary and fold on E1-masked, event-free, out-of-fold residuals.
- The noise unit therefore varies in time: it is higher on cold evenings and sunny middays.
- **Check:** 90% coverage of the 6-hour block means on event-free rows, in 0.87 to 0.93 at 13 or
  more Primaries.

**B3. A fair comparison with arm C.**

- **Rows:** the same 15 Primaries and 2022-01 to 2026-06. The 2019-to-2021 years are used only for
  training the neural model.
- **Decoding:** the same E1-J-sym, and the A scorer if it passes, retuned per residual on that
  residual's own validation worlds with the same grid size.
- **Worlds:** the same seeds, injected into the power, with the residual recomputed per arm.
- **Measures:**
    - the noise unit per Primary (robust 6-hour block difference);
    - found and paired shares per cell;
    - the false-transfer share;
    - real-record events per Primary-year;
    - the decoded rate in the event-free stratum;
    - D5 and D16, including whether the +11-point bias before the event is gone.
- **Predictions that could fail.**
    - The neural noise unit is 15% or more below arm C's at 10 or more of the 15 Primaries.
    - The long and deep found share rises by 0.05 or more.
    - The decoded rate in the event-free stratum falls by 20% or more.
    - The PV term lowers sunny-midday false events at the six MW Primaries by 30% or more.
- **Stop rule.** If neither the noise-unit bar nor the event-free bar holds, keep arm C.

**Cost.** About 5 GPU-minutes per fold, from experiment 640's 2.5 GPU-minutes per cutoff with margin
for 7 million rows. 8 folds × 2 seeds come to about 1.5 GPU-hours, ceiling 2. One GPU process at a
time; the maintainer's backtests keep priority. A CPU-only fallback is feasible at about 1 to 2
hours per fold on 2 cores, about 16 CPU-hours in all. The decoding re-runs take about 6 CPU-hours.

## C. Cutting false positives, with a Claude reviewer

**Automatic steps first, cheapest first.**

1. The A scorer, with synchrony at both ends if A0 supports it.
2. The shape match.
3. A common-mode and weather veto, through the context features rather than a hard rule.
4. Competition: one partner per end, with the best competitor's score as a feature.
5. A tiered output:
    - "confirmed": both ends seen, synchronous, and a score of 0.8 or more;
    - "probable";
    - "candidate".

**The manual review.**

- **One figure per candidate**, panels stacked on a shared time axis:
    1. The donor: observed power, the expected-power median, and its 90% band.
    2. The donor's residual in noise units, with the decoded steps.
    3. The recipient's two panels, the same as the donor's.
    4. Small multiples of the other neighbours' residuals, at both ends.
    5. The all-Primaries median residual.
    6. ERA5 temperature and the CAMS clear-sky index, with PV-high hours shaded.
- **Time ranges.**
    - An overview of daily means from 7 days before the onset to 7 days after the end, capped at 60
      days.
    - Two zooms at half-hour resolution: the onset ±48 hours and the end ±48 hours.
- **What is hidden.** No NGED labels, no injected-truth marks, and no generator names. Every case is
  rendered identically.
- **Fixed rubric.** Score each item 0 to 2:

| Item | Question |
|---|---|
| R1 | Is the step instant (1 hour or less) at both ends? |
| R2 | Do the donor and recipient start together, and end together? |
| R3 | Are the signs opposite, with a ratio of 0.3 to 2? |
| R4 | Does each Primary return to its earlier level? |
| R5 | Does the recipient's gained profile look like the donor's lost profile? |
| R6 | Does the all-Primaries median or the weather explain the step instead? |
| R7 | Does another neighbour show a better match? |
| R8 | Is there a PV or sunny-run confound? |
| R9 | Are the steps quantised like a generator's? |

The verdict is one of: transfer (with direction), solo shift, generator or routine, weather or
common mode, PV, or unclear. The reviewer adds a confidence from 1 to 5 and writes the output as
JSON.

- **Scoring the reviewer first, on a blinded calibration set of 400 figures:**
    - 150 injected true transfers;
    - 150 of the scorer's highest-scoring false pairs on injected worlds;
    - 100 hard negatives: placebo pairs, routine steps, and sunny runs;
    - plus D5 and D16, and D3 and D22.
- **Gate.** Precision of 0.80 or more at recall of 0.60 or more, direction correct for 90% of
  transfers, and D5 and D16 called a transfer.
- **Visual realness check.** A separate reviewer is shown 100 mixed figures and asked "injected or
  real?". Its accuracy must be 60% or less.
- **Model tier.**
    - Run Sonnet and Opus on the calibration set. Use Sonnet if its precision is within 0.05 of
      Opus's.
    - On real candidates, use two independent Sonnet reviewers, with Opus only on their
      disagreements.
- **Real-record sample.**
    - Every candidate scoring above the validation threshold in the long and the medium cells,
      expected at about 50 to 300 for 2022 to 2026.
    - 50 drawn at random from the lower tiers.
    - 50 random event-free windows, to estimate misses.
- **Cost.** About 4,000 tokens per review. The calibration set at two tiers plus about 400 real
  reviews with two Sonnet reviewers come to about 5 million tokens. Rendering is a few CPU-minutes.
- **Feedback.**
    - Reviewed real labels are first a validation set for the scorer, split by year.
    - They join training only after the gate holds, at a weight of 0.2, with label noise set from
      the reviewer's measured precision, and never in the year they score.
    - Calls of "transfer" with confidence 4 or more go to NGED as a list of candidates.
- **Limits.**
    - The reviewer sees the same residual as the scorer, so an unmetered partner is invisible to
      both.
    - Its injected-set precision carries over to real events only if injected figures look real.
    - NGED's labels are incomplete, so the reviewer's real-record calls cannot be checked against
      them. They are opinions with a measured injected-set precision.
- **Stop rule.** If the reviewer fails the gate at both tiers, drop the manual review and keep the
  automatic tiers.

## D. Causal features for the production forecast

**Per lag (the day lag and the week lag), at the lag's own time, from readings at least 24 hours
before the origin.**

| Feature | What it is | Needs at inference time |
|---|---|---|
| D1 | The common-mode-removed own gap: the gap minus the all-Primaries median gap, in noise units | All Primaries' power and lag-free twins, already in the K chain |
| D2 | The mirror at a neighbour: max over neighbours of −sign(own gap) × the neighbour's gap, in noise units, with the onset-time difference of the best mirror | The neighbours' power |
| D3 | The routine-step rate per Primary over the last 60 days | The Primary's own history |
| D4 | Causal step statistics at 6, 24, and 72 hours: mean gap over the last w minus over the w before that, at the lag's time and at the newest reading | The newest readings |
| D5 | "Lag inside a suspected event": a causal cumulative-sum detector on D1, with time since the suspected onset and the suspected depth | The newest readings |
| D6 | For the prevailing head: the return towards the earlier level, (current gap − gap at the suspected onset) / suspected depth, and a mirror reversal at the neighbour | The newest readings |

**How the features are learnt.**

- D1 to D5 enter the lag encoder per lag, beside experiment 606's gap signals, with 50% signal
  dropout, as in experiment 610.
- The synthetic shifts come from experiment 660's generator, applied to the lags only. The target is
  never shifted (experiment 590).
- D4 to D6 enter experiment 650's departure head and experiment 652's R-haz head.
- **The prevailing head gains most from fresher readings.** Also report, in lab conditions, the same
  features with readings up to 6 hours before the origin, to measure what lower latency would be
  worth.

**Cheap test, on CPU, with no refit.**

1. On the stored banks of experiments 650 and 652, with K1 inference already on file, compute D1 to
   D6 causally per origin.
2. **Separability.** The AUROC of "this lag lies inside an injected shift", for logistic regressions
   on:
    - experiment 606's summaries;
    - those summaries plus D1 to D5.
3. **The prevailing head.** Refit experiment 652's B-haz and R-haz heads, which are small and fit on
   CPU, with and without D4 to D6. Score them in the origin strata of experiment 652.

**Predictions that could fail.**

- The AUROC rises by 0.05 or more on bank S and bank X, as well as on bank M.
- In-force scaled CRPS improves by 0.30 or more against R-haz.
- Near-end and just-ended scaled CRPS improve by 0.30 or more.
- Event-free cost rises by no more than 0.02 against K1.

**Stop rule.** If the AUROC gain is under 0.03 and no prevailing stratum gains 0.15, do not refit
the lag encoder. Otherwise fine-tune K1 with D1 to D5 on 4 screen folds. That costs about 2
GPU-hours, one process at a time, and the backtests keep priority.

**Cost.** CPU only, about 3 to 5 hours.

**Dependencies.** The banks of experiments 650 and 652, the gap-signal code of experiments 606 and
610, and experiment 660's generator.

## Order and running rules

1. Run A0, then A1 to A4, then C's calibration set. These are CPU only and can run in parallel with
   B1 when the GPU is free.
2. Run D's cheap test independently.
3. Re-decode B3 only after A has its verdict, so the comparison of residuals uses the final decoder.

All parts:

- Use `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`.
- Never read the final-test window from 2026-07-01.
- Name no NGED generator.
- Follow the ml-experiment skill's review order.
- Use one experiment id per part.
- Do not change the event-table run of 2022-01-01 to 2026-10-31 that is already in progress with
  E1-J-sym. Its outputs become the real-record input to A and C.

## Maintainer amendment to part D (2026-11-10)

The "this lag is inside a suspected event" flag need not be an explicit flag. Give the lag encoder
the features that the simple detector uses (per-lag causal step statistics in noise units at several
scales, the gap with the all-Primaries median removed, the opposite-sign neighbour gap, the
routine-step rate, the return-to-level statistic), and let the network learn how to use them. Test
the explicit flag only as a comparison arm: features only against features plus flag.
