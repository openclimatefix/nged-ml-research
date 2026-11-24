---
id: "00724"
slug: msa-mva-history-untangler-part-1-first-build
title: >-
  Can a first, small hindsight model rebuild the signed power of the Primaries that report only
  apparent power, with a calibrated probability of export and calibrated size ranges, on worlds
  where the true signed power is known?
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  Anything else. (1) It stores two private posterior tables per Primary (smoothed and filtered), one
  row per half-hour, and per-world score tables; (2) more than one design defensibly tests the
  hypothesis (with or without a free fast gain, with or without the daylight BSP factor), and the
  design note orders those as later rungs; (3) it imports the hidden Markov code of experiment 720
  and the PV-estimate code of experiments 716 and 717, and the callers of that code cannot be
  named without searching the repository; (4) it touches no serving path or degradation rule; (5) it
  changes no contract. Full set of reviews: plan, code (Opus, before the first run, and a second
  Opus review of the fixed code), results, write-up, prose.
hypothesis: >-
  A switching state-space model with three hints (a prior on D12's solar output scaled by an
  as-of capacity, the fact that solar output is zero at night so that the flow is demand and is
  positive, and the continuity of demand between neighbouring half-hours), with the daylight bulk
  supply point factor switched off and the gain on fast cloud set to 0, returns for each
  half-hour a two-lobe signed posterior. On worlds where the signed power is known by
  construction, the probability of reverse flow is calibrated, the size range of the lobe on the
  true side covers the truth at close to its stated rate, the signed error score beats a version
  that commits to one sign, and the sign beats two references that do not read the meter. A private
  historical series for D12, D22, and the five settled Primaries can then be built.
prediction: >-
  Written before any fit, and proposed by the planner because the design states bars and no
  prediction for this build. In the known-answer worlds the smoothed posterior passes the sign
  reliability bar, both lobe-coverage bars, and the night-import bar, and invents reverse flow on
  under 1% at D5 with the gate forced open. The filtered posterior's reliability term is larger
  than the smoothed one's and passes its bar of 0.02. The references bar is the likeliest miss, at
  injection level 1.0, where the PV prior alone already fixes most midday signs. The placebo
  removes the gain over the references. The lobe-coverage bar is the second likeliest miss, for
  the filtered posterior. The prediction fails if any of these misses.
decision_rule: >-
  Every bar is a named `Final` constant of `e00724_core.py`, set before any real data are read.
  Sign (`SIGN_RELIABILITY_MAX_SMOOTHED` = 0.01, `SIGN_RELIABILITY_MAX_FILTERED` = 0.02,
  `SIGN_BIN_MIN_HALF_HOURS` = 200, `SIGN_BIN_MAX_OFF_DIAGONAL` = 0.1, `DEAD_BAND_POINTS` = 2):
  the reliability term of the Brier score of `p_neg` at most 0.01 for the smoothed posterior and at
  most 0.02 for the filtered posterior, outside the 2-point dead band, with no bin of at least 200
  half-hours off the diagonal by more than 0.1. Lobes (`LOBE_COVERAGE_MIN` = 0.85,
  `LOBE_COVERAGE_MAX` = 0.95): the 5% to 95% range of the lobe on the true side covers the truth
  on 85% to 95% of half-hours. Signed CRPS: below the hard-sign version's, with a paired interval
  above 0. References: the sign area exceeds the reading-blind and the hour-of-day references,
  with the lower end of the paired interval above 0. Night import (`NIGHT_P_NEG_MAX` = 0.05): no
  dark half-hour at D12 or D22 has `P(X < 0)` above 0.05. Invented reverse flow
  (`D5_INVENTED_REVERSE_MAX` = 0.01): under 1% at D5 with `c` = 1. Placebo
  (`PLACEBO_SKY_SHIFT_DAYS` = 28): the sky shifted by 28 days removes the gain over the references.
  Proposed verdict mapping, which the design does not state and the maintainer or manager must
  confirm (decision 12 below): `supported` when every bar passes for the posterior it names;
  `not supported` when the sign or lobe calibration, the invented-reverse-flow bar, the night-import
  bar, or the references bar fails (a failed references bar means the untangler's sign is no better
  than the PV estimate alone, and the work stops before the as-of rung); `invalid` when a
  live-fairness or known-answer check of the code fails; `inconclusive` when the timing gate stops
  the run or a world cannot be built. A failed bar is closed with no retuning. The experiment ports
  nothing, so no smallest gain applies.
benchmark: >-
  The hard-sign version of the same posterior (all mass on the more probable sign, at the observed
  magnitude), the reading-blind reference and the hour-of-day reference of experiment 720 (neither
  reads the meter), the shifted-sky placebo, and D5 with the export gate forced open as the
  no-export control.
row_set_id: >-
  not applicable (no forecast is scored): the untangler labels past half-hours, and no valid time on
  or after 2026-07-01 is read. The design names no row set for this build, so the known-answer
  worlds reuse the rows of experiment 720's rung 0b (folds 2025-12 and 2026-04 to 2026-06), which
  decision 13 below asks the maintainer to confirm.
metric: >-
  The reliability term of the Brier score of the probability of reverse flow; the coverage of the
  5% to 95% range of the lobe on the true side; the signed CRPS of the two-lobe distribution; the
  area under the receiver operating characteristic curve for the sign; and the share of dark
  half-hours with a probability of export above 0.05.
---

**We plan the first buildable version of a method that rebuilds the signed power of a substation
whose meter reports only a magnitude, and we plan how to test it before it is used.** A **Primary**
is a primary substation, which steps the voltage down to 11 kV. A **bulk supply point (BSP)** is the
larger substation that feeds a group of Primaries, and its meter reports signed power. Eight of the
16 Primaries report **apparent power in megavolt-amperes (MVA)**, which has no sign, so a Primary
that exports solar output shows a rise in its reading and not a negative value. The other eight
report **real power in megawatts (MW)**, which has a sign. Folding is what happens when a signed flow
that crosses zero is read as a magnitude: the reading falls to a small floor, the **reactive floor**,
and rises again. Primaries are labelled D3, D8, D12, and so on, and the two BSPs are BSP A and BSP B.

**The method fits one hindsight model per Primary and returns, for every half-hour, a two-lobe
posterior of the signed power.** The **hindsight model** reads the whole record, before and after
each half-hour, and splits each reading into **gross demand** (what the customers would draw with no
solar panels), unmetered solar output, and the flow that results. The **PV prior** is an estimate of
D12's solar output, built from satellite sunshine and an estimated solar capacity, which gives the
model a first guess of the output at each half-hour. A **two-lobe posterior** is a probability
distribution with one lobe below zero (export) and one above zero (import): the probability of
export, then the mean and spread of each lobe, five numbers in all. The **smoothed** posterior uses
every reading before and after the half-hour. The **filtered** posterior uses only the readings up
to the half-hour, with parameters that were still fitted on the whole record, so the filtered
posterior is not fair to a live service. No forecast is made or scored, and every score is a
ceiling for what a live service could do.

**This page is the plan, written before any code runs.** The experiment tests the method on
**worlds with a known answer**: we add the recorded output of a real solar farm to a Primary that
reports MW, fold the result into a magnitude, and check whether the method finds the sign and the
size again. The experiment also builds a private history of D12 and D22 and applies a plain formula
to five Primaries whose reading never nears zero. The verdict is `planned` until the bars below have
been read.

## Status and place in the line

**The experiment is the first build of the untangler, and its design is section 9 of
`mva_untangling_design_v3_amendment_v2.md`.** The design note sits in the private
`data/ml-experiments/briefs/msa_architecture/` folder. It amends `mva_untangling_design_v2.md`
(version 2 of the design of the switching state-space model), `mva_untangling_design_v2_review.md`
(the second Opus review, whose section 8 is the first-slice specification),
`mva_to_mw_addendum_1.md` (the hindsight-model addendum, which defines the known-answer tests V1 to
V4 and K1), and `mva_prior_experiments_review.md` (the list of earlier sign experiments).
`cams_latency_sources.md` records the latency of the satellite sunshine product.

**The experiment follows [experiment 722](../00722-msa-mva-bsp-closure-and-fast-hint/result.md), the
last experiment to test a sign label, and is one rung of the ladder that version 2 defines.** The
ladder's rung U3 is the hindsight model of experiment 720 with its known-answer rungs V2, K1, and
V3, and rung U4 would test whether an added hint improves the posterior. This experiment is the
build that the amendment frees from rung U4: with the daylight BSP factor off and the fast gain `a`
at 0, rung U4 has no hint to add (decision 9 below). Experiment 720's code sits on the unmerged
branch `msa-mva-to-mw-rung-0` (pull request 412), and the PV-estimate code of experiments 716 and
717 sits on the unmerged branches `msa-f7-pv-part-1` and `msa-f7-pv-part-2`. This experiment imports
the three folders through `PYTHONPATH` and copies none of them.

## The model and its inputs

**The untangler is one hidden Markov model per Primary, with a gridded demand residual, and the
first build uses three hints and no other.** The reading is
`m = sqrt(X² + Q²)`, where `X` is the signed flow in megawatts (positive for import) and `Q` is the
reactive floor. The flow is `X = D − PV`, where `D` is gross demand and `PV` is unmetered solar
output. The generator-count state of experiment 720 stays at zero counts (`K` = 0) in this build,
so the hidden state is the residual's grid bin only (41 states).

- **The PV prior at D12:** `PV_t = c × C × u_slow,t`. The scale `c` is fixed at 1 in every fit that
  decides a sign, because a scale fitted on the rows that it labels invented reverse flow in
  experiments 358, 362, 412, and 417. The capacity `C` is drawn from a prior that spans two values:
  the as-of capacity of experiments 716 and 717 (carrying the flag "capacity look-ahead", because
  that capacity was fitted through the fold on D12's own readings in 7 of 8 folds), and the capacity
  table that job 1 of experiment 722 found for D12, which is independent of D12's readings. The slow
  term `u_slow,t` is the clear-sky solar yield times a centred 2.5-hour running median of the
  clear-sky index. The fast term is dropped (`a` = 0), so half-hour cloud is ignored.
- **Night import:** the PV term is 0 in the dark, so the flow is gross demand and is positive. The
  hint fails where a generator runs at night, which is why D8 is outside this build.
- **Continuity:** gross demand is a regression mean on half-hour of day, day type, holidays, and
  recorded temperature, plus a first-order autoregressive residual bounded at 0.98. The reading's
  emission is the pair form of experiment 720, which averages `sqrt(X² + Q²)` along the path between
  consecutive half-hours, because D12's crossing half-hours read 5 to 15 points of capacity.
- **Off by design:** the daylight BSP factor (no BSP meter is read, so injecting a metered farm into
  a host counts no megawatt twice), and the fast gain `a`.

**Every input is already stored, and no download is needed.**

| Input | Source | Note |
|---|---|---|
| Readings | The power table, through `studies.power.scan_power()` | Period-end stamps; nothing on or after 2026-07-01 |
| PV estimate and as-of capacity | Experiments 716 and 717 | D12's capacity carries the "capacity look-ahead" flag |
| Second capacity value | The capacity table found by job 1 of experiment 722 for D12 | Independent of D12's readings |
| Sunshine yield, clear-sky yield, and sun elevation | The frame of experiment 39 and the CAMS files of experiment 412 | Hourly index; sufficient with `a` = 0 |
| Temperature | Recorded temperature for the mean of gross demand | The stored ENS table holds 67 leads, for the later as-of rung |
| Switching-log windows | The switching-log workbook | Sets the `transition` flag; read with `engine="openpyxl"` (decision 10 below) |
| Hindsight model code | Experiment 720: the pair emission, the gridded residual, and the forward-backward pass | Imported through `PYTHONPATH`, never copied |

**The first scope is D12 and D22 through the model, and D2, D3, D4, D19, and D21 by formula.** D8
waits for the D8 scope, because night import fails where a generator runs at night. The five
settled Primaries never read near zero (their lowest readings are 16.1 to 32.6 points of capacity),
so the physical flow never crosses zero. For each of the five, `p_neg` is 0, `mu_pos` is
`sqrt(max(m² − Q², 0))`, and `sd_pos` combines the reading noise with the uncertainty of `Q`. The
formula inverts the instantaneous fold and not the path mean, an error under 0.01 points above a
reading of 16 points.

**Parameter uncertainty enters by mixing 12 refits per Primary.** Each refit uses one block
bootstrap of whole months, one draw of `C` from the prior of `C`, and one draw of `Q` from the
profile likelihood of `Q`. The capacity and the floor are the two largest structural unknowns, and a
bootstrap of the other parameters alone would leave both out. The posterior is also reported under
each capacity value alone, because D12's capacity is the weakest input.

**Three changes to experiment 720's code are needed, and the page records where each lands.**

- The forward pass must return the filtered state probabilities. The `Posterior` class of
  experiment 720 already holds `filtered`, so this change needs a check and no new code.
- The five numbers must be integrated from the mixed density on each side of zero. The new module
  `e00724_five.py` does so, and the numbers are never read from a state's point value.
- The running median inside `u_slow` is centred on the half-hour, so the filtered posterior needs a
  trailing form of the median. The module `e00724_model.py` builds both forms of `u_slow`.

## Smoothed and filtered posteriors: what differs

**Both posteriors come from one fit and differ in the passes that produce them.** The smoothed
posterior is the forward-backward pass. The filtered posterior is the forward pass alone, taken at
the same fitted parameters, with the trailing form of the running median in `u_slow`.

**Neither posterior is live-fair, and the page says so wherever it quotes a filtered score.** Four
inputs of the filtered posterior look ahead of a live origin: the parameters (fitted on the whole
span), the capacity (the latest table, and D12's capacity has look-ahead in 7 of 8 folds), the
temperature (recorded, not forecast), and the sunshine (every half-hour, with no publication lag).
Every stored row carries the kind of posterior (`smoothed` or `filtered`) and the flag "capacity
look-ahead" for D12.

**The as-of posterior of rung U5 differs from the filtered posterior in six inputs, and this
experiment builds none of the six.** The table below is the specification of rung U5, kept here so
that a reader sees exactly what the filtered posterior of this experiment lacks.

| Input | Smoothed (this experiment) | Filtered (this experiment) | As-of at origin `o` (later rung) |
|---|---|---|---|
| Readings | Every reading before and after the half-hour | Readings up to the half-hour | Readings up to the last half-hour that experiment 713's 27-hour power rule lets the MSA network see |
| Parameters and era table | Whole span | Whole span | Fitted on the fold's training window; an era that starts after the cutoff does not exist |
| Sunshine (CAMS) | Every half-hour | Every half-hour | Half-hours published by `o` under the dated lag below |
| Capacity | Latest table | Latest table | The as-of capacity table; D12's rows flagged "capacity look-ahead" |
| Temperature in the demand mean | Recorded | Recorded | For each past half-hour, the shortest-lead ENS forecast from a run available at `o` |
| BSP meter and neighbours | Not read | Not read | The same 27-hour rule as the Primary's own reading |

**The sunshine lag, which only the as-of rung uses, depends on the date.** `L` is the number of
hours after the end of a UTC day at which CAMS for that whole day is in hand, and the MSA origin is
09:00 UTC on day `D`, with the last visible reading at 09:00 on `D − 1`. The service's documentation
gives "1 day delayed mode" and a 2026-02 entry "Service provision with 1 day latency instead of 2
day latency", and names no hour of publication (`cams_latency_sources.md`).

| Case | CAMS in hand at the origin | Readings with no CAMS | Consequence |
|---|---|---|---|
| 1: `L` ≤ 9 h | Through the end of `D − 1` | None | CAMS in every row |
| 2: 9 h < `L` ≤ 33 h | Through the end of `D − 2` | 00:00 to 09:00 on `D − 1` | The newest morning uses PV from ENS, with a wider PV error and `a` = 0 |
| 3: 33 h < `L` ≤ 57 h | Through the end of `D − 3` | 33 hours | History age 2 and half of age 3 use ENS-based PV with `a` = 0 |

**The default case is case 2 for origins before 2026-03-01 and case 1 from that date, and case 3 for
every date is the conservative bound.** Case 1 for a 2025 origin would describe a service that did
not exist then, and case 1 is unconfirmed even after the change, because the documentation gives no
hour. In this experiment both posteriors read CAMS throughout, so every stored row records the case
of `L` as "hindsight".

## The stored output

**Each posterior is stored as five numbers per Primary and half-hour, in the private data store
and nowhere else.** The five numbers are the probability of reverse flow `p_neg = P(X < 0)`, the
mean `mu_neg` and standard deviation `sd_neg` of `X` given `X < 0` (null when `p_neg` is 0), and the
mean `mu_pos` and standard deviation `sd_pos` of `X` given `X ≥ 0` (null when `p_neg` is 1). The
mean of the signed power is `p_neg × mu_neg + (1 − p_neg) × mu_pos`, and the variance follows
exactly from the five numbers.

- **The numbers are integrals of the final mixed density,** computed after the per-state kernel and
  after the mixture over the 12 refits, and never read from a state's point value.
- **Quantiles need a shape.** The default shape for a lobe is a normal truncated at zero, matched
  to the lobe's two moments. A lobe whose coefficient of variation exceeds 0.8 takes a gamma
  distribution matched to the same two moments, and a stored flag marks the fallback. The page
  reports the share of lobes that take the fallback and the largest quantile error of each shape
  against the discrete posterior at 11 quantile levels.
- **The visibility flag takes one value per half-hour:** `settled`, `anchored` (a dark half-hour at
  a Primary with no generator), `modelled`, `no_reading`, `transition` (inside a logged switching
  window), and `added_back`.
- **Each row also records** the kind of posterior, the case of `L` ("hindsight" here), the
  parameter cutoff, the "capacity look-ahead" flag, and the code hash.
- **Two optional channels are stored apart:** the posterior mean and standard deviation of the PV
  term, and a generator-state channel. The generator-state channel is empty in this build, because
  `K` = 0. **An inferred generator-state series is a time series of that generator and stays in the
  private data store.** The PV channel on dates also stays private until the maintainer decides to
  publish it (decision 14 below).

## Known-answer tests, placebos, and calibration

**Four kinds of world have a signed truth by construction, and the bars are those of version 2 and
of section 9 of the amendment.** No BSP meter is read in any world.

| World | Truth | Role |
|---|---|---|
| The six signed hosts, each with an injected farm at levels 1.0 and 2.0 (V1) | The host's signed reading minus the farm's output | Sign, lobes, and signed CRPS |
| D10 folded | D10's signed readings | Real export on 58 crossing days, by version 2 |
| The six un-injected, with D5 forced open | No reverse flow | Invented reverse flow under 1% at D5 with `c` = 1 |
| D15 folded | D15's signed readings | Report only: D15 exports at night, so night import does not hold |

**An injected world adds a metered solar farm's recorded output to a host's signed reading and
folds the sum** with `Q` of 2 points of capacity and the within-half-hour averaging of experiment
720. The six hosts are D5, D9, D10, D11, D14, and D16.

**The bars are listed in the decision rule above and are repeated here, one per row, with the
constant that holds each.**

| Test | Bar | Constants |
|---|---|---|
| Sign | The reliability term of the Brier score of `p_neg` at most 0.01 for the smoothed posterior and at most 0.02 for the filtered posterior, outside the 2-point dead band, with no bin of at least 200 half-hours off the diagonal by more than 0.1 | `SIGN_RELIABILITY_MAX_SMOOTHED`, `SIGN_RELIABILITY_MAX_FILTERED`, `DEAD_BAND_POINTS`, `SIGN_BIN_MIN_HALF_HOURS`, `SIGN_BIN_MAX_OFF_DIAGONAL` |
| Lobes | The 5% to 95% range of the lobe on the true side covers the truth on 85% to 95% of half-hours | `LOBE_COVERAGE_MIN`, `LOBE_COVERAGE_MAX` |
| Signed CRPS | Below the hard-sign version's, with a paired interval above 0 | `SIGNED_CRPS_PAIRED_LOWER_MIN` = 0 |
| References | The sign area exceeds the reading-blind and the hour-of-day references, with the lower end of the paired interval above 0 | `REFERENCE_PAIRED_LOWER_MIN` = 0 |
| Night import | No dark half-hour at D12 or D22 has `P(X < 0)` above 0.05 | `NIGHT_P_NEG_MAX` |
| Invented reverse flow | Under 1% at D5 with `c` = 1 | `D5_INVENTED_REVERSE_MAX` |
| Placebo | The sky shifted by 28 days removes the gain over the references | `PLACEBO_SKY_SHIFT_DAYS` |

**Three limits of the first buildable version are known in advance.** The sign within about 2
points of zero stays undetermined. The split between PV and demand inside a daylight stretch is
driven by the prior. D12's capacity is the weakest input, so the posterior is reported under each
capacity value and under the mixture. A failure of the references bar means that the untangler's
sign is no better than the PV estimate alone, and the page says so.

**Four known-answer tests of the addendum are not run here, and each waits for a later scope.**
V2 (injected generator schedules), K1 (D8's own record with the generator subtracted), and V3
(real negative controls at D8) need `K` above 0 and the D8 scope, and they belong to rung U3, which
experiment 720 owns. V4 (read-outs at D12 and D8 that need no truth, reported with no bar) runs here
at D12 only, as aggregates: `P(X < 0)` in the dark, and the monthly share of export days. V4 at D12
is gated on V1 passing, as the addendum requires.

**Calibration is reported for both posteriors, and each lobe's coverage is also reported within each
third of the reading's level.** Version 2 states a range of 80% to 97% within each third as a
report, and this page lists it with no bar because section 9 of the amendment states only the pooled
bar (decision 15 below). Reliability diagrams use 10 bins of `p_neg`.

**Live-fairness checks do not apply to this experiment and are written for the later rung.** The
four checks of version 2 section 8.3 (perturbation, truncation, parameter cutoff, and poison) test
the as-of posterior. The filtered posterior of this experiment is labelled "not live-fair" instead.

## Compute, launch rules, and gates

**The build needs about 2 core-hours on one CPU core and no GPU, and the page sets a cap of 3
core-hours.** Experiment 720 measured 53 s per whole-span pass at 84 states. At 41 states and 10
passes per fit, one fit takes about 2 minutes. The design counts 12 refits at each of D12 and D22
(24 fits) and about 14 fits for the known-answer worlds, which is about 1.3 core-hours. The
placebo worlds (about 14 fits), the capacity-value arms at D12 (4 fits), and the D5 forced-open
world are not in that count, so the planner's estimate is about 2 core-hours and the cap is 3
(decision 16 below). Every figure is a designer's estimate until the first fit is timed.

**Every job follows the launch rules of version 2, and the implementation brief restates them.**

- One `systemd-run --user --scope` at a time, on one pinned core, with `MemoryMax=12G`,
  `MemorySwapMax=0`, and `OOMPolicy=continue`.
- One thread for BLAS and Polars, and `CUDA_VISIBLE_DEVICES=""`.
- A check for 25 GB of free memory before launch, made by the launch script.
- A timing gate: the first fit of each job is timed and the job stops when the projection exceeds the
  job's cap.
- A check-in with the ML manager before each job, and a "fits starting" line to the manager before
  each family of fits (the worlds, the placebos, and the builds at D12 and D22).

**Six jobs run in order, each under 1 core-hour.**

| Job | Content | Core-hours (estimate) | Stops the series when |
|---|---|---|---|
| 1 | Input checks, switching-log windows, the three code changes, and the timing of one fit | Under 0.1 | An input is missing or the timing gate projects more than 3 core-hours |
| 2 | Known-answer worlds, smoothed and filtered (V1, D10 folded, D5 forced open, D15 folded) | About 0.5 | Never |
| 3 | Placebo worlds (sky shifted by 28 days) | About 0.5 | Never |
| 4 | Scoring of every bar and the V4 read-outs at D12 | Under 0.1 | Never |
| 5 | The build at D12 and D22, with 12 refits each, and the formula for the five settled Primaries | About 0.7 | The references bar of job 4 failed (decision rule: the build stays unpublished) |
| 6 | Report, figures, and the private tables | Under 0.1 | Never |

**Two reviews come before any run that produces numbers the page quotes.** An Opus code review of
all new code comes first, because the experiment imports unmerged code, builds a new emission
integral, and changes a running median. A second Opus review re-reads the fixed code before the long
runs, because the experiment's first review is expected to flag problems in code that has
never run on real data. The code review checks leakage
(the fit never reads a reading on or after 2026-07-01), the forward and backward passes against
experiment 720's, the integral of the five numbers, and that the trailing median looks back only.
The manager is told "fits starting" before the first fit, and no job starts without the manager's
reply.

## What the output would change downstream

**A pass would give a private, probabilistic series of past signed power, in megawatts, for D12 and
D22, and a formula series for five more Primaries.** The series has three possible uses, and this
experiment delivers none of them.

- **NGED historical series with uncertainty:** whether any series is delivered to NGED, and in which
  form, is a decision for the maintainer (decisions 11 and 17 below). The page does not offer one.
- **Later history channels for the MSA network:** a posterior arm for the MSA network would carry
  `p_neg`, the four lobe moments, the visibility flag, and a mask, and is deferred. The fully
  smoothed posterior is never an input to a scored forecast. Experiment 721 keeps its registered
  design.
- **Analyses of the network's losses at D12:** the posterior replaces the 10-point heuristic flag in
  a re-cut of experiment 713's stored scores, which is also deferred.

**A fail changes the plan less than a pass.** A failed references bar means the sign adds nothing to
the PV estimate alone, so the untangler would not be built further, and the maintainer decides
whether the series is still worth keeping.

## Stop rule, counted against earlier attempts

**This experiment runs once, and a failed bar is closed with no variant.** A code defect found
before a number is quoted is fixed and the job reruns as the same attempt. The thresholds, the
scale `c`, and the sensors are never changed for a second attempt.

**The count of earlier attempts is the reason for the single run.** Version 2 lists 18 numbered
experiments that produced or gated a sign label (79, 152, 358, 362, 365, 366, 370, 387, 408, 412,
417, 420, 424, 427, 506, 508, 543, and 674). Three experiments tracked the generator behind D8 (694,
695, and 696), and none gave a usable label at D12 or D8. Experiment 722 was the fourth use of BSP
closure for a sign and the 19th experiment that produced or gated a sign label, and it was
inconclusive. The amendment's follow-up experiment would be the 20th. This build adds no new sign
hint beyond a structural prior, night import, and continuity, so the amendment does not number it
(decision 18 below). If the maintainer counts it, it is the 20th or 21st attempt, and the rule that
no third experiment on either of the two open ideas starts without the maintainer's recorded
approval is not affected, because this build tests neither the daylight BSP hint nor the within-day
hint.

**The series stops when the references bar fails, and the work does not proceed to the as-of rung.**
A failed references bar means the untangler's sign is no better than the PV estimate alone.

## Decisions for the maintainer or manager

**The amendment lists ten decisions (section 10 of `mva_untangling_design_v3_amendment_v2.md`), and
the table says how each one bears on this build.** None is taken by this page. Decisions 11 to 18
are new questions that the plan raised.

| No. | Decision (amendment's wording, shortened) | Bears on this build as follows |
|---|---|---|
| 1 | Whether the follow-up experiment of jobs W0, W2, and W3 runs as the 20th experiment | Indirectly: its outcome decides whether a later rung adds a within-day arm. This build neither waits for it nor depends on it |
| 2 | The stop rule and the outcome table of section 8 | Directly: this build keeps the single-run rule and the count above, and takes the "daylight BSP factor off" outcome as given |
| 3 | Flaw 1: drop the D15 sign test and its placebo | Not concerned, except that D15 is folded here as report only |
| 4 | Flaw 2: remove every size bar and keep dark-hour closure as a read-out | Not concerned, because this build reads no BSP meter |
| 5 | Flaw 3: the rule for future generator records | Not concerned, because `K` = 0 |
| 6 | Flaw 4 and rung U1: remove the day class, adopt the farm sensor, admit metered farms as injected farms | Concerned only through the injected farms of the V1 worlds, which are the farms of experiment 39's frame and are not shown |
| 7 | The form of the daylight hints in rung U4 | Not concerned: the daylight BSP factor stays off and `a` = 0 |
| 8 | The CAMS download (decline for now, or approve 22 requests) | Concerned: this build uses the hourly series, which suffices with `a` = 0, so it proceeds on the "decline" default |
| 9 | Whether the build may follow its known-answer test directly, without a rung U4 that has no hint to test | Directly: the plan assumes yes. If the answer is no, this experiment is held |
| 10 | The switching-log repair: `engine="openpyxl"` in the experiment's own code, with no change to `pyproject.toml` | Directly: the plan uses the repair to set the `transition` flag |

**Eight further decisions came up while planning, and none is in the amendment.**

- **11. Delivery form to NGED.** Version 2 section 10 gives three options (no delivery, D12 and the
  five settled Primaries, or all eight), and recommends building the first scope privately and
  deciding after the known-answer scores. Decision needed: whether any series is offered at all.
- **12. The verdict mapping.** The design states bars and no verdict mapping for this build. The
  mapping in the decision rule is the planner's proposal.
- **13. The rows of the known-answer worlds.** The design names no row set. The plan reuses the
  folds of experiment 720's rung 0b.
- **14. Publication of D12's signed posterior on dates.** Version 2 allows it subject to a check
  that no single metered farm dominates D12's PV. Job 1 of experiment 722 found none, but the plan
  keeps every dated series private until the maintainer confirms.
- **15. The within-third lobe coverage.** Version 2 states 80% to 97% within each third of the
  reading's level, and section 9 of the amendment states only the pooled bar. The plan reports the
  thirds with no bar.
- **16. The compute cap.** The design's 2 core-hours omits the placebo and capacity-value fits.
  The plan sets a cap of 3.
- **17. Contract decisions that belong to a human in the production repository.** Production code
  is not touched here, and the following stay open: which unit label a series carries when the
  observation is in MVA and the series is in signed megawatts, and whether the power-factor
  assumption is stated beside the label; whether the sign convention extends to MVA series, and how
  an inferred sign is marked; how the production metrics score a signed series against an unsigned
  observation, and whether the clamp at zero stays for the eight; whether the five-number posterior,
  a quantile set, or a mean with a flag is the stored form; which effective capacity in megawatts
  normalises D12; and whether a fitted `Q` per Primary may replace the assumed small floor.
- **18. Whether this build counts toward the number of sign experiments,** and so whether a failure
  is the 20th or the 21st attempt on the line.

## Checks

- **Median forecast minus mean forecast per site, and the error of the mean or median forecast beside
  each quantile-score contrast:** not applicable, because no forecast is scored.
- **Audit triggers:** none has fired, because nothing has run. A failed known-answer test or
  placebo, or a result that contradicts the prediction above, triggers an audit before the write-up.
- **Anonymisation:** the page and its commit hold no generator name, identifier, type, unit size, or
  dated generator value, and no BSP name other than BSP A and BSP B. The V1 worlds add the output of
  real metered solar farms, and no page shows any two of a host's reading, an injected reading, and
  the signed truth on shared dates.

## Method

The experiment is planned and has no result yet, so these bullets describe the design.

- **Architecture:**
    - **Model:** a hidden Markov model per Primary, with a gridded autoregressive residual of gross
      demand (41 states, `K` = 0), the pair emission of experiment 720 with Student-t noise of 4
      degrees of freedom, and the three hints above. The model is fitted by expectation-maximisation
      on the CPU.
    - **Inputs:** the Primary's reading, the clear-sky solar yield and the hourly clear-sky index,
      the as-of capacity of experiments 716 and 717, the second capacity value, recorded
      temperature, and calendar. No BSP meter, sibling Primary, or generator record is read.
    - **Reused unchanged:** the forward-backward pass, the pair emission, and the gridded residual
      of experiment 720, and the PV estimate of experiments 716 and 717.
    - **Added here:** the capacity mixture, the trailing and centred forms of the slow term, the
      integral of the five numbers, and the formula for the settled Primaries.
    - **Information after the origin:** the filtered posterior uses the forward pass alone but is not
      live-fair, for the four reasons in the section "Smoothed and filtered posteriors".
- **Training regime:**
    - **Data:** past readings up to the holdout, with the half-hours that the `substation_zero` rule
      flags dropped, and every D15 and D18 row before 2025-03-01 cut. The number of dropped
      half-hours per Primary is reported.
    - **Injected worlds:** a real metered solar farm's recorded output is added to a signed Primary
      at levels 1.0 and 2.0 and folded with `Q` of 2 points. The farm's output enters as private
      data only.
    - **Fitting:** about 10 passes per fit, 12 refits per Primary with a month-block bootstrap, a
      draw of `C`, and a draw of `Q`, with `c` fixed at 1. No seed other than the bootstrap seed
      exists, and it is fixed.
    - **Resources:** CPU only, one core, no GPU, 2 core-hours estimated and 3 as the cap.
- **Measurement:**
    - **Rows:** the folds of experiment 720's rung 0b, with no valid time on or after 2026-07-01.
    - **Scores:** the metrics of the front matter, for the smoothed and the filtered posterior, each
      against its truth by construction. Folded CRPS at D12 is not reported, because the smoothed
      posterior has read the reading it labels.
    - **Intervals:** paired intervals come from 1,000 bootstrap replicates that resample whole
      days, at the 95% level.
    - **Bars and verdict:** the decision rule above, with every bar a named constant set before any
      real data are read.
    - **Controls:** the 28-day sky shift, D5 forced open, D10 folded, and D15 folded.
    - **Standing practices followed:** cleaned data and the short-history cut. Not applicable:
      monthly refit of a network, saved network states, and forecast plots.

## Reviews

An Opus code review of all new code comes before any number is quoted, and a second Opus review of
the fixed code comes before the long runs. A Sonnet review checks the results and the write-up
against the printed report, and a second Sonnet reviewer runs the prose review. The plan has not
yet been reviewed.

## Files

Planned files only; none exists yet.

- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_core.py`: the named
  constants of every bar, the dead-band and bin helpers, the bar functions, and the verdict rule.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_inputs.py`: loads the
  readings, the PV estimate and as-of capacity from experiments 716 and 717, the second capacity
  value, the CAMS files, the temperature, and the switching-log windows. Job 1.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_model.py`: the capacity
  mixture, the centred and trailing forms of the slow term, and the call of experiment 720's
  forward-backward pass. Jobs 2, 3, and 5.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_five.py`: integrates the
  five numbers from the mixed density, the truncated-normal and gamma lobe shapes, and the formula
  for the settled Primaries.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_worlds.py`: builds the
  injected, folded, and forced-open worlds and their placebos. Jobs 2 and 3.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_score.py`: the
  reliability term, the lobe coverage, the signed CRPS, the reference areas, and the paired
  bootstrap. Job 4.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_build.py`: the 12 refits
  at D12 and D22, and the writer of the private posterior tables. Job 5.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_guard.py`: the holdout
  guard, the output-path guard, and the anonymisation guard that refuses to write a dated
  generator-derived value outside the private folder.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_report.py` and
  `e00724_figures.py`: the printed report and the charts of pooled reliability and coverage. Job 6.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/run.sh`: the launch wrapper,
  which runs no job without `--go`, and sets the scope, memory, thread, and free-memory checks.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/tests/`: tests on synthetic
  arrays only, with no real data.
- `ml-experiments/00720-msa-mva-to-mw-rung-0-signed-power-from-mva-meters/` (read, on
  `PYTHONPATH`): `e00720_hmm.py`, `e00720_fit.py`, and `e00720_fold.py`. The experiment does not
  link the page of experiment 720, because the page is on an unmerged branch.
- `ml-experiments/00716-msa-unmetered-pv-part-1-history-estimate/` and
  `ml-experiments/00717-msa-unmetered-pv-part-2-target-row-forecast/` (read, on `PYTHONPATH`): the PV
  estimate and the as-of capacity table.
- `data/ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/` (written): the private
  posterior tables (smoothed and filtered, five numbers, visibility flag, kind, case of `L`,
  parameter cutoff, capacity flag, code hash), the per-world score tables, and the run logs. Dated
  detail is in the `PRIVATE/` subfolder.
- `data/ml-experiments/briefs/msa_architecture/` (read, private): the design files named in the
  section "Status and place in the line".
