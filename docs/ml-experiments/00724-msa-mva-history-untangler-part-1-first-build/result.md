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
  Written before any fit, and fixed by erratum 1. It can fail on any line. The smoothed posterior
  passes the sign bar, and the filtered posterior's reliability term is 1 to 3 times the smoothed
  posterior's and passes 0.02. The smoothed sign area exceeds the better reference in the pool, by
  more at injection level 2.0 than at level 1.0, and at level 1.0 alone the paired interval
  includes 0. The placebo bar passes, and the shifted-sky posterior keeps a gain over the
  reading-blind reference. Fewer than 1% of sunny midday half-hours at un-injected D5 have `p_neg`
  above 0.5. The likeliest miss is the lobe bar within the lowest third of the reading's level, and
  the second likeliest miss is the pooled lobe bar of the filtered posterior.
decision_rule: >-
  Every bar is a named `Final` constant of `e00724_core.py`, set before any real data are read.
  Sign (`SIGN_RELIABILITY_MAX_SMOOTHED` = 0.01, `SIGN_RELIABILITY_MAX_FILTERED` = 0.02,
  `SIGN_BIN_MIN_HALF_HOURS` = 200, `SIGN_BIN_MAX_OFF_DIAGONAL` = 0.1, `DEAD_BAND_POINTS` = 2):
  the reliability term of the Brier score of `p_neg` at most 0.01 for the smoothed posterior and at
  most 0.02 for the filtered posterior, outside the 2-point dead band, with no bin of at least 200
  half-hours off the diagonal by more than 0.1. The verdict reads the smoothed posterior; the
  filtered posterior's term of 0.02 is read on its own line (`passes`, `fails`, or `not readable`),
  which gates only later work that reads a filtered or as-of posterior. Lobes
  (`LOBE_COVERAGE_MIN` = 0.85, `LOBE_COVERAGE_MAX` = 0.95, `LOBE_THIRD_MIN` = 0.80,
  `LOBE_THIRD_MAX` = 0.97, `LOBE_THIRD_MIN_HALF_HOURS` = 200): the 5% to 95% range of the lobe on
  the true side covers the truth on 85% to 95% of half-hours, and on 80% to 97% within each third
  of the reading's level; a third with fewer than 200 scored half-hours is `void`, is neutral, and
  is listed beside the verdict. Signed CRPS: below the hard-sign version's, with a paired interval
  above 0. References: the sign area exceeds the reading-blind and the hour-of-day references,
  with the lower end of the paired interval above 0. Night import (`NIGHT_P_NEG_MAX` = 0.05): no
  dark half-hour at D12 or D22 has `P(X < 0)` above 0.05, read on the build, on half-hours where
  the clear-sky yield is 0 at the half-hour and at both neighbours, outside rows flagged
  `transition` or `no_reading`. Invented reverse flow (`D5_INVENTED_REVERSE_MAX` = 0.01,
  `D5_MIN_SUNNY_MIDDAY_HALF_HOURS` = 200): the share of sunny midday half-hours (experiment 720's
  mask, `u ≥ 0.5` and within 4 half-hours of slot 24) at un-injected D5 with `p_neg` above 0.5 is
  under 1%, with `c` = 1; under 200 such half-hours the run is `inconclusive`. Placebo
  (`PLACEBO_SKY_SHIFT_DAYS` = 28, `PLACEBO_MIN_AREA_GAIN` = 0.1): the sign area under the real sky
  minus the sign area under the sky shifted by 28 days is at least 0.1, with a paired interval
  above 0, as addendum 1 registered for V1 (the erratum's wording, a paired interval above 0 alone,
  is weaker than addendum 1's size of 0.1, and this page restores the size). The shifted sky moves
  the clear-sky index only and keeps the clear-sky yield at its true date, the contrast is read on
  the rows where the shifted sky exists, and the references use the same shifted sky. Pool: the
  sign, lobe, CRPS, and references bars are read on the 12 injected worlds and D10 folded
  together, with one bootstrap that resamples whole days across every world. Verdict, read on the
  smoothed posterior, first row that matches: (1) a correctness check fails after a number is
  written: `invalid`, rerun under the rule of erratum section 2.1; (2) in jobs 1 to 4, the timing
  gate stops a job after it started, a world cannot be built, the pooled worlds hold fewer than
  200 half-hours of true export outside the dead band, or fewer than 200 sunny midday half-hours
  survive at un-injected D5: `inconclusive`, and job 5 does not run; (3) the sign, references, D5,
  or placebo bar fails: `not supported`, a failed references bar also stops the work before the
  as-of rung, and job 5 does not run; (4) the bars of row 3 pass and the lobe bar (pooled or within
  a third) or the signed CRPS bar fails: `partly supported`, job 5 does not run, and the maintainer
  decides whether a sign-only series is wanted; (5) every known-answer bar passes and the built
  posterior fails night import at D12 or D22: `partly supported`, and the failing Primary's series
  is withheld; (6) every known-answer bar passes and night import passes at D12 and D22:
  `supported`, and the series is stored. Rows 1 to 4 are read after job 4, job 5 runs only when
  none of them matches, and rows 5 and 6 are read after job 5. A stop or failure inside job 5 is
  reported as the build status `not built` beside the known-answer result, and the known-answer
  bars that passed stay on the page. Version 2's gate for the build names rows 5 and 6, not the
  label `partly supported`. A `void` third, and the mixture arm, change no verdict. A failed bar is
  closed with no retuning, and `inconclusive` and `not supported` are never rerun. The experiment
  ports nothing, so no smallest gain applies.
benchmark: >-
  The hard-sign version of the same posterior (all mass on the more probable sign, at the observed
  magnitude), the reading-blind reference and the hour-of-day reference of experiment 720 (neither
  reads the meter), the shifted-sky placebo, and the un-injected D5 world, passed through the
  model like any other world, as the no-export control.
row_set_id: >-
  not applicable (no forecast is scored): every half-hour of the fitted span, and no valid time on
  or after 2026-07-01 is read. The fitted span of a Primary runs from the first day of its current
  era to 2026-06-30. An injected world's span is the days on which both the host's reading and the
  farm's output exist. Job 1 asserts each span's day count, and another count is a code defect. The
  four folds of experiment 720's rung 0b (2025-12 and 2026-04 to 2026-06) are a reported subset.
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
Primaries report **real power in megawatts (MW)**, which has a sign. **Reverse flow**, also called
**export**, means that power flows from the Primary towards the grid, and its sign is negative.
**Folding** is what happens when a signed flow that crosses zero is read as a magnitude: the reading
falls to a small floor, the **reactive floor**, and rises again. A **point** is 1% of a Primary's
capacity. Primaries are labelled D3, D8, D12, and so on, and the two BSPs are BSP A and BSP B.

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

## Status and place in the series of sign experiments

**The experiment is the first build of the untangler, and its design is section 9 of
`mva_untangling_design_v3_amendment_v2.md`, corrected by `mva_untangling_erratum_1.md` (items 9 to
14) and by the Opus review `erratum_and_plans_review.md`.** The notes sit in the private
`data/ml-experiments/briefs/msa_architecture/` folder. Where the erratum and the amendment
disagree, the erratum governs. The amendment amends `mva_untangling_design_v2.md` (version 2 of the
design of the switching state-space model), `mva_untangling_design_v2_review.md` (the second Opus
review, whose section 8 is the first-slice specification), `mva_to_mw_addendum_1.md` (the
hindsight-model addendum, which defines the known-answer tests V1 to V4 and K1), and
`mva_prior_experiments_review.md` (the list of earlier sign experiments). `cams_latency_sources.md`
records the latency of the satellite sunshine product.

**The experiment follows [experiment 722](../00722-msa-mva-bsp-closure-and-fast-hint/result.md), the
last experiment to test a sign label, and is one rung of the ladder that version 2 defines.** The
ladder's rung U3 is the hindsight model of experiment 720 with its known-answer rungs V2, K1, and
V3, and rung U4 would test whether an added hint improves the posterior. This experiment is the
build that the amendment frees from rung U4. The **daylight BSP factor** is the likelihood term
that reads the BSP meter in daylight to label a sign, and the **fast gain** `a` is the weight of
half-hour cloud in the PV term. With the factor off and `a` at 0, rung U4 has no hint to add. The
erratum rules that the known-answer test of this experiment is the case of rung U4 with an empty
list of hints (decision 9 below). Experiment 720's code sits on the unmerged branch
`msa-mva-to-mw-rung-0` (pull request 412), and the PV-estimate code of experiments 716 and 717
sits on the unmerged branches `msa-f7-pv-part-1` and `msa-f7-pv-part-2`. This experiment imports
the three folders through `PYTHONPATH` and copies none of the three folders.

## The model and its inputs

**The untangler is one hidden Markov model per Primary, with a gridded demand residual, and the
first build uses three hints and no other.** The reading is `m = sqrt(X² + Q²)`, where `X` is the
signed flow in megawatts (positive for import) and `Q` is the reactive floor. The flow is
`X = D − PV`, where `D` is gross demand and `PV` is unmetered solar output. The generator-count
state of experiment 720 stays at zero counts (`K` = 0) in this build, so the hidden state is the
residual's grid bin only (41 states). The daylight BSP factor is off and `a` is 0 in the first
buildable version under every outcome (erratum ruling 13).

- **The PV prior at D12 and D22:** `PV_t = c × C × u_slow,t`. The scale `c` is fixed at 1 in every
  fit that decides a sign, because a scale fitted on the rows that it labels invented reverse flow
  in experiments 358, 362, 412, and 417. The capacity `C` has a log-uniform prior between the
  smallest and the largest of two stored values, and the 12 refits take the stratified quantiles
  `(i − 0.5) / 12`. The first value is experiment 662's capacity, which experiments 716 and 717
  already read, and it carries the flag "capacity look-ahead" at D12 (the capacity was fitted
  through the fold on D12's own readings in 7 of 8 folds) and at D22. The second value is the
  registered PV capacity inside the Primary's polygon from experiment 673's table, less the
  capacity of the farms that have their own meters, which is independent of the Primary's
  readings. Where experiment 673's table gives a low and a high variant that both meet this
  definition, both bound the span and the second value is their geometric mean. Job 1 names the
  columns and records the values under `PRIVATE/` before any fit. The slow term `u_slow,t` is the
  clear-sky solar yield times a running median of the clear-sky index over 5 half-hours. The fast
  term is dropped (`a` = 0), so half-hour cloud is ignored.
- **Night import:** the PV term is 0 in the dark, so the flow is gross demand and is positive. The
  hint fails where a generator runs at night, which is why D8 is outside this build.
- **Continuity:** gross demand is a regression mean on half-hour of day, day type, holidays, and
  recorded temperature, plus a first-order autoregressive residual bounded at 0.98. The reading's
  emission is the pair form of experiment 720, which averages `sqrt(X² + Q²)` along the path between
  consecutive half-hours, because D12's crossing half-hours read 5 to 15 points.
- **Off by design:** the daylight BSP factor (no BSP meter is read, so injecting a metered farm into
  a host counts no megawatt twice), and the fast gain `a`.

**Every input is already stored, and no download is needed.**

| Input | Source | Note |
|---|---|---|
| Readings | The power table, through `studies.power.scan_power()` | Period-end stamps; nothing on or after 2026-07-01 |
| PV estimate and as-of capacity | Experiments 716 and 717 | Experiment 662's capacity; D12's and D22's capacity carry the "capacity look-ahead" flag |
| Second capacity value | Experiment 673's table: registered PV capacity inside the Primary's polygon, less metered farms | Independent of the Primary's readings; values under `PRIVATE/` only |
| Sunshine yield, clear-sky yield, and sun elevation | The frame of experiment 39 and the CAMS files of experiment 412 | Hourly index; sufficient with `a` = 0 |
| Temperature | Recorded temperature for the mean of gross demand | The stored ENS table holds 67 leads, for the later as-of rung |
| Switching-log windows | The switching-log workbook | Sets the `transition` flag; read with `engine="openpyxl"` (decision 10 below) |
| Hindsight model code | Experiment 720, imported unchanged (section "The file plan") | Imported through `PYTHONPATH`, never copied |

**The first scope is D12 and D22 through the model, and D2, D3, D4, D19, and D21 by formula.** D8
waits for the D8 scope, because night import fails where a generator runs at night. The five
settled Primaries never read near zero (their lowest readings are 16.1 to 32.6 points), so the
physical flow never crosses zero. For each of the five settled Primaries, `p_neg` is 0. The
formula integrates over a prior on `Q`, which is the least informative density that respects three
facts: `Q` is 0 or more, `Q` is at most `L` (the Primary's lowest valid reading, taken after the
`substation_zero` cleaning and outside rows flagged `no_reading` or `transition`), and the project
holds one registered scale for the floor, 2 points, the value that folds every known-answer world.
The density of greatest entropy on values of 0 or more with a fixed mean is the exponential, so `Q`
is exponential with a mean of 2 points before truncation, truncated at `L + 2 σ_e`, where the
reading noise `σ_e` is the smallest step between distinct stored readings divided by `sqrt(12)`,
which job 1 measures. At the five settled Primaries `L` is 16.1 points or more and the truncation
removes under 0.1% of the mass. At D22, where `L` is 2.6 points, the truncation keeps about 73% of
the mass, and job 1 reports how many of D22's dark half-hours read below 2 × `L`, because the
night-import bar is at risk there. Then `mu_pos` is the prior mean of `sqrt(max(m² − Q², 0))`, and
`sd_pos²` is the prior variance of the same quantity plus `σ_e²` times the prior mean of
`m² / (m² − Q²)`. To first order `mu_pos` is `m − 4 / m`, and the floor adds `8.9 / m` to `sd_pos`:
0.25 and 0.55 points at a reading of 16.1 points. The power-factor assumption moves the five series
by more than the floor does, because a power factor of 0.97 lowers the signed power by 3% of the
reading (1.5 points at a reading of 50 points, against 0.08 points from the floor). The page
reports `mu_pos` under `Q` = 0 and under a power factor of 0.97 as two labelled alternatives with no
bar.

**Parameter uncertainty enters by mixing 12 refits per Primary.** Each refit uses one block
bootstrap of whole months, one draw of `C`, and one draw of `Q`. The 12 refits take `C` at the
stratified quantiles `(i − 0.5) / 12` of its prior, and `Q` at the stratified quantiles
`(i − 0.5) / 12` of the exponential prior times the conditional likelihood of `Q` on a grid, permuted
at random against the strata of `C` with the fixed bootstrap seed. The likelihood is conditional
because the other parameters are held at the fit with `Q` at 2 points. One fit at 2 points is
followed by one forward pass at each grid value, from 0 to the smaller of 7.5 points (3.75 prior
means, which keeps 97.6% of the prior mass) and `L`, in steps of 0.5 points. The capacity and the
floor are the two largest structural unknowns, and a bootstrap of the other parameters alone would
leave both out. The posterior is also reported under each capacity value alone at D12 and at D22,
because D12's capacity is the weakest input.

## The file plan

**The amendment lists three "changes to experiment 720's code", and none changes that code.**
Experiment 720's `forward_backward` takes the PV estimate as an input array, and its `Posterior`
already holds the filtered probabilities. Experiments 716, 717, and 720 hold no code for `u_slow`.

| Imported unchanged from experiment 720, through `PYTHONPATH` | New code in experiment 724's folder |
|---|---|
| `e00720_hmm`: `HmmSpec`, `forward_backward`, `Posterior`, `state_power`, `brier_reliability`, `signed_crps`, `CostGate` | `u_slow`, in a centred and a trailing form |
| `e00720_fit`: `design_matrix`, `fit_demand_mean`, `fit_residual`, `k0_spec` | The whole-span fitting loop, the capacity prior, and the draws of `Q` |
| `e00720_worlds`: `reading_from_signed`, `shift_days` | The five numbers, integrated from the mixed density on each side of zero; experiment 720's `two_lobe_mixture` is not used, because its normal lobes leak mass across zero |
| `e00720_rung0b`: `reading_blind_rho`, `hour_of_day_rho`, `paired_auc_difference` | The whole-span worlds and scoring, the formula for the five settled Primaries, and the guards |

**A function of experiment 720 that is private, or that takes a fold or a `ProbeWorld`, is written
afresh and listed as new in the code-review request.** Job 1 records the file and line of each
imported name. The hour-of-day reference takes every day of the span as a reference day. The code
review also checks that importing `e00720_rung0b` (which imports `e00720_real` and `e00720_sign`)
has no side effect that reads data or names a path under the home directory.

**`shift_days` is a trap.** The function shifts by rows of its input and fills the first rows with
NaN. Pass `(days, 48)` arrays, never a flat half-hour vector (which would shift by 28 half-hours
and not 28 days), and never replace the NaN rows with 0, because 0 is night import.

**Both medians run on the clear-sky index over 5 half-hours.** The centred form uses the half-hour
and two half-hours on each side, and the trailing form uses the half-hour and the four half-hours
before it. The fit uses the centred form, and the filtered posterior comes from one further forward
pass at the fitted parameters with the trailing form. Three tests on synthetic arrays are required:
the trailing output at a half-hour does not change when a later input changes, the two forms agree
on a day of constant index, and both forms are 0 in the dark.

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
experiment builds none of the six inputs.** The table below is the specification of rung U5, kept
here so that a reader sees exactly what the filtered posterior of this experiment lacks.

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
of section 9 of the amendment, as the erratum corrects them.** No BSP meter is read in any world.

| World | Truth | Role |
|---|---|---|
| The six signed-MW Primaries (D5, D9, D10, D11, D14, and D16), each with an injected farm at levels 1.0 and 2.0 (V1) | The Primary's signed reading minus the farm's output | Sign, lobes, and signed CRPS |
| D10 folded | D10's signed readings | Real export on 58 crossing days, by version 2 |
| The six signed-MW Primaries, un-injected, with D5 forced open | No reverse flow | Invented reverse flow at D5 with `c` = 1; un-injected D10 is the D10 folded world |
| D15 folded | D15's signed readings | Report only: D15 exports at night, so night import does not hold |

**An injected world adds a metered solar farm's recorded output to a Primary's signed reading and
folds the sum** with `Q` of 2 points and the within-half-hour averaging of experiment 720. The six
signed-MW Primaries that host an injected farm are D5, D9, D10, D11, D14, and D16. D5 forced open
is one of the six un-injected signed-MW Primary fits and passes through the model like any other world.

**The scored rows are every half-hour of the fitted span.** The build writes a series for the whole
span, so calibration is read on the whole span, and the four folds of experiment 720's rung 0b are a
reported subset. Night rows dilute the pooled reliability term, so the page also reports the term on
rows with the sun at 10° or more. The bin rule is not diluted. The thirds of the within-third lobe
bar are cut on each world's scored half-hours by the folded reading `m`.

**The bars are listed in the decision rule above and are repeated here, one per row, with the
constant that holds each.**

| Test | Bar | Constants |
|---|---|---|
| Sign | The reliability term of the Brier score of `p_neg` at most 0.01 for the smoothed posterior and at most 0.02 for the filtered posterior, outside the 2-point dead band, with no bin of at least 200 half-hours off the diagonal by more than 0.1 | `SIGN_RELIABILITY_MAX_SMOOTHED`, `SIGN_RELIABILITY_MAX_FILTERED`, `DEAD_BAND_POINTS`, `SIGN_BIN_MIN_HALF_HOURS`, `SIGN_BIN_MAX_OFF_DIAGONAL` |
| Lobes | The 5% to 95% range of the lobe on the true side covers the truth on 85% to 95% of half-hours, and on 80% to 97% within each third of the reading's level (a third under 200 scored half-hours is `void` and neutral) | `LOBE_COVERAGE_MIN`, `LOBE_COVERAGE_MAX`, `LOBE_THIRD_MIN`, `LOBE_THIRD_MAX`, `LOBE_THIRD_MIN_HALF_HOURS` |
| Signed CRPS | Below the hard-sign version's, with a paired interval above 0 | `SIGNED_CRPS_PAIRED_LOWER_MIN` = 0 |
| References | The sign area exceeds the reading-blind and the hour-of-day references, with the lower end of the paired interval above 0 | `REFERENCE_PAIRED_LOWER_MIN` = 0 |
| Night import | No dark half-hour at D12 or D22 has `P(X < 0)` above 0.05, read on the build | `NIGHT_P_NEG_MAX` |
| Invented reverse flow | The share of sunny midday half-hours at un-injected D5 with `p_neg` above 0.5 is under 1%, with `c` = 1; `inconclusive` under 200 such half-hours | `D5_INVENTED_REVERSE_MAX`, `D5_MIN_SUNNY_MIDDAY_HALF_HOURS` |
| Placebo | The sign area under the real sky minus the sign area under the sky shifted by 28 days is at least 0.1, with a paired interval above 0 | `PLACEBO_SKY_SHIFT_DAYS`, `PLACEBO_MIN_AREA_GAIN` |
| D5 without its lowest-reading days | Report only (addendum 1: D5's metering may clip exports) | None |

**The placebo bar restores the size that addendum 1 registered, and the erratum's wording is
weaker.** The erratum asks for a paired interval above 0 and states no size. Addendum 1 registered
the V1 placebo as an area with the real sky at least 0.1 above the shifted sky. No number exists
from this experiment, so restoring the size is not tuning. The amendment's wording ("removes the
gain over the references") would fail a posterior that works, because the shifted-sky posterior still
reads the meter through the fold, night import, and continuity, and the references do not. The page
reports the shifted-sky posterior's gain over the references as the gain that the reading alone
gives. The shifted sky moves the clear-sky index only and keeps the clear-sky yield at its true date,
so that the placebo removes the cloud information and nothing else, because shifting the whole PV
estimate would also move the sun's arc by 28 days and corrupt night import. The first 28 days have
no shifted sky, so the contrast is read only on rows where the shifted sky exists, and the
references use the same shifted sky.

**The pool and the filtered line are fixed.** The sign, lobe, CRPS, and references bars are read on
the 12 injected worlds and D10 folded together, with one bootstrap that resamples whole days across
every world. The filtered posterior's line is `passes` when its reliability term is at most 0.02
with the bin rule and its pooled lobe coverage lies within 85% to 95%, `fails` otherwise, and `not
readable` when the pooled worlds hold fewer than 200 half-hours of true export outside the dead
band. The line gates only later work that reads a filtered or as-of posterior.

**Three limits of the first buildable version are known in advance.** The sign within about 2
points of zero stays undetermined. The split between PV and demand inside a daylight stretch is
driven by the prior. D12's capacity is the weakest input, so the posterior is reported under each
capacity value and under the mixture. A failure of the references bar means that the untangler's
sign is no better than the PV estimate alone, and the page says so.

**The mixture arms test the object that the build stores, and only withhold.** Each world is fitted
once, with the injected capacity known and `Q` at the folding value, while the build mixes 12 refits
over `C` and `Q`. On each of the six level-2.0 injected worlds, job 4b runs 4 refits with `C` at the
stratified quantiles `(i − 0.5) / 4` of a log-uniform prior between 0.7 and 1.3 times the injected
capacity (the bracket that experiment 720's rung 0b registered), and `Q` at the stratified quantiles
of the exponential prior. The 4 refits are mixed as the build mixes 12, and the arm scores the sign
reliability and the pooled lobe coverage. That is 24 fits. If the mixture arm's reliability exceeds
0.01, or its pooled coverage falls outside 85% to 95%, the D12 and D22 series are stored with the
flag `mixture_untested_failed` and are not offered. The arm is a read-out and adds no test. If job
1's projection cannot fit the arm within 5 core-hours in total, the arm is deferred to rung U4 and
the page states that the stored mixture is untested. The page also reports, under `PRIVATE/`, the
ratio of D12's two capacity values beside the 0.7 to 1.3 bracket, so that the reader can judge
whether the bracket covers D12's own spread.

**Four known-answer tests of the addendum are not run here, and each waits for a later scope.**
V2 (injected generator schedules), K1 (D8's own record with the generator subtracted), and V3
(real negative controls at D8) need `K` above 0 and the D8 scope, and they belong to rung U3, which
experiment 720 owns. V4 (read-outs at D12 and D8 that need no truth, reported with no bar) runs at
D12 only, as aggregates (`P(X < 0)` in the dark, and the monthly share of export days), after job 5
has fitted D12 and D22, and only when the verdict reached row 5 or row 6.

**Calibration is reported for both posteriors.** Reliability diagrams use 10 bins of `p_neg`. The
live-fairness checks of version 2 section 8.3 (perturbation, truncation, parameter cutoff, and
poison) test the as-of posterior, so they do not apply here, and the filtered posterior of this
experiment is labelled "not live-fair" instead.

## Compute, launch rules, and gates

**The build needs about 3.4 core-hours on one CPU core and no GPU, and the cap is 5 core-hours.**
Experiment 720 measured 53 s per whole-span pass at 84 states. One pass at 41 states takes about
12.6 s if the cost scales with the square of the state count. Linear scaling gives about 26 s and a
total of 5.0 core-hours without the mixture arms, which is why the cap equals the pessimistic
projection. One fit is 10 passes and one trailing forward pass.

| Job | Fits | Estimate, core-hours | Cap, core-hours |
|---|---|---|---|
| 1: input checks, spans, capacity values, `Q` measurements, switching-log windows, timing of one whole-span pass | 0 | Under 0.1 | 0.1 |
| 2: worlds (12 injected, 6 un-injected, D15 folded) | 19 | 0.75 | 1.5 |
| 3: shifted-sky worlds (12 injected, D10 folded) | 13 | 0.5 | 1.0 |
| 4: scoring of every known-answer bar and the filtered line | 0 | Under 0.1 | 0.1 |
| 4b: mixture arms (4 refits on each of the six level-2.0 worlds) | 24 | About 1.0 | 2.0 |
| 5a: D12 (12 refits, 2 single-capacity fits, the grid of `Q`) | 14 | 0.6 | 1.25 |
| 5b: D22 (the same) | 14 | 0.6 | 1.25 |
| 5c: V4 read-outs at D12, night import, the formula series, verdict rows 5 and 6 | 0 | Under 0.1 | 0.1 |
| 6: report, figures, and private tables | 0 | Under 0.1 | 0.1 |

**The jobs above 1 core-hour each have two Opus code reviews, as version 2 requires.** Version 2
section 9.2 sets a cap of 1 core-hour for a job with one code review, and jobs 2, 4b, 5a, and 5b
exceed that cap, so the second review of the fixed code is not optional.

**The timing gate has three parts, built on experiment 720's `CostGate`.** Job 1 times one
whole-span pass on real input, and releases no job whose projected passes exceed that job's cap or
take the series above 5 core-hours. Each fitting job times its first fit and projects again. Each
job checks elapsed time after every fit, and asserts at the end that the passes run do not exceed
the passes planned. A projection over a cap at job 1 is reported to the manager, who may raise the
cap, and writes no verdict. The verdict row 2 "the timing gate stops a job" applies only to a job
stopped after the job started. Under linear scaling, jobs 2 and 3 project just above their caps, so
the gate would hand the decision to the manager before job 2.

**Every job follows the launch rules of version 2, and the implementation brief restates them.**

- One `systemd-run --user --scope` at a time, on one pinned core, with `MemoryMax=12G`,
  `MemorySwapMax=0`, and `OOMPolicy=continue`.
- One thread for BLAS and Polars, and `CUDA_VISIBLE_DEVICES=""`.
- A check for 25 GB of free memory before launch, made by the launch script.
- A check-in with the ML manager before each job, and a "fits starting" line to the manager before
  each family of fits (the worlds, the shifted-sky worlds, the mixture arms, and the builds at D12
  and D22), with no job started before the manager's reply.

**Four gates close before the first fit.** First, an Opus code review of all new code comes before
any run that produces numbers the page quotes, because the experiment imports unmerged code, builds
a new emission integral, and changes a running median. Second, a second Opus review re-reads the
fixed code before the long runs. The code review checks leakage (the fit never reads a reading on or
after 2026-07-01), the forward and backward passes against experiment 720's, the integral of the
five numbers, that the trailing median looks back only, and the `shift_days` trap. Third, decision 9
is answered before job 5, and the manager approves the cap of 5 core-hours, the whole-span rows,
and the placebo bar before job 1. Fourth, this page matches erratum 1 and the review's must-fix
items.

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
before a number is quoted is fixed and the job reruns as the same attempt, under four conditions.
First, a test on synthetic data, or a check that reads no bar statistic, fails before the fix and
passes after the fix. Second, the fix changes no constant, population, sensor, or bar. Third, an
Opus review confirms the fix. Fourth, both sets of outputs are kept and the page reports both. Each
job reruns at most once without the manager's approval, and `inconclusive` and `not supported` are
never rerun. The thresholds, the scale `c`, and the sensors are never changed for a second attempt.

**The count of earlier attempts is the reason for the single run.** Version 2 lists 18 numbered
experiments that produced or gated a sign label (79, 152, 358, 362, 365, 366, 370, 387, 408, 412,
417, 420, 424, 427, 506, 508, 543, and 674). Three experiments tracked the generator behind D8 (694,
695, and 696), and none gave a usable label at D12 or D8. Experiment 722 was the fourth use of BSP
closure for a sign and the 19th experiment that produced or gated a sign label, and it was
inconclusive.

**A test is a statistic with a bar or pattern stated in advance, of which at least one outcome
admits, and two counters use that definition.** An outcome admits when it produces or opens a gate
to a sign label, admits a hint or factor, sets a prior, or licenses a later design to cite the
statistic as evidence for a sign. A read-out describes or withholds, and a read-out that did not
withhold may not later be cited as evidence for a hint. The first counter counts experiments that
hold at least one test that produces or opens a gate to a sign label, which is the count of
section 8 of the amendment. The second counter counts tests that use BSP closure for a sign (the
"no fifth test" rule), and this build reads no BSP meter, so the second counter does not move.

**Experiment 724 counts as an experiment that produces a sign label,** because it writes `p_neg` for
D12 and reads bars on `p_neg`. The manager has decided (decision 18): through experiment 722 there
were 19 attempts, experiment 723 is the 20th, and experiment 724 is the 21st. The rule that no
third experiment on either of the two open ideas starts without the maintainer's recorded approval
is not affected, because this build tests neither the daylight BSP hint nor the within-day hint.

**The series stops when the references bar fails, and the work does not proceed to the as-of rung.**
A failed references bar means the untangler's sign is no better than the PV estimate alone.

## Decisions for the maintainer or manager

**The amendment lists ten decisions (section 10 of `mva_untangling_design_v3_amendment_v2.md`), and
the table says how each one bears on this build.** Where the manager has ruled, the row is marked
**[Decided by the manager]**. This page takes no other decision, and the marker names who decides.

| No. | Decision (amendment's wording, shortened) | Bears on this build as follows |
|---|---|---|
| 1 | Whether the follow-up experiment of jobs W0, W2, and W3 runs as the 20th experiment | Indirectly: its outcome decides whether a later rung adds a within-day arm. This build neither waits for it nor depends on it |
| 2 | The stop rule and the outcome table of section 8 | Directly: this build keeps the single-run rule and the count above, and takes "daylight BSP factor off" as given |
| 3 | Flaw 1: drop the D15 sign test and its placebo | Not concerned, except that D15 is folded here as report only |
| 4 | Flaw 2: remove every size bar and keep dark-hour closure as a read-out | Not concerned, because this build reads no BSP meter |
| 5 | Flaw 3: the rule for future generator records, and half the gap or the full gap for the detection limit | Not concerned, because `K` = 0. **[Decided by the manager]** The claim needs the full gap, and half the gap is reported as a secondary figure. The erratum's recommendation, that the claim decides (a claim of presence against a point null takes the full gap, and a test with an outcome that states a size takes half the gap), is applied to this rule; no gate changes for a record whose result has been read |
| 6 | Flaw 4 and rung U1: remove the day class, adopt the farm sensor, admit metered farms as injected farms | Concerned only through the injected farms of the V1 worlds, which are farms of experiment 39's frame and are not shown |
| 7 | The form of the daylight hints in rung U4 | Not concerned: the daylight BSP factor stays off and `a` = 0. **[Decided by the manager]** The rule "one factor at most" is accepted, because the latent-sky form needs the meter sensor, which is the paused daylight BSP factor under another name, and the meter sensor holds the child's own flow |
| 8 | The CAMS download (decline for now, or approve 22 requests) | Concerned: this build uses the hourly series, which suffices with `a` = 0, so it proceeds on the "decline" default |
| 9 | Whether the build may follow its known-answer test directly, without a rung U4 that has no hint to test | **[Decided by the manager]** Yes: rung U4 is skipped when it has no hint to test. Version 2 gates the build on a rung U4 that is `supported` or `partly supported`, where `partly supported` means "no hint is kept and the model passes alone", which is the case of an empty list of hints. The build runs when this experiment reaches row 5 or row 6 of the verdict table, and the label `partly supported` in rows 4 and 5 does not meet version 2's gate for rung U6 |
| 10 | The switching-log repair: `engine="openpyxl"` in the experiment's own code, with no change to `pyproject.toml` | Directly: the plan uses the repair to set the `transition` flag |

**Eight further decisions came up while planning.** Each is marked with who decides, or with the
manager's ruling where one exists.

- **11. Delivery form to NGED [Maintainer].** Version 2 section 10 gives three options (no
  delivery, D12 and the five settled Primaries, or all eight MVA-only Primaries), and recommends
  building the first scope privately and deciding after the known-answer scores.
- **12. The verdict mapping [Decided by the manager].** The six ordered rows in the decision
  rule, on the smoothed posterior, from erratum section 3.1, with the planner's four additions (a
  void third, the D5 count floor, the filtered line's bars, and a build status beside the
  known-answer result) that the Opus review requires. The manager accepted the verdict word
  `not built` for a stop inside job 5, with the known-answer bars kept on the page.
- **13. The rows of the known-answer worlds [Manager; open].** The erratum departs from the rows
  that addendum 1 registered for V1 and scores every half-hour of the fitted span, with the four
  folds of experiment 720's rung 0b as a reported subset.
    - *Design lead's recommendation:* take the erratum's whole-span rows, which change no threshold
      set before data were read.
    - *Opus reviewer's view:* the whole-span scoring "is sound, and the span is never defined"; the
      reviewer asks for a start date, a rule for the era structure at D12 and D22, and a rule for
      an injected world whose farm record starts later than the host's, which the page now states
      in `row_set_id`. This agrees with the recommendation.
- **14. Publication of D12's signed posterior on dates [Maintainer].** Version 2 allows it subject
  to a check that no single metered farm dominates D12's PV. Job 1 of experiment 722 found none,
  but the plan keeps every dated series private until the maintainer confirms, and keeps every
  capacity value from experiment 673 under `PRIVATE/`, because a polygon's registered capacity less
  the metered farms can expose a unit size.
- **15. The within-third lobe coverage.** Ruled by the erratum: the bar of 80% to 97% within each
  third binds, and a third under 200 scored half-hours is `void`.
- **16. The compute cap.** The manager accepted the per-job caps of 2.0 core-hours for job 4b and
  0.1 core-hours for job 5c **[Decided by the manager]**. The cap of 5 core-hours for the series
  stays open **[Manager; open]**; the estimate with the mixture arms is about 3.4.
    - *Design lead's recommendation:* the erratum recommends 5 core-hours in place of 3, with a cap
      on each job.
    - *Opus reviewer's view:* "the estimate checks, and the cap equals the pessimistic
      projection"; the reviewer adds that a gate that stops before job 2 should hand the decision
      to the manager and write no verdict. This agrees with the recommendation.
- **17. Contract decisions that belong to a human in the production repository [Maintainer].**
  Production code is not touched here, and the following stay open: which unit label a series
  carries when the observation is in MVA and the series is in signed megawatts, and whether the
  power-factor assumption is stated beside the label; whether the sign convention extends to MVA
  series, and how an inferred sign is marked; how the production metrics score a signed series
  against an unsigned observation, and whether the clamp at zero stays for the eight MVA-only
  Primaries; whether the five-number posterior, a quantile set, or a mean with a flag is the stored
  form; which effective capacity in megawatts normalises D12; and whether a fitted `Q` per Primary
  may replace the assumed small floor. The power-factor alternatives are the open part of this
  decision that the manager also weighs.
    - *Design lead's recommendation on the power-factor alternatives:* report `mu_pos` under `Q` = 0
      and under a power factor of 0.97 as two labelled alternatives with no bar (erratum section
      3.5), which changes no threshold.
    - *Opus reviewer's view:* the review records no position on the power-factor alternatives. It
      checks the arithmetic of the prior and asks only that the page list the two alternatives
      with no bar, so it does not disagree with the recommendation.
- **18. Whether this build counts toward the number of sign experiments [Decided by the
  manager].** Yes. Through experiment 722 there were 19 attempts, experiment 723 is the 20th, and
  experiment 724 is the 21st.
- **19. The placebo bar [Manager; open].** This page restores addendum 1's size of 0.1, which the
  erratum omitted without comment, and shifts only the clear-sky index.
    - *Design lead's recommendation:* the erratum states the bar as a paired interval above 0 with
      no size, and records no reason for dropping addendum 1's size of 0.1. The recommendation that
      changes no threshold set before data were read is to keep the size of 0.1.
    - *Opus reviewer's view:* the erratum's bar "is sound in its reasoning and is weaker than a bar
      already registered"; the reviewer asks that the size of 0.1 from addendum 1 be restored,
      because "leaving the size out without comment would be a silent weakening". This agrees with
      the recommendation and disagrees with the erratum's wording.

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
    - **Fitting:** about 10 passes per fit, and 12 refits per Primary with a month-block
        bootstrap, a stratified draw of `C`, and a stratified draw of `Q`, with `c` fixed at 1. A
        read-out of 24 further fits mixes 4 refits on each of the six level-2.0 injected worlds. The
        only seed is the bootstrap seed, and it is fixed.
    - **Resources:** CPU only, one core, no GPU, about 3.4 core-hours estimated and 5 as the cap.
- **Measurement:**
    - **Rows:** every half-hour of the fitted span, with the four folds of experiment 720's rung 0b
      as a reported subset and no valid time on or after 2026-07-01.
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
  forward-backward pass. Jobs 2, 3, 4b, and 5.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_five.py`: integrates the
  five numbers from the mixed density, the truncated-normal and gamma lobe shapes, and the formula
  for the settled Primaries.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_worlds.py`: builds the
  injected, folded, and forced-open worlds and their placebos. Jobs 2 and 3.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_score.py`: the
  reliability term, the lobe coverage, the signed CRPS, the reference areas, and the paired
  bootstrap, the filtered line, the mixture arms, and the V4 and night-import scoring after job 5.
  Jobs 4, 4b, and 5c.
- `ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/e00724_build.py`: the 12 refits
  at D12 and D22 with the grid of `Q`, and the writer of the private posterior tables. Jobs 5a and
  5b.
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
  `PYTHONPATH`): `e00720_hmm.py`, `e00720_fit.py`, `e00720_worlds.py`, and
  `e00720_rung0b.py`. The page does not link experiment 720's write-up, because the write-up is on
  an unmerged branch.
- `ml-experiments/00716-msa-unmetered-pv-part-1-history-estimate/` and
  `ml-experiments/00717-msa-unmetered-pv-part-2-target-row-forecast/` (read, on `PYTHONPATH`): the PV
  estimate and the as-of capacity table.
- `data/ml-experiments/00724-msa-mva-history-untangler-part-1-first-build/` (written): the private
  posterior tables (smoothed and filtered, five numbers, visibility flag, kind, case of `L`,
  parameter cutoff, capacity flag, code hash), the per-world score tables, and the run logs. Dated
  detail is in the `PRIVATE/` subfolder.
- `data/ml-experiments/briefs/msa_architecture/` (read, private): the design files named in the
  section "Status and place in the series of sign experiments".
