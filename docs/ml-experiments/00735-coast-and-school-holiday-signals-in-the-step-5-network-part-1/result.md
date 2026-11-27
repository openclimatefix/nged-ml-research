---
id: "00735"
slug: coast-and-school-holiday-signals-in-the-step-5-network-part-1
title: >-
  Does telling the Primaries' step-5 demand network when Lincolnshire's schools are on holiday, and
  how close each Primary is to the coast, lower its forecast error around school holidays and bank
  holidays, especially at the coastal Primaries and on warm, sunny holiday days?
theme: demand-inputs
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  Anything else. (1) The experiment stores about 220 fitted neural networks (6 new arms, 12 monthly
  refits, 3 seeds), each with per-row forecasts, all private. (2) More than one design defensibly
  tests the hypothesis (coast signal as input columns or as an embedding term), so each design is an
  arm with a pre-registered contrast and a placebo twin. (3) The experiment patches the runner and
  network of experiments 677, 591, 574, and 510 through a copy of experiment 731's runner, whose
  callers cannot be named without searching; it touches no serving path or degradation rule.
  Reviews, in order: a Sonnet and an Opus plan review, Sonnet code, an Opus code review before any
  fit, the smoke test, a Sonnet result review that re-derives the headline contrasts, a Sonnet
  numbers review, and a Sonnet prose review.
hypothesis: >-
  The step-5 demand network of experiment 677 (arm 9a, `k1_5w_sp1`, monthly refits) has no input
  that says when schools are on holiday, and nothing that ties a Primary's holiday response to its
  distance from the coast. Adding school-holiday columns to the network's mixer lowers its CRPS in
  the weeks around school holidays at both metered groups. Adding the coast distance, as an
  interaction with the holiday, weekend, and weather columns, lowers the CRPS further at the 4
  Primaries within 3 km of the coast (D2, D4, D9, and D19) in those weeks, and the gain is largest
  on warm, sunny holiday days. A permuted coast distance and a shifted school calendar give no such
  gain.
prediction: >-
  Written before any fit. CRPS, arm minus reference, in points of capacity (a point is 1% of a
  Primary's capacity), days 1 to 10, on the rows of the school windows (7 days before, the holiday
  block, and 7 days after) of the 8 held-in refit months. P1: school columns minus the control (C1)
  is -0.03 to -0.10 at the six signed-MW Primaries and at the eight MVA-only Primaries, and the
  whole-year change is within -0.01 to +0.01. P2: school columns minus the shifted calendar (C2) is
  -0.02 to -0.08 at both groups. P3: the coast-and-weather columns minus their permuted twin (C4)
  is 0.00 to -0.08 at the 4 coastal Primaries, with zero the likeliest value, because the
  stored-forecast diagnostic gave no evidence either way. P4: the gain of C4 on warm, sunny holiday
  days is at least 0.04 points larger than on cold or wet holiday days. P5: the coast term in the
  embedding minus its permuted twin (C5) is within -0.02 to +0.02, because a term that is constant
  per Primary adds nothing a free embedding cannot represent. P6 (October 2026, outside the
  verdict): on 2026-10-24 to 2026-11-01 the school arm raises the signed error of the 12 to 36 hour
  median at D19, D4, and D9 by at least 3 points of capacity (the control is 10 to 19 points under).
  P7 (July to September 2026, outside the verdict): no prediction. The likeliest misses are P3, P4,
  and P6, because 2026-10-24 was a cold day. The likeliest verdict is `inconclusive`, because the
  interval treats fits as the unit and the effects sit near the seed noise.
decision_rule: >-
  The reference of C1 is R0, the stage-2-and-3 replay of arm 9a through this experiment's runner (or
  experiment 677's stored `k1_5w_sp1` forecasts, if the replay gate passes). A contrast is a paired
  difference of CRPS (11-level quantile score, points of capacity) at days 1 to 10 on the
  school-window rows of the 8 held-in refit months (2025-11 to 2026-06, valid times before
  2026-07-01), averaged over seeds 0 to 2. Interval: a two-level bootstrap with 20,000 draws, which
  draws the 8 refit months with replacement, then 3 seeds with replacement inside each drawn month,
  then 7-day blocks of valid dates inside that month's windows, read at 99.17% (95% Bonferroni over
  6 readings: C1 and C2 at the six and at the eight, C4 and C5 at the 4 coastal Primaries). A gain
  needs the interval wholly below zero, a point estimate of at most -0.02 (the protocol's minimum
  gain for a new input and a new data source), a negative seed-mean difference in at least 6 of the
  8 refit months, and a negative difference in at least 18 of the 24 fits. A reading wholly above
  zero is harm. Placebo readings, at 95%: SP minus R0 at both groups, SKWP minus S and SEP minus S
  at the 4 coastal Primaries. A placebo fails, as the protocol states, when its interval lies wholly
  below zero. School claim at a group: C1 and C2 are gains there, SP minus R0 has not failed there,
  and the whole-year contrast of S against R0 (all rows, days 1 to 10, the demand-v2 month-block
  bootstrap) has a one-sided 95% upper bound of at most +0.02. Coast claim: C4 or C5 is a gain at
  the 4 coastal Primaries, its placebo reading has not failed, the whole-year contrast of that arm
  has an upper bound of at most +0.02 in each group, and the same contrast at the 3 inland placebo
  Primaries (D11, D10, and D22) does not move in the same direction beyond its own 95% interval.
  Verdict, first rule that applies: (1) `refuted` when C1 is harm at either group; (2) `supported`
  when the school claim holds at both groups and the coast claim holds; (3) `partly supported` when
  the school claim holds at one or both groups or the coast claim holds, but rule 2 does not apply;
  (4) `small gain` when C1, C4, or C5 has an interval wholly below zero with a point estimate
  between 0 and -0.02; (5) `not supported` when the intervals of C1 at both groups, of C4, and of C5
  lie inside -0.05 to +0.05; (6) `inconclusive` otherwise. The family stays at 6 readings and 99.17%
  even if arms are dropped. Every reading in October 2026 and in July to September 2026 is
  pre-named and decides nothing. Numbers before the final scoring are interim.
benchmark: >-
  R0 (arm 9a replayed from experiment 677's stored stage-1 weights); experiment 677's stored
  `k1_5w_sp1` forecasts as the reference row `stored`; the manual heuristic and climatology from
  experiment 677's joint pass; experiment 704's stored arm B (CAMS columns dropped, the production
  variant) in its five months as a reference row.
row_set_id: backtest_year_2025_11_to_2026_10_daily_12utc
metric: >-
  CRPS (headline) and normalised MAE of the median, in points of capacity, with the signed error of
  the median, q01, q05, q95, and q99 coverage and pinball loss, by holiday window, weather split,
  group, and Primary, at days 1 to 10 (headline) and in every demand-v2 lead band from 0.5 hours to
  14 days.
---

<!-- Draft plan for experiment 735, revised 2026-11-27. The opening, figures, and results sections
are written after the run. -->

**This experiment asks whether the Primaries' step-5 demand network forecasts better around school
holidays if it is told when Lincolnshire's schools are on holiday, and whether it forecasts the
seaside Primaries better if it is also told how close each Primary is to the coast.** The step-5
network is the neural network that forecasts the power drawn at 16 Primary substations in
Lincolnshire up to 14 days ahead, and it is the first production demand network for them. On
Saturday 2026-10-24, the first day of the October half-term, the network under-forecast four
Primaries near the coast by 10 to 19 points of capacity. A colleague suggested a "coastal" flag,
because seaside resorts may react to holidays differently from inland towns. We will retrain the
network with school-holiday columns, with the coast distance added in two different ways, and with
placebo versions of each signal. We then compare the error in the weeks around each school holiday
and bank holiday, at the coastal and inland Primaries, and on warm and cold holiday days.

## Plan

### What is already known

**The step-5 network sees bank holidays and Christmas but not school holidays.** Its 21 date-time
columns hold the sun position (3), two hour-of-day harmonics (4), the day of the week (7), the
lockdown flag `covid`, three Christmas columns, `bank_holiday`, `nearest_days` (signed days to the
nearest England bank holiday, clipped to ±7, divided by 7 and standardised), and `nearest_inside`.
Saturday 2026-10-24 had no bank holiday within 7 days, so the network saw an ordinary October
Saturday.

**Each Primary has a free 8-number embedding, which every encoder and the mixer read**
(`e00559_model.ModularMdn`, `e00591_model.AuxLagMdn`). Stage 1 pre-trains the encoders on ERA5 and
CAMS rows from 2019-09. Stage 2 attaches a fresh mixer and trains it, with the weather and date-time
encoders frozen and the embedding, lag encoder, and per-Primary output offset trainable. Stage 3
fine-tunes the same parameters with recency weights and injected load shifts (`e00574_fit.train`,
`step2`, `fine_tune`).

**A per-Primary constant cannot add information that the free embedding lacks.** The network can
already represent any function of a Primary's coast distance through the embedding. A coast column
can therefore change the forecast only by sharing a learned response across Primaries in proportion
to their closeness to the coast. Sharing matters most where each Primary has few examples. A
placebo with permuted coast distances has the same capacity, so the contrast against it tests the
geography alone.

**The stored forecasts show the October 2026 under-forecast and no evidence either way over the
scored year.** The diagnostic `e00735_diagnostic.py` read experiment 677's stored `k1_5w_sp1`
forecasts at 12 to 36 hours, for seed 1 only (line 119). On 2026-10-24 the largest under-forecasts
were D19 -18.7, D18 -16.3, D4 -15.7, D9 -10.8, D8 -9.8, and D10 -9.7 points. Over 2025-11 to
2026-06 the correlation across the 16 Primaries between coast distance in kilometres (not
closeness) and the excess signed error on holiday days was +0.14 for school-holiday days, -0.10 for
half-term days, and +0.12 for bank holidays. A positive correlation with distance means that
Primaries nearer the coast have the more negative excess error, which weakly favours the coast
hypothesis for two of the three kinds of day. With |r| of at most 0.14 across 16 Primaries and one
seed, the diagnostic is no evidence either way, and it has already looked at the held-in months.

### Maintainer decisions this plan follows

- **Fit option B, about 5 GPU-process-hours.** The maintainer approved about 1 hour above the
  4-hour cap. Anything above 5.5 GPU-process-hours needs a new request.
- **October 2026 and July to September 2026 are scored as two named blocks, by the maintainer's
  decision of 2026-11-27,** which overrides the demand-v2 rule that no experiment scores valid times
  on or after 2026-07-01. The verdict rests on the 8 held-in refit months (2025-11 to 2026-06, valid
  times before 2026-07-01), which were fixed before the idea arose. Both named blocks are pre-named
  contrasts that decide nothing and are never pooled with the held-in result.
- **Scoring July to October 2026 uses up that holdout for any later adoption of these columns.** The
  `release-model` skill reads the holdout once, so a later adoption needs months from 2026-11
  onward. The 3 km coastal cut was chosen after the October errors at D9, D4, and D19 were seen.
  D18 (3.94 km), D8 (6.72 km), and D10 (15.76 km) under-forecast on 2026-10-24 as well, so the slope
  of each Primary's paired difference against its distance is a co-primary description beside the
  3 km cut.
- **Both entry points for the coast signal are tried, each with a permuted twin:** input columns
  (SKW and SKWP) and a term in the Primary embedding (SE and SEP).
- **Holiday behaviour from year to year is checked, and one arm (SKW) can represent the interaction
  of school holiday, coast, and warm, sunny weather.**
- **Sonnet writes the code and an Opus agent reviews it before any fit.** No XGBoost.

### The arms

**Every arm reruns stages 2 and 3 and the prediction pass of arm 9a for each refit month and seed,
from experiment 677's stored stage-1 weights** (`chains/5/<month>/work/pretrained_P2/`, read only).
The new columns enter the stage-2 mixer with first-layer weights initialised to zero, so every arm
starts stage 2 as exactly the control's network. Stage 2 trains the mixer for up to 60 epochs, which
is where a new column can be learned. Stage 3 stores its first epoch in 72% of fits (experiment
731), too short to learn a new column from zero.

| Arm | In words | Columns added to the mixer | Coast in the embedding |
|---|---|---|---|
| R0 | the control: arm 9a replayed through this experiment's runner | none | no |
| S | school holidays | S1 to S3 | no |
| SP | placebo for S: the school calendar moved 21 days later | S1 to S3, shifted | no |
| SKW | school holidays, coast interactions, and coast × school × weather | S1 to S3, K1 to K5, W1 to W4 | no |
| SKWP | placebo for SKW: SKW with the coast closeness permuted between Primaries | as SKW, permuted | no |
| SE | school holidays, with the coast closeness as a term in the embedding | S1 to S3 | yes |
| SEP | placebo for SE: SE with the coast closeness permuted | S1 to S3 | yes, permuted |

**SK (school plus coast interactions without weather) is not run,** because its only reading, SK
minus S, had no placebo twin. SKW minus S stays as a descriptive reading.

### Exact definition of each signal

**The coast closeness `c` falls from 0.87 at the nearest Primary to under 0.05 beyond 15 km.**
`c = exp(-d / 5 km)`, where `d` is the distance in kilometres from the Primary's latitude and
longitude (`primaries.json` of the experiment 705 bundle) to the boundary of
`studies/beam_diffuse_split/domains/great_britain.geojson`, in EPSG:27700, as `e00735_diagnostic.py`
computes it. `c` is static, known at every origin, and not standardised, so products with an
indicator are zero off the indicator. Values: D9 0.87, D4 0.84, D19 0.71, D2 0.58, D15 0.46, D18
0.46, D5 0.35, D8 0.26, D14 0.23, D21 0.18, D3 0.17, D16 0.11, D12 0.08, D10 0.04, D22 0.04, D11
0.003.

**Every new column uses one local-date convention: the local date (Europe/London) of the stamp
minus 15 minutes.** Stamps mark the end of a half-hour. The existing `bank_holiday` and
`day_of_week_local` columns use the stamp itself (`e00032_common`, lines 301 to 310) and so label
the 00:00 stamp as the next day. The new columns recompute the bank-holiday and weekend flags from
the shifted date and never multiply by the existing columns.

**School columns.** A holiday block runs from the day after a term's last day to the day before the
next term's first day, weekends included. S1 `school_off` is 1 on every date of a block. S2
`school_first_weekend` is 1 on the first Saturday and Sunday of a block (inside the block, for a
block that starts on a weekday). S3 `school_half_term` is 1 inside a block of at most 10 days, so
the network can tell a half-term from a long holiday.

**School calendar sources.** Dates for 2021-22 to 2025-26 come from experiment 11's
`lincolnshire_term_dates.csv`, which is immutable and stays as it is. The experiment reads, in
addition, `lincolnshire_term_dates_2026_27.csv` in its own folder, which holds only terms 1 and 2 of
2026-27 because only those are used: term 1 runs 2026-09-03 to 2026-10-23 and term 2 runs 2026-11-02
to 2026-12-18, so the October half-term block is 2026-10-24 (a Saturday) to 2026-11-01. The source
is a [third-party page](https://www.savemyexams.com/learning-hub/school-holidays/england/lincolnshire/lincolnshire-county-council/),
fetched with the maintainer's permission and not checked against the council's own page, as the
csv's `source` column says. It is the only new data source. Dates after 2026-12-18 are unknown, so
the calendar ends on 2026-11-30.

**Lincolnshire's calendar is a proxy.** The 2024-25 rows are marked `source=school`, and academies
set their own dates. Resort demand also follows the school calendars of the visitors' home counties
(the East Midlands and Yorkshire), which the experiment does not model.

**School placebo (SP).** S1 to S3 read the real calendar 21 days earlier, so each block appears 21
days later on its real weekdays. The placebo overlaps the real windows: placebo Christmas
(2026-01-10 to 01-26) overlaps the real "7 days after" window, and placebo Easter (2026-04-24 to
05-11) overlaps the real "after" window and the 2026-05-04 bank holiday. The summer block overlaps
its real dates by about 3 weeks. SP can therefore move the score, which is why SP is read against R0
as well as against S. A test prints the share of placebo holiday days that are real holiday days.

**Coast interaction columns (SKW and SKWP).** K1 `c`; K2 `c × school_off`; K3
`c × school_first_weekend`; K4 `c × bank_holiday`; K5 `c × weekend` (Saturday or Sunday). K1 alone
adds nothing an embedding cannot do; it gives the products their main effect.

**Weather interaction columns (SKW and SKWP).** W1 to W3 are products with `c × school_off`, read at
the row's valid time from the 00Z ECMWF ensemble run issued 9 hours before the 09:00 UTC origin.

- W1 warm for the month: the `t_max_day` B1 column (input index 52) minus its mean over the same
  calendar month of the refit's training rows at that Primary, divided by the standard deviation of
  those anomalies, clipped to ±3. The standardised column is negative all February and positive all
  Easter, so it would encode which block a row is in.
- W2 sunny for the month: the same anomaly of the `sw_mean_24h` column (index 56).
- W3 first warm day: the valid day's `t_max_day` (in kelvin) minus the mean daily maximum 2 m
  temperature over the 7 local days ending 24 hours before the origin (from the expected-weather
  table, which uses the freshest run issued before each half-hour), divided by 5 K and clipped to
  ±3. Every input of W3 was issued before the origin.
- W4 `c × W1` without the school factor, so that a warm-day coastal effect outside school holidays
  does not load onto W1.

**Coast in the embedding (SE and SEP).** The embedding of Primary i becomes `E[i] + c_i × u`, where
`u` is a learned 8-number vector initialised to zero, applied wherever the embedding is read. Because
`E[i]` stays free, SE can represent exactly the functions the control can and differs from S only in
how updates are shared across Primaries. In SEP the term uses the permuted `c`.

**Placebo permutation.** The 16 coast values are permuted with `numpy.random.default_rng(735)` until
no Primary keeps its value and the correlation between permuted and true `c` lies within ±0.3. The
permutation is saved and shared by every month, seed, and permuted arm.

### Holiday behaviour changes from year to year

**The new columns are learned only from rows after 2024-04-01, all of them after COVID.** Stages 2
and 3 train on ENS rows within 730 days before each refit, and the ENS archive starts on 2024-04-01.
The `covid` column and the 2019 pre-training act only on the frozen encoders, which the new columns
never enter.

**Each refit sees each kind of holiday block once or twice, and often at few leads.** Each refit
holds out one validation week per cycle and purges 15 days before and 9 days after
(`e00422_prep.purge_masks`). According to the plan review, the October 2025 and February 2025
half-terms are entirely purged, the May 2025 half-term is held out, and Easter 2026 is mostly purged,
so the 2026-10 refit sees the 2025 half-term only at leads of 6 to 14 days. The `columns` phase
tabulates the training rows that each block contributes to each refit by lead band, and the write-up
reports the table.

**A per-year check, from the cleaned power and without any fit.** `e00735_regime.py` computes, for
each Primary, school-holiday block, and bank holiday from 2021-22 to October 2026, the mean demand
on the block's days minus the mean on the same weekdays and half-hours of the two term-time weeks
on either side, in points of capacity, and plots it against coast distance, one panel per academic
year. Term dates before 2021-09 are not on disk. The write-up reports whether the coastal holiday
response changes sign or size between years. One scored year cannot tell a lasting holiday effect
from that year's behaviour, and the held-in months hold 4 full school-holiday blocks plus a 2-day
stub, so the write-up states this limit beside the verdict.

### Scored period, windows, and splits

**The scored rows are those of `backtest_year_2025_11_to_2026_10_daily_12utc` in 12 refit months.**
Origins are daily at 12:00 UTC, with 672 half-hours of lead and 16 Primaries each, against cleaned
actuals with the `substation_zero` half-hours dropped (counts per fold reported) and D15 and D18 cut
before 2025-03-01. The held-in block (refits 2025-11 to 2026-06) carries the verdict. The July to
September block (refits 2026-07 to 2026-09) and the October block (refit 2026-10, valid times to the
end of the stored power) are labelled "tuning-adjacent" and decide nothing.

**Holiday blocks in the scored rows.** Held-in: the end of the October 2025 half-term (2025-11-01
and 11-02, a stub with no "before" window, counted out of every block-sign statement); Christmas
2025-12-20 to 2026-01-05; February half-term 2026-02-14 to 02-22; Easter 2026-04-03 to 04-20; May
half-term 2026-05-23 to 05-31. July to September: the summer holiday 2026-07-23 to 09-02. October:
half-term 2026-10-24 to 11-01, its "after" window cut where the power stops. Bank holidays held-in:
2025-12-25, 12-26, 2026-01-01, 04-03, 04-06, 05-04, and 05-25; July to September: 2026-08-31; none
in October 2026.

**Pre-registered windows, by local date of the valid time.** For each school block: days 7 to 4
before; days 3 to 1 before; the first weekend; the rest of the block; the first 3 days after term
resumes; and days 4 to 7 after. The union (7 days before through 7 days after) is the "school
window", which carries the headline. For each bank holiday: the day before, the day itself, the day
after, and the weekend it extends, if any. Ordinary days outside every window are the comparison
slice. Christmas Day and Boxing Day count in both the Christmas block and the bank-holiday slice.

**Weather split of holiday days, on realised weather.** A Primary-day in a window is "warm and
sunny" when its ERA5 daily maximum 2 m temperature and daily mean shortwave both reach the same
calendar month's 2019 to 2025 median at that Primary, "cold or wet" when the daily maximum is below
that median or the ERA5 daily precipitation is at least 1 mm, and "mixed" otherwise. Realised
weather makes the split the same for every arm. The first CPU phase prints how many Primary-days of
each kind each window holds, before any fit, and the write-up reports those counts beside every
split result. If ERA5 does not yet cover November 2026, the ENS analysis proxy of experiment 704
stands in for the October block, and the page says so.

**Groups.** The six signed-MW Primaries are D5, D9, D10, D11, D14, and D16. The eight MVA-only
Primaries (every number labelled "reactive power assumed zero") are D2, D3, D4, D8, D12, D19, D21,
and D22. D15 and D18 are reported only. The 4 coastal Primaries within 3 km are D9 (signed-MW) and
D2, D4, and D19 (MVA-only), each reported by group as well as pooled. The inland placebo Primaries
are D11, D10, and D22. The slope of each Primary's paired difference against its distance in
kilometres, with and without D15 and with a bootstrap over Primaries, is a co-primary description,
and the same slope for the placebo arm is its placebo.

### Planned contrasts, intervals, and placebo readings

**The family has 6 readings at 99.17%, each on the school-window rows at days 1 to 10 of the
held-in months:** C1 (S minus R0) and C2 (S minus SP), each at the six and at the eight; C4 (SKW
minus SKWP) and C5 (SE minus SEP), each at the 4 coastal Primaries, with the 3 inland Primaries as
the placebo group.

**Placebo readings, at 95%, gate the claims through the decision rule:** SP minus R0 at both
groups, and SKWP minus S and SEP minus S at the 4 coastal Primaries. Each fails when it gains.

**Descriptive readings, at 95%, decide nothing.** Whole-year no-harm contrasts of every arm against
R0 per group; SKW minus S; SKW minus SKWP on warm, sunny days against cold or wet days (P4); each
contrast on the bank-holiday slice and in each finer window; each contrast in October 2026 (at D19,
D4, D9, and all 16 Primaries) and in July to September 2026; per-Primary and leave-one-out readings;
the slope against distance; N1 (R0 minus `stored`); a table of every contrast by refit month and
seed (24 held-in fits).

**What the intervals can show.** All rows of one refit month come from the same 3 fitted networks,
so an arm difference carries a random effect for each fit. Experiment 704 reported that the
control's pooled CRPS differs by up to 0.09 points between seeds, more than the P1 band. The
single-seed `_resample_bounds` of `studies.bootstrap` and a bootstrap of blocks inside fixed fits
would treat blocks that share a fit as independent and give intervals that are too narrow. The
two-level bootstrap resamples refit months and seeds first. The sign rules (6 of 8 months, 18 of 24
fits) add a check that does not use the bootstrap, and even 6 of 8 months has probability 0.145
under no effect. The experiment can show a direction and a size, and the likeliest verdict is
`inconclusive`. Per-block results are tabled, never given intervals.

**Reported for every arm and window:** CRPS, normalised MAE of the median, signed error of the
median, q01, q05, q95, and q99 coverage and pinball loss, the avoidable-curtailment cost, and cold
winter nights, per group and Primary and by lead band; the whole-year CRPS and MAE in every
demand-v2 lead band and slice; climatology and the manual heuristic on the same rows.

### Control and noise floor

**R0 is the fair control, because it shares the runner, code state, and run date of the new arms.**
Experiment 684 found the chain from stage 2 onwards deterministic, and experiment 731's stage-3
replay matched the stored forecasts to 9e-7 of capacity. Experiment 731 replayed stage 3 only
(`e00731_k1.py` asserts `command == "fit"`), so a replay of stage 2 is untested. R0 therefore runs
first in one gate month (2026-10). If every quantile of R0 is within 5e-4 of capacity (experiment
704's replay tolerance) of experiment 677's stored `k1_5w_sp1` in that month, the stored forecasts
are R0 in the other 11 months and N1 is reported. Otherwise R0 runs in all 12 months (+0.73
GPU-process-hours). The gate compares rows common to both, and reports how many rows differ:
experiment 731 dropped 24,060 D15 and D18 rows because the cleaned power table was rewritten.

**The network is arm 9a with the two CAMS columns, not the production variant without them.**
Stored stage-1 weights exist for all 12 months only for arm 9a. D15 alone drove the arm difference
in experiment 704, so the production port of a winner is re-checked on experiment 704's arm B weights
(open question 1). The four load-shift scenarios are not required, because no arm changes the lags,
the injected events, or the masks.

### Tests to add before any fit

- Every attribute taken from an imported module exists (`hasattr`).
- The school columns equal the two csv files on every date from 2024-04-01 to 2026-11-30 (every block
  start, end, and first weekend); S3 is 1 only in blocks of at most 10 days.
- Every new column uses the stamp-minus-15-minutes local date, at the 00:00 stamp and across the
  clock changes of 2025-10-26, 2026-03-29, and 2026-10-25.
- The placebo calendar is the real calendar 21 days later on the same weekdays; the share of placebo
  holiday days that are real holiday days is printed by block and window.
- The coast permutation is a derangement, fixed by the seed.
- Every window mask against hand-written dates for two blocks, at midnight and at a 00:30 stamp after
  a clock change.
- W3 is causal (experiment 437's `check` pattern), and W1 and W2 equal the monthly anomaly of the B1
  columns.
- Zero start: with the new columns' weights at zero, every arm's network at the start of stage 2
  gives R0's outputs on 1,000 rows, with labels attached and detached.
- The new columns reach stage-3 rows; the arm's files are written under experiment 735's folder and
  the hashes of 677's files are unchanged.

### Smoke test

One refit month (2026-01), seed 0, every arm: stages 2 and 3 for one epoch each, a prediction pass,
and a scorer read on the stage-3 validation week only, writing to `.claude/scratch/e735_smoke/`. The
smoke test checks that each new column is non-zero where it should be (S1 on holiday rows, K2 at the
coastal Primaries on holiday rows, W1 to W3 on SKW's rows), prints one example row per arm, and
measures the time per arm-month for the GPU-hours check.

### GPU-process-hours

**One arm-month of stages 2, 3, and the prediction pass, for 3 seeds, takes about 4.0 minutes on the
GPU.** Experiment 704's logs show stage 2 at 1.5 to 1.8 minutes, stage 3 at 1.3 to 1.5, and the
prediction pass at about 1.0. Experiment 731's `fit` log shows about 10 seconds of training per seed
inside a process of about 1.45 minutes, so most of the process time is CPU row building. Pre-training
is not rerun.

| Item | Arm-months | Hours |
|---|---|---|
| Smoke test (7 arms, 3 processes) | n/a | 0.35 |
| R0 in the gate month | 1 | 0.07 |
| S, SP, SKW, SKWP, SE, SEP, each in 12 refits | 72 | 4.80 |
| **Expected total** | 73 | **5.22** |
| If the replay gate fails: R0 in the other 11 months | +11 | +0.73 (total 5.95) |

**The line of 5.5 hours holds if the measured time stays at or below 4.2 minutes per arm-month,**
because 73 times 4.2 minutes plus 0.35 hours is 5.5 hours. The smoke test measures the time, and the
check runs before the `fit` phase. Building W3 on 2 cores may push the time up. If the harness can
build each month's rows once for all arms, the saving is recorded and not assumed.

**If the measured time or a failed gate takes the total above 5.5 hours, drop SEP first,** in every
month (saving 0.80 hours, so a failed gate totals 5.15), then SE (another 0.80), leaving SE minus S
as a descriptive reading. Never drop SP or SKWP while the arm they test runs. The phase runner stops
at 5.5 GPU-process-hours and asks.

### Memory for scoring

**The scorer's peak should be about 4 GB.** One arm and seed over 12 refits is about 3.9 million rows
of 11 float32 quantiles plus keys (about 300 MB). The scorer streams one arm and seed at a time
against R0 and builds the bootstrap loss frame from the window rows at days 1 to 10 only (about 1.2
million rows). The string `time` key that `losses_for_bootstrap` builds adds about 120 MB per arm
and seed. The scorer runs under `MemoryMax=12G` and records its measured peak.

### Figures planned

- Signed error of the 12 to 36 hour median on holiday days against coast distance, for R0, S, and
  SKW, one panel per block type.
- Observed demand and the R0, S, and SKW medians with 90% bands around each half-term, 14 days
  before to 7 days after, for D19, D4, D9, D11, D10, and D22 (substations may be named; no metered
  generator appears).
- C1, C2, C4, and C5 with their intervals, by finer window and by weather split with the counts,
  beside the placebo readings; each paired difference by refit month and seed; the slope against
  distance for SKW and SKWP; the per-year regime check.
- A day-ahead forecast and the full 14-day horizon of one forecast, for D19 around 2026-10-24.
- The new columns against date for the 4 coastal and 2 inland Primaries; training and validation
  loss against epoch (refit 2026-10, seed 0, every arm); per-group tail results by window.

### Checks the write-up will carry

Median minus mean forecast per Primary; the MAE of the median beside every CRPS contrast; the replay
gate and N1; the zero-start test; the placebo overlap share; the holiday-day counts by weather kind;
dropped half-hours per fold; which audit triggers fired; leave-one-out contrasts (including D15); the
tails; the table of training rows per block and refit.

### Standing practices

- **Followed:** monthly refits; cleaned data with dropped half-hours counted; the D15 and D18 cut;
  every fitted network and loss history saved; stored predictions reused; tails, avoidable-curtailment
  cost, and cold winter nights reported; example forecasts; loss plots; a test that every input with
  a valid time is known at the origin; streamed scoring with a stated peak; a "fits starting" line,
  time log, and GPU-hours stop; review markers; a draft pull request; anonymisation.
- **Departed from:** the four-month screen (a holiday signal acts on a few blocks a year); the
  demand-v2 interval for the window contrasts (the two-level bootstrap); the 2026-07-01 holdout, by
  the maintainer's decision.
- **Not applicable:** load-shift plots, the synthetic world gallery (the smoke test still checks the
  new columns are non-zero), and the calendar-memorisation control (no day-of-year column is added).

### Advice from earlier experiments

**Followed:** experiment 11's placebo calendar shifted by a fixed number of days (SP) and its call
for more half-terms (still few here); experiment 6's scoring of a calendar column on special-day
windows; experiment 704's several seeds, leave-one-Primary-out, and 5e-4 replay tolerance;
experiment 731's "a zero-size placebo is not an independent control"; experiment 677's monthly
refits. **Departed from:** experiment 325's six seed sets (budget).

### Review gates, phase by phase

1. `columns` (CPU): build and test the columns, the per-year check, the weather counts, and the table
   of training rows; the manager views the column gallery.
2. Opus code review of every script, triage, a re-check of each edited script; marker `smoke`.
3. `smoke` (GPU, about 0.35 hours); marker `gate`.
4. `gate`: R0 in 2026-10 against `stored`; marker `fit`, with the arm list fixed from the measured
   times.
5. `fit` (GPU), stopping at 5.5 GPU-process-hours; marker `score`.
6. `score` (CPU), then a Sonnet result review that re-derives C1, C2, C4, and C5 from the raw fits,
   the audit triggers, the write-up, a Sonnet numbers review, and a Sonnet prose review.

### Traps found while planning

- **`step2` silently reuses experiment 677's weights.** The runner sets `e.W = CHAIN`, and `step2`
  skips any seed whose file exists, so pointing `CHAIN677` at 677's chain would skip stage 2 and
  could write into 677's folder. Point the chain at the arm's own folder and read only
  `pretrained_P2` from 677.
- **`AuxLagMdn.forward` detects the injection labels by `x.shape[1] == label_col + 2`,** so appended
  columns break the detection unless `label_col` moves past them.
- **`fresh_mixer` re-draws every weight from seed 12345 for the layer's width,** so a wider mixer
  would change the old columns' weights too. Draw the control's width, then pad zero columns.
- **The embedding term `u` breaks the missing-key assertion in `e00572_model.load_lag_pre_into_ft`
  (line 75),** which allows only `mixer_ft.*` and two knot keys to be missing.
- **The `fit` command installs `arm.build_month_syn` only when `MINE.synthetic` is set**
  (`e00677_k1.py`, lines 167 to 171), and `build_month_syn` calls the `_build_month_p5` captured when
  `e00591_arm` is imported. A wrapper on `e.k.build_month` is dropped in stage 3. Wrap
  `arm.build_month_syn` (or patch `arm._build_month_p5`) and test that the new columns reach
  stage-3 rows.
- **`losses_for_bootstrap` refuses a frame with a `month` column** (`RESERVED_LOSS_COLUMNS`), so the
  scorer relabels to 7-day blocks of valid dates after the call.
- **`e00011_term_dates.LAST_DATE` is 2026-07-31,** and `calendar_frame` filters to it, so October
  2026 would silently read as term time. `validate` also requires six terms per academic year, so
  the two-term 2026-27 file fails it. The experiment's own calendar module reads both csv files,
  ends on 2026-11-30, and runs its own checks (order, no overlap, weekday starts, gap sizes).
- **Stage 3 is in effect one epoch** (experiment 731), so the columns enter at stage 2.
- **The stored power may stop before the October fold's longest leads** (about 2026-10-26), and the
  scorer reports rows per lead band.
- **The clocks change on Sunday 2026-10-25, inside the October half-term.**
- **D15 and D18 share one location, and D2 and D4 share one weather cell,** so their coast values
  and weather columns are identical pairs.

## Method

### Architecture

- The step-5 network is arm 9a (`k1_5w_sp1`) of [experiment
  677](../00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md): a
  weather encoder and a date-time encoder pre-trained on ERA5 and CAMS rows, a shared lag encoder, an
  8-number embedding and an output offset per Primary, and a mixer that outputs a three-component
  mixture with 11 quantile levels.
- Each new arm adds 3 to 12 zero-initialised columns to the mixer's input, or a term `c × u` in the
  embedding (SE and SEP), as defined above. The encoders take no new column.
- Weather inputs come from the 00Z ECMWF ensemble run issued 9 hours before the 09:00 UTC origin, and
  lags follow the 24-hour rule, so no input is later than the origin (W3 has a causality test).

### Training regime

- Stages 2 and 3 train on ENS rows within 730 days before each of 12 monthly refits (2025-11 to
  2026-10), seeds 0 to 2, on cleaned power with D15 and D18 cut before 2025-03-01, from experiment
  677's frozen stage-1 weights.
- Stage 2 trains the mixer for up to 60 epochs without recency weights; stage 3 fine-tunes with
  recency weights and injected load shifts, with experiment 677's loss, optimiser, and validation
  weeks. Placebo arms differ only in the shifted calendar or permuted coast closeness. The run takes
  about 5.2 GPU-process-hours.

### Measurement

- The scored rows are 12 refit months of daily 12:00 UTC origins at days 1 to 10 and every demand-v2
  lead band, against cleaned actuals. The 8 held-in months carry the verdict.
- The metrics are CRPS, normalised MAE of the median, signed error, and the tails, per group, Primary,
  window, and weather split. The baselines are R0, `stored`, the manual heuristic, and climatology.
- Intervals, sign rules, placebo readings, and the verdict follow the decision rule above.

## Files

- `e00735_diagnostic.py` (already run): reads experiment 677's stored forecasts; computes the coast
  distance and the holiday-day error diagnostic. `lincolnshire_term_dates_2026_27.csv`: terms 1 and
  2 of 2026-27, read in addition to experiment 11's csv.
- `e00735_columns.py`: the coast closeness, the school calendar (both csv files), the placebo
  calendar, the coast permutation, and W1 to W4. `e00735_model.py`: a subclass of
  `e00591_model.AuxLagMdn` with the extra mixer columns, the embedding term, and a zero-padding
  `load_state_dict`. `e00735_k1.py`: a patched copy of experiment 731's `e00731_k1.py` with marked
  `E00735 PATCH` blocks.
- `e00735_run.sh` and `e00735_phases.py`: the driver and phase runner (`columns`, `smoke`, `gate`,
  `fit`, `score`), with review markers, one GPU process, `MemoryMax=12G`, the time log, and the
  5.5-hour stop. `e00735_windows.py`: window, weather-split, and group masks. `e00735_score.py`: the
  scorer. `e00735_regime.py`, `e00735_figures.py`, and `e00735_tests_known_answer.py`.
- Read: experiment 677's `chains/5/<month>/work/pretrained_P2/` weights and stored forecasts;
  experiment 704's arm B forecasts; the experiment 705 bundle's `primaries.json`; experiment 437's
  B1 table; the cleaned power table. Written: `data/ml-experiments/00735-*/` (fits, loss histories,
  predictions, the permutation, logs).
