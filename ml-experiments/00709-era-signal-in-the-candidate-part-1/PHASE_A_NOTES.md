# Experiment 709, phase A: notes for the code review

**Nothing in this folder has been run.** The scripts pass `ruff check`, `ruff format`, and `ty check`
only. Every number, shape, and import below was desk-checked, not executed. Run the scripts with
these folders on `PYTHONPATH`: `00086-*`, `00598-*`, `00660-*`, and `00687-*` (all under
`ml-experiments/`), after `e00709_era_table.py`. Order of runs: `e00709_era_table.py`, then
`e00709_stage0.py --part tune`, then `--part all`.

## Files

- `e00709_features.py`: paths, `Panel` (16 Primaries on the 2019-09-01 grid, 2026-06-30 last day),
  ERA5 weather, climatology residual, six daily features, robust scaling.
- `e00709_era_table.py`: `era_table.parquet` and `known_events.parquet`.
- `e00709_bocpd.py`: Bayesian online change-point posterior, alarm rule, episodes, threshold tuning.
- `e00709_hmm.py`: sticky three-state hidden-Markov model, forward filter, class rule.
- `e00709_classify.py`: evidence, rule classifier, soft classifier, arms `all`, `rule`, `soft`,
  `hmm`, and the per-day `SiteSignal`.
- `e00709_inject.py`: era injector, transfer injector, worlds, host rules.
- `e00709_stage0.py`: tuning, known cases, injected bank, example, and perturbation test.

## Existing code reused

- `ml-experiments/00687-*/e00687_features.py`: `block_unit`, `build_grids`, `features_at`, and
  `FEATURE_NAMES`. It imports `e00598_signals` (folder `00598-*`).
- `ml-experiments/00086-*/e00086_labels.py` (`load_labels`) and `e00086_common.py`
  (`series_table`), for the experiment 41 labelled set.
- `ml-experiments/00660-*/e00660_generator.py`: `PowerPanel`, `EVENT_COLUMNS`, `apply_events`,
  `POSITIVE_EDGE`. Rows are written by this module, not by the generator's random builder.
- `ml-experiments/00191-*/e00191_ecr.py`: the three-slot extraction pattern and the name map were
  copied, because experiments may not edit each other and the file runs as a script.
- `src/studies`: `ecr_reader.read_sheet`, `eval_protocol.slices.MVA_ONLY_LABELS`,
  `sources.ML_EXPERIMENTS_DATA_DIR`, `sources.study_dir_for`.

## Ambiguities in the plan and the simplest reading chosen

1. **Gate G4 is ported, not imported.** `e00592_common.py` imports `e00582_arms` and
   `e00583_common`, which need `torch` and stored forecasts of experiments 565 and 584, so it cannot
   load on the CPU environment. `classify.g4_on` ports `gate_fresh` (turn on above the threshold with
   an opposite-sign neighbour, off below half the threshold). The reading is the day's mean residual
   against the climatology in place of experiment 565's lag-free median, and the threshold is
   `max(3 sd, 5% of capacity)` from the first 365 days of features.
2. **The mirror statistic is recomputed at the end of every day** by `features_at` of experiment 687,
   with the all-Primaries median taken over the uninjected panel (as experiment 687 does). The
   running maximum of `mir_12` and `mir_48` since the day before the onset is the evidence. A site
   with no metered neighbour (no readings in the 14 days before the onset) gets mirror evidence 0
   and the flag `no_neighbour` = 1.
3. **Four bands of the day** (UTC): night 22:00 to 06:00, morning 06:00 to 11:00, midday 11:00 to
   16:00, evening 16:00 to 22:00. The plan names the bands without limits.
4. **Climatology exists** when the 730-day window holds at least 330 valid days (24 readings or
   more) and at least 10 valid days in the target month of the year. With these numbers D15 and D18
   get a climatology from 2026-03, as the plan says, but their 365-day scale window never completes,
   so their detectors never start (state `none`) and they are not hosts.
5. **The scale** is the median absolute deviation (times 1.4826) over the first 365 days after the
   first day with all six features, with a floor of 0.2% of capacity. The detectors start on day
   365. Origins in those 365 days have state `none`. Sites with fewer than 200 valid days in the
   window never start.
6. **Change-point model:** Normal-Gamma prior with mean 0, kappa 0.1, alpha 2, beta 1.5, independent
   across the six features. Run length truncated at 730 with the mass beyond kept at 730. A day with
   a missing feature leaves the posterior as it was. The onset estimate is the most probable run
   length in 1 to 27 days. The size test needs one feature that passes both 3 scaled units and 5% of
   capacity (the same feature).
7. **Previous segment** is the mean of up to 180 days before the estimated onset, starting no earlier
   than the previous episode's onset, or its end after a return. A return to the earlier level
   disarms the detector until the posterior mass falls below the threshold once.
8. **Episode end:** the 3-day mean of every feature within 1.5 scaled units of the previous mean for
   3 consecutive days. An episode older than 28 days may be superseded by a later alarm.
9. **Class persistence:** an era stays `era` after its episode, with the change vector frozen 28 days
   after the era class, until a later era or a reversal inside 120 days. A switch or `unknown`
   episode returns to the earlier state when it ends.
10. **Weather evidence is ERA5 only.** It uses days that are at least 7 days old at the origin and
    returns 0 with fewer than 6 such days, so it arrives at day 12 or later after the onset. The
    plan's ensemble-mean weather for the newest days is not used.
11. **Rule weights** are fitted by class-balanced logistic regression on evidence rows at ages 7,
    14, 21, 28, 45, and 60 days of injected tuning-world shifts. The threshold is a log-odds of 2.2
    (probability 0.9) in either direction. The class locks at the first crossing.
12. **Soft classifier:** a numpy network (13 inputs, 16 tanh units), full-batch Adam, whole injection
    passes held out for Platt scaling. No torch is needed.
13. **Duration prior** uses the durations of the 580 verified events without the ones whose onset
    falls in the scored window (side `a` of each transfer). `onset_utc` of the event table is read
    as the transfer's side `a`.
14. **Verified events** carry `end_a_idx` as exclusive (the end index is the first half-hour after
    the event). This was not checked against experiment 580.
15. **Event match:** each arm is matched to the first of its own episodes that starts (the alarm
    day, or for the hidden-Markov arm the first day the mode leaves the base state) in -2 to +90 days
    of the onset. Undetected events have an infinite delay. See the review section below for the
    state at each check day.
16. **Tuning:** the threshold is the larger of the thresholds for real clean history (outside labelled
    events, widened by 3 days before and 14 days after) and for three month-of-year surrogates
    (each day swapped with a random day of the same calendar month, in any year), Primaries other
    than D8, 2021-11 to 2024-12. The grid is 0.5 to 0.999, and the printout says `GATE FAILURE` if
    the 0.999 fallback still exceeds 0.5 false alarms per Primary-year.
17. **Hidden mirror** writes the donor's row only, so the recipient is unmetered. Every host of this
    kind gets an empty neighbour list for the mirror evidence, the weather evidence, and gate G4
    (`process_hosts`), so `no_neighbour` is 1 as for a real Primary with an unmetered partner.
18. **Hosts:** D8, D15, and D18 are never hosts in any world (D8 keeps its real eras apart). A
    switching donor with no metered neighbour can host only the hidden-mirror kind. In the bank,
    every fourth pass replaces its first kind with `very_long` (121 to 180 days). **Each pass holds
    2 era hosts and 2 switching pairs** (see the host-feasibility section below).
19. **Perturbation test:** the panel is cut after day `D` (day `D` included and perturbed), because
    nothing later can reach a signal that reads day `D - 1`. The neighbours' features are the
    untouched real ones; only the Primary and its neighbours are recomputed. The check covers the
    arms `all`, `rule`, `soft`, and `hmm` of the Primary.
20. **L-hmm** is fitted per Primary on its own days before 2025-01-01, D8 included (on D8's own
    pre-2025 data), for each site with enough clean days. The base state is the modal filtered state
    in the training days. The arm's state is `none` before 2025-01-01, and the arm is scored only on
    events whose onset is on or after that date.
21. **Day bands of an `unknown` shift and the duration bands** are `<60`, `60-120`, and `>120` days.
    Day-band limits of the four daily bands are item 3.
22. **Anchor relative to the previous era.** The change vector and anchor of an era are measured
    against the level of the era before it. After a return that lasts more than 120 days the era
    ends: the signal reverts to the previous era's anchor and change vector and sets `withdrawn` on
    that day, so phase B withdraws the anchor from the network input.

## Not verified

- No script has run, so no import, array shape, or data column has been confirmed at run time. The
  riskiest points are `e00687_features.features_at` (called with every day of one site at once) and
  `e00660_generator.apply_events` (called on a `PowerPanel` built from capacity shares).
- `era_table.py` assumes the `Date Connected` column holds Excel serial numbers as text (as
  experiment 191 did) and that the register rows for D8 are found by the substring `Marsh Lane`.
- The experiment 41 events are runs of `in_event` per Primary, so overlapping events merge.
- The D8 telemetry date and size of the 2025 solar connection are estimated by
  `d8_solar_estimate` (clear-day midday-minus-night step in 2025-02..08, with the same search on 2024
  as a control, and the April-August clear-day midday median of 2025 against 2024). A reverse flow
  at the MVA-only Primary folds under `abs`, so the size is a lower bound; the count of folded-looking
  days is printed. Nothing has run, so no estimate exists yet.
- Run time: the bank (64 passes with 8 hosts) recomputes the detector over the whole record for
  every host. A restart from a saved posterior at the onset would be faster.
- `pydoclint` was not run (not installed). Docstrings follow the Args and Returns rules by hand.

## Changes made after the Opus review of phase A

**Applied (all 14 findings; none rejected, because each held when checked against the code).**

1. *Matched-episode scoring.* `score_event` matches each arm to one episode. The state at a check
   day is `none` before the episode starts, the episode's own state while it is open, `era` after it
   closed only if classed `era` and not reversed, and `none` otherwise. An era that was current
   before the onset is in a separate `pre_state` column. The confusion state `none` therefore counts
   undetected events and events whose episode closed without an era class.
2. *Hidden-Markov arm.* State `none` before 2025-01-01. Scored only on onsets on or after that date.
   The perturbation test covers it.
3. *Hidden mirror neighbours.* See item 17 above.
4. *Era withdrawal after a return beyond 120 days.* See item 22 above.
5. *Undetected counts as infinite delay.* The gates use the share detected within 7 days
   (`level_m25`) and 21 days (`level_m10`), at least 0.5, and print `alarmed n/N`.
6. *Surrogates* are month-of-year shuffles, with a printed gate failure on the 0.999 fallback.
7. *Solar-shaped era kind* (`solar_25`) is in the tuning world with its own sigma (0.12; curriculum
   0.15, bank 0.20, retrained 0.25).
8. *Disjoint pairs.* A donor and a recipient pair never share a neighbourhood with another pair, and
   `EXCLUDED_HOSTS` are never recipients.
9. *Transfer share* is drawn from U(0.9, 1.0).
10. *Frozen models* are saved by `tune` to `stage0/models/models.npz` and `models.json`, so
    `--part tune` writes them and `--load-models` reads them.
11. *D8 and the register.* D8's eras come from `era_table.parquet`. The solar telemetry date and size
    are estimated (see above). A penalised least-squares segmentation of D8's midday feature
    (penalty 3 x noise variance x log n, minimum 28 days) is printed against the detector's
    onsets. The era table's other register rows are scored (descriptively).
12. *Verified-set gate.* Per transfer (either side ever era counts), 23 transfers outside the scored
    window carry the gate with Wilson intervals; the 4 inside it are reported apart.
13. *HMM detection* is the first day the mode leaves the base state (the excursion start).
14. *Example* is printed for the episode matched to D8's 2023-08-15 telemetry date, or for a bank
    event with its truth if no episode matches.

Also: D8 gates print PASS or FAIL; G4's false triggers per Primary-year and its median onset delay on
the verified events are printed; `MirrorGrid` is typed, `Final` carries type parameters, the
file-wide `E501` suppression is gone, the Excel date is a polars expression, and the docstrings of
`Models`, `classify_states`, and `choose_hosts` follow the field order and Args/Returns rules.

**Still not run:** no script has run after these edits either. The first run should be
`e00709_stage0.py --part tune`, then `--load-models --part real`, then `--load-models --part injected`
and `--load-models --part cut`.

## Host feasibility and the per-pass counts

The first `--part tune` run failed with `no host assignment satisfies the separation rules`.
`e00709_probe_hosts.py` (2 cores, 8 GB, nothing fitted) draws 5000 random host assignments for each
onset day (every 14th day of the tuning window and the 8 scored month starts) and reports the share
that satisfy the plan's separation rules. The rules are unchanged. Output: `host_feasibility.csv`.

| Onset days | Host pool | Old pass (4 eras + 4 switching) | New pass (2 eras + 2 switching) |
|---|---|---|---|
| 2021-12-31 to 2023-02-24 | 11 | 0.0% | 79.4% to 96.8% |
| 2023-03-10 to 2023-08-11 | 12 | 0.0% | 85.3% to 97.2% |
| 2023-08-25 to 2024-05-31 | 12 | 0.1% to 0.3% | 85.3% to 97.5% |
| Scored months 2025-11 to 2026-06 | 13 | 0.8% to 1.2% | 86.4% to 98.2% |

The new range is over the four rotations of the two switching kinds. The lowest values are the
rotations that contain two plain transfers or a plain transfer with a long kind (each plain transfer
needs a metered recipient).

**Change.** `N_ERA_PER_PASS` is 2, and `SWITCH_ROTATION` gives the two switching kinds of pass `i`
from `i % 4` (plain+hidden, plain+long, hidden+long, plain+plain), so four consecutive passes
cover every kind. Era kinds rotate with the pass index. To keep the number of events: the bank has
16 passes per scored month (128 passes over the 8 months, 512 events, as before from 64 passes of
8 events), and tuning has 80 passes (320 events). A tuning pass draws onset days until the pool
holds at least 8 hosts and a host assignment exists (30 tries), and skips the pass with a printed
line otherwise. The probe shows every tuning onset day already qualifies.

## Leak found by the perturbation test, and its fix

**Cause.** The first stage 0 run failed pair 41 (D12, origin day 2011, arm `soft`, change 0.0024).
The `rule` and `all` arms passed because they read only the first 7 evidence columns, and the leak is
in the columns that only the soft classifier reads (`cu_ratio`, `cu_ret`, `rate`, from experiment
687's grids) and in the running maxima of the mirror columns. The mirror grid divides each site's
block means by a noise unit that is measured on that site's first 365 days of features. Two
neighbours of D12 had an incomplete window at the origin (`ready_day` equal to the number of
days), so their window ran past the origin and their unit, and the all-Primaries median that uses
it, were measured on days at or after the origin. Perturbing those days changed `lvl_*`, `cstep_*`,
`cu_depth`, `cu_age`, `cu_ratio`, `cu_ret`, and `rate` on days before the origin (the diagnostic
printed the differences). The old test could not see the same problem in gate G4, whose threshold
is also measured on the first 365 days, because the test reused the unperturbed thresholds.

**Fix.** `build_mirror_grid` now counts a site's readings only from its `ready_day`, and measures
the unit only for sites whose window is complete (`_known_residual`). `g4_on` takes `ready_days`
and shows no reading above threshold before a site's threshold exists. With both changes,
`--part cut --load-models` passes for 200 pairs and all four arms, largest change 2.2e-16.

**Re-run.** The rule weights and the soft classifier were fitted on evidence rows that included the
leak, and every mirror and G4 evidence column changes for hosts with a neighbour that is not ready.
Re-run `--part tune` (the detector threshold does not change) and then `--part all`; all stage 0
tables that depend on the classifiers (confusion tables, classification delays, the verified-set
gate, D8's classes) are affected. The detection tables (alarm delays) are not.

## Why the detector misses level_m10, solar_25, and D8's 2025 solar farm

The cause is the size test, not the 0.999 threshold and not the 5% floor. The test needs 3 scaled
units in one feature, and a unit is the site's robust spread of that feature, which is 9% to 13% of
capacity at D8 (10.4 MW) and larger than 3% at many other hosts, so 3 units is 30% to 40% of capacity
at D8 and the 5% floor never binds. A 10% step (`level_m10`) clears 3 units only at quiet hosts, and
`solar_25` moves one band only at midday. At D8 the detector (threshold 0.999) finds onsets on
2025-02-20, 2026-01-19, and 2026-05-11 but none near the register date or the measured step
(2025-06-10, 2.75 MW = 26% of capacity = 2 units in the midday feature); lowering the probability
threshold to 0.5 or 0.9 adds no onset in 2025 other than 2025-02-20, so the size test blocks it.
The hidden-Markov arm detects 56/64 `level_m10` events because it has no size test.

## Arm L-stat (stage 0 part)

Design: `data/ml-experiments/briefs/eras_embedding/lstat_design.md`. New files `e00709_stats.py`
(12 window statistics, 28-day change, year-on-year reference, expanding scale) and
`e00709_statdetect.py` (alarm, onset, episodes, era ids, threshold tuning). Edited:
`e00709_classify.py` (`shape_evidence`, arm `stat`, `SiteSignal.era_id`, `Models.stat_threshold`
and `Models.stat_weights`, saved in `models.npz`/`models.json`), `e00709_inject.py` (the `engine`
era kind in its own `engine` world; the bank keeps four kinds), `e00709_stage0.py`,
`e00709_era_table.py`. Nothing in the stage 0 scripts has run; a read-only probe ran
`window_statistics` and `detect_stat` on D8 and D12 only (about 3 s per Primary). With threshold 4
the detector's D8 episodes include onsets on 2023-08-20, 2024-06-27, and 2025-03-21 (alarm
2025-04-26), the dates in the design's section 1; at threshold 5 the 2023-08-15 onset appears and
the 2025 alarm is on 2025-04-30.

**Ambiguities and the reading chosen.**

1. *Daily analogue for the onset split.* The design does not define it. Each statistic's daily
   analogue is the same statistic of one day's 48 readings (level and profile statistics exactly;
   residual statistics around the median profile of the 28-day window that ends that day; the
   weekday-minus-weekend statistic as the day-mean residual on weekdays only, NaN at weekends). The
   split needs 7 values before it and 3 after it.
2. *Contamination mask.* The design says "inside an earlier alarm episode". An episode here
   contaminates the 56 days from its onset (the span in which a 28-day change straddles the step)
   and, if it closed by a return, the 56 days from 3 days before its end. Open episodes (eras) do
   not contaminate forever. The same mask removes days from the scale.
3. *Scale refresh.* The scale (median and 1.4826 MAD of `y`, floor 0.5% of capacity, at least 365
   values for every statistic) is recomputed on every 7th day over days up to `t - 28`, not every
   day. It reads only older days, so the cut holds.
4. *Return closure and previous segment* use the six band features and the constants of
   `e00709_bocpd` (3 days within 1.5 scaled units; previous segment up to 180 days). The statistic
   detector has no size test (3 scaled units and 5% of capacity); its `z` threshold is the only gate.
5. *Era id change vector.* The vector is the window ending 27 days after the onset minus the window
   ending the day before it, over the scale on that day, readable on the later of that day and the
   alarm day. An era is confirmed on the later of its first `era` day and that day. Recurrence
   ignores withdrawn eras (an id of a withdrawn era may still be reused).
6. *Dwell.* `era` needs 14 days of age from the estimated onset; `switch` has no minimum.
7. *Shape evidence.* The post window is the days from the onset to the origin day minus 1, capped at
   28 days; the pre window is the 28 days before the onset, at least 17 valid days. The profile
   misfit is the RMS of the post median profile minus `a * pre profile + b`, over the scale of the
   p50 statistic on the alarm day. The residual-shape change is the largest of the three
   residual-shape changes over their scales. Both are clipped at 8 and are zero before 7 days after
   the onset.
8. *Tuning.* The threshold grid is 3 to 7. The false alarm rate counts alarms outside labelled
   events on 2021-11 to 2024-12 without D8 (no surrogates, because shuffling days would destroy the
   windows). The stat weights use the tuning-world rows at ages 7, 14, 21, 28, 45, and 60 days of
   episodes of the statistics detector, so the tuning pass now also computes window statistics
   (about 4 hosts per pass, 80 passes). The two-column ablation is printed as class-balanced
   training accuracy.
9. *Irregular engine.* -30% of capacity on half the days, for 4 to 12 hours starting at a random
   half-hour, scored in 4 passes per scored month (stage 0 only; `--part engine`, included in
   `all`). It is not in the bank.
10. *D8 solar telemetry.* `e00709_era_table.py` now fits a ramp to the year-on-year change of the
    28-day midday-minus-night level (2025-01-01 to 2025-08-31) and writes the first day and size
    into the era table's solar row, so `e00709_era_table.py` must be re-run before stage 0. The
    probe's fit puts the first day on 2025-04-05, between the register date (2025-03-24) and the
    2025-04-26 alarm. The stage 0 clear-day estimate (2025-06-10) disagrees by about 9 weeks and is
    still printed. The D8 solar gate accepts a detection within 30 days of either the register date
    or the telemetry date and prints which date passed.
11. *Start of the arm.* The detector starts when the scale exists (about 785 days after a Primary's
    first reading, 2021-11-25 at D8). Events with an onset earlier than that are not scored for the
    arm. The arm needs the Primary's raw power (`process_site(power=...)`), so the cut test passes
    the perturbed power row.
12. *`SiteSignal.era_id`* is 0 for every arm other than `stat`, and is part of `at_origin`, so
    the cut test compares it.

## Changes made after the Opus review of arm L-stat

Applied after checking each finding against the code.

- **B1 (supersession).** A 28-day change keeps `|z|` above the threshold for about 55 days after a
  step, so a second alarm could supersede a real step about 30 days after its onset. An open
  episode is now frozen for `CONTAMINATED_DAYS` (56) days from its onset, and on creating an
  episode the alarm counter and the arming flag are reset. Probe on the real panel with a
  synthetic step added to the power row (read-only): at D3, -25% on 2024-03-01 alarms on
  2024-03-04 and stays open until 2024-06-14, where it is superseded by the genuine 2024-04-29
  event that the unperturbed D3 also shows (alarm 2024-06-14); +15% alarms on 2024-03-10 with the
  same ending. D12 with -25% on 2022-03-26 keeps the real episode (onset 2022-03-08, alarm
  2022-03-26) open past the injected step instead of superseding it. D2 was not probed.
- **B2 (tuning mask).** `tune_stat_detector` uses its own mask, 35 days before and 56 days after
  a labelled event, and prints the false-alarm rates with the detector's 3/14-day margins too. The
  threshold is chosen on the new mask.
- **S1 (closure).** An episode closes by a return only if the six band features read back for 3
  days and the alarming statistic has `|z|` below 1.5 for 3 days. The tuning step prints the share
  of statistics episodes closed by a return within 14 days of the alarm. Closure now lags the
  return by up to about 28 days (the window), and a reversal shows in the statistics for up to
  about 90 days. The return step's contamination starts 31 days before the closure.
- **S2 (onset and contamination).** The onset is split on the level statistic (percentiles 5, 50,
  95, night, midday) with the largest `|z|`, and the contamination starts at the earlier of the
  onset and the alarm minus 28 days.
- **S3 (ablation).** Arm `stat_noshape` is the statistics detector with the seven rule columns
  only. Its weights are the 7-column fit padded with zero shape weights (`stat_noshape_weights`,
  saved and loaded). It goes through the same gates as every other arm.
- **S4 (D8 solar).** The solar pass on the table's telemetry date is labelled as resting on a date
  fitted to the same telemetry. A table date more than `D8_RELIABLE_DAYS` from the register date
  is not used. The era table states that the size is a lower bound (D8 reports megavolt-amperes
  only) and prints and stores the same sign (the fall, positive).
- **S5.** `main` raises if `stat_weights` or `stat_noshape_weights` are zero after loading the
  models (run `--part tune`).
- **S6 (gaps).** The window row is computed with at least 17 valid days even if the last day is
  invalid (the daily analogue still needs a valid day). A day without a finite statistic resets
  the alarm counter. A gap of 7 or more days without readings suppresses alarms until both 28-day
  windows are full again.
- **Minor.** `Final[int]` on the percentile and time-of-day constants; the named constant
  `ENGINE_WITHIN_DAYS`; `SiteResult.stat_evidence` is documented; "12" replaces "twelve".

Rejected or left unchanged.

- **S7 (vectorising `window_statistics`).** Left as a loop. A vectorised version would have to
  reproduce the quantile interpolation and the validity rules exactly, and the loop runs once per
  Primary (seconds), so the risk outweighs the saving.
- **Era id 0.** Era id 0 means no era vector (its indicator is zero, so the model input is zero).
  Ids of 1 or more index the shared new-era table of `e00709_model.py`, and an unseen id or a
  curriculum id uses table index 0, as for every arm.
