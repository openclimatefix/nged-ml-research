---
id: "00726"
slug: msa-candidate-follow-share-on-injected-shift-bank
title: >-
  Does the axial-attention network carry a shift in a Primary's recent power into its forecast
  more than the Candidate does, on the same injected shifts?
theme: lagged-power-and-error-correction
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  Not simple. (1) It stores one forecast file per world, fold month, pass, and seed, and fits no
  neural network; (2) more than one design defensibly tests the hypothesis (a host-only or an
  all-shifted Candidate run, and two slices of origins), and the plan settles both; (3) it copies
  the injection code of experiment 667 and the prediction loop of experiment 713, whose callers
  cannot be listed without searching the repository; (4) it touches no serving path or degradation
  rule; (5) it changes no contract. Full set of reviews: plan, code (Opus, each new script, twice),
  results, write-up, prose.
hypothesis: >-
  The axial-attention network `M1` of experiment 713 reads 14 days of half-hourly history through
  attention, and the Candidate (experiment 667, arm `P7+D50`) reads a day-old lag, weekly lags, and
  neighbour summaries. On experiment 581's bank of injected shifts, `M1` moves its median forecast
  towards a shift in a Primary's recent power by a larger share of the shift than the Candidate does,
  both at origins in the 7 days after a shift ends and at origins 7 to 28 days after the onset of a
  shift that is still in force.
prediction: >-
  Written before any run. T1 (follow share of `M1` minus follow share of the Candidate, in the 7
  days after a shift ends) lies between +0.03 and +0.12, and L1 (the same difference 7 to 28 days
  after the onset) lies between +0.05 and +0.15, in both groups of Primaries. `M1` minus `K` (`M1`
  without attention) lies inside +-0.05 in both slices. The Candidate's follow share in the
  all-shifted world is higher than in the host-only world. The prediction fails if T1 or L1 falls
  outside its range in either group.
decision_rule: >-
  Two counted contrasts, T1 and L1, each at the six signed-MW Primaries and at the eight MVA-only
  Primaries, so four counted cells, each read at 98.75% (the Bonferroni level for four intervals)
  from a bootstrap that resamples the 8 fold months and then the 24 passes within each fold month.
  The smallest difference that counts is 0.05 of the injected depth, which is experiment 581's own
  target for following after the end and the tolerance of experiment 713's check KA12. A cell reads
  "cannot tell", "`M1` follows more", "`M1` follows less", "small difference", or "no detectable
  difference" by the table in the section "Decision rule and verdict", and the verdict follows the
  table below it. The 0.50 sign rule (a cell that would otherwise read "`M1` follows more", "`M1`
  follows less", or "small difference" reads "cannot tell" when its point estimate changes sign
  without the 0.50 shifts) never applies to a "no detectable difference" cell. Every verdict states
  that experiment 713's known-answer gate was waived.
benchmark: >-
  The Candidate's stored weights (experiment 667 for seeds 0 and 1, and experiment 713's
  `candidate_seeds/` for seeds 2 to 4), run on the same injected panels as `M1`.
row_set_id: backtest_2025_11_to_2026_06_daily_12utc_before_holdout
metric: >-
  The follow share of the median forecast: the mean over scored rows of the median forecast on the
  shifted panel minus the median forecast of the same weights on the unshifted panel, times the sign
  of the shift, divided by the depth of the shift. Leads from 24 h up to but not including 120 h
  after the 09:00 UTC native origin, at origins from day 0 to day 20 of each fold month.
---

**We ask whether the axial-attention demand network of experiment 713 moves its forecast towards a
shift in a Primary substation's recent power by a larger share of the shift than the Candidate
does.** Both neural networks forecast a Primary's demand from its recent power and from weather. A
real load transfer shifts a Primary's recent power for days or weeks. A forecast that carries the
shift forward is right while the transfer lasts and wrong after the transfer ends. We add a constant
shift to the recent power that each network reads, run the stored weights of each network on the
shifted power, and measure the share of the shift by which each median forecast moves. This page is
the plan: no code has been written, nothing has been run, and every number below is a design value
or a figure from an earlier experiment.

**The experiment fits nothing.** It runs stored weights on shifted inputs and compares each forecast
with the same weights' stored forecast on unshifted inputs. The compute is prediction only, about 3.5
to 4.6 GPU-process-hours with a reporting ceiling of 5, and the run starts straight after F4's
counted fits (experiment 715), on the manager's word.

## Why this runs

**Experiment 713 reported a follow share for `M1` alone, and no follow share for the Candidate.**
[Experiment 713](https://github.com/openclimatefix/nged-ml-research/pull/402) planned the comparison
and could not run it, for two reasons.

- **The Candidate's stored injected forecasts hold other events.** The files under experiment 667's
  `P7_D50_<month>/work/inject/` were written with [experiment
  650](../00650-extended-shift-departure-head-against-decayed-recent-error/result.md)'s event
  generator: random depths from 0.03 to 0.50 of capacity, random signs, free onsets, ramps, and
  mirrored receivers. Experiment 581's event definition has no meaning on those files.
- **The two neural networks were scored on different origins.** The 30-day shifts of [experiment
  581](../00581-robust-multiweek-lag-encoder-x3/result.md)'s bank start 16 days before the fold
  month. Experiment 713's `shifts` command forecast `M1` at origins before the month start wherever
  the ensemble table held a row. The Candidate's test rows hold origins inside the fold month only.

**The CRPS contrasts of experiment 713 on [experiment
709](../00709-era-signal-in-the-candidate-part-1/result.md)'s bank leave the question open.** On that
bank `M1` minus the Candidate is +1.28 [+1.08, +1.49] points of capacity from the onset of a
temporary switch to 7 days after the switch ends, and -3.23 [-3.74, -2.72] at origins 7 to 28 days
after a lasting step. The audit of experiment 713 attributes the lasting-step gain to any network
that reads its last 14 days, because `K` minus the Candidate is -2.75 and `E` (`M1` with a
convolutional stem) minus the Candidate is -3.25. A CRPS contrast mixes how far a forecast moves with
how wide its bands are. The follow share measures the movement alone, on a bank whose depths and
lengths are balanced across Primaries.

**The result changes what the write-ups of the MSA line may say and ports no change.** Following
more is harm in the after-switch slice and help in the lasting-step slice, so neither direction is a
gain, and `production_status` is `not-for-production` under every verdict. That is a stated
departure from the `ml-experiment` skill, which flags a `small gain` experiment as `candidate`: this
experiment is a diagnostic of how the forecast behaves, not a change to the forecast, and a
"`M1` follows slightly more" verdict carries no claim of a better forecast.

**Honest expectation: `M1` follows more than the Candidate in both slices, by 0.03 to 0.15 of the
depth.** Experiment 713 stores a follow share of 0.181 for `M1` on a nearby set of origins. The arms
of experiment 581 that were trained with synthetic shifts, as the Candidate is, followed 0.038 to
0.104.

## What is compared

**Terms.** `M1` is experiment 713's selected arm. `K` is `M1` without attention, `E` is `M1` with a
convolutional stem, and `N` is `M1` without history rows. `C` is the Candidate, the demand network of
experiment 667 (arm `P7+D50`). The six signed-MW Primaries are D5, D9, D10, D11, D14, and D16. The
eight MVA-only Primaries are D2, D3, D4, D8, D12, D19, D21, and D22. D15 and D18 are forecast and
reported apart. The native origin of a forecast is 09:00 UTC on the origin day. The newest power
reading that either neural network reads is stamped 24 hours before the native origin. CRPS is the
continuous ranked probability score, in points of capacity.

| Network | Stored weights | Seeds | Fold months | Role |
|---|---|---|---|---|
| `C` | Experiment 667 (seeds 0 and 1) and experiment 713's `candidate_seeds/` (seeds 2 to 4) | 0 to 4 | 8 | Reference of both counted contrasts, in the host-only world |
| `M1` | Experiment 713, stage 3 | 0 to 4 | 8 | Treatment of both counted contrasts |
| `K` | Experiment 713, stage 3 | 0, 1 | The 4 bank folds | Report-only: `M1` without attention |
| `E` | Experiment 713, stage 3 | 0, 1 | The 4 bank folds | Report-only: `M1` with the stem |
| `N` | Experiment 713, stage 3 | 0 | 2 | Known-answer check: a network that reads no history |

**The bank is experiment 581's.** `e00581_bank.contaminate(clean, month_start_idx, r)` adds a constant
shift to the power of every Primary over a window that ends 14 days after the fold month starts.
Primary `p` in pass `r` gets combination `(p + r) % 12` of a depth of 0.15, 0.30, or 0.50 of capacity
and a length of 3, 7, 15, or 30 days. Passes 0 to 11 raise the power and passes 12 to 23 lower it, so
across the 24 passes every Primary meets every depth and length with both signs. The shift enters the
inputs only, and the targets are never shifted. Experiment 713 ran passes 0 to 11 only, so this
experiment runs all 24 passes and reports every reading by sign.

**The follow share of a network at an event is the mean, over the scored rows, of the median forecast
on the shifted panel minus the median forecast of the same weights on the unshifted panel, times the
sign of the shift, divided by the depth.** This is `follow_q50` of `e00581_tails._aggregate`. A
network that ignores recent power scores 0, and a network that carries the whole shift forward
scores 1. The unshifted forecast is the network's ordinary stored forecast, so no second prediction is
needed. Quantiles are averaged over seeds before the difference is taken, as in experiment 581.

**An event is one Primary in one pass of one fold month, and a slice selects the origins of an
event.** With `d` the origin's day of the month, counted from 0:

| Slice | Origins | Reading of a follow share |
|---|---|---|
| After-switch (experiment 581's `after0-7`), counted as T1 | The native origin lies 0 to 7 days after the end: `d` from 14 to 20 | 0 is ideal from `d` = 16; at `d` = 14 and 15 no input shows the end |
| Lasting-step, counted as L1 | The native origin lies 7 to 28 days after the onset and before the end: 15-day shifts at `d` from 6 to 13, and 30-day shifts at `d` from 0 to 11 | The share of a shift of 7 to 28 days' standing that the forecast carries |
| Day 4 to end (experiment 581's `d4-end`), report-only | 4 days or more after the onset and before the end | Comparison with experiment 581's page and with experiment 713's stored 0.181 |

**At origin days 14 and 15 no input shows that the shift has ended, so T1 mixes two behaviours.** The
newest visible power reading at `d` = 14 lies inside the shift, and the newest usable history row of
`M1` is 2 days old. T1 therefore mixes how far a network followed during the shift with how fast it
lets go after the end. The follow share by origin day (report-only reading 4) separates the two.

**The lasting-step slice stands in for a lasting step.** Experiment 581's bank holds no shift that
lasts beyond day 14, and both neural networks were fine-tuned with experiment 581's synthetic shifts,
so both were trained not to follow a shift of the bank's depths and lengths. L1 therefore measures
how far each neural network learned to ignore a shift of 7 to 28 days' standing. Report-only reading
10 adds the follow share on experiment 709's true level steps.

**The lead band is experiment 581's: at least 24 and under 120 hours from `power_fcst_init_time`.**
The bands of 120 to 240 hours and 24 to 240 hours are report-only. A fold month holds, in the
after-switch slice, 144 events at the six signed-MW Primaries and 192 at the eight MVA-only
Primaries, and half of each count in the lasting-step slice.

## What is run

**Every job is prediction only, with stored weights.** No fit, refit, or fine-tune runs, and no stored
state was trained on a row of the bank.

**The counted Candidate run shifts the host only.** The Candidate's neighbour summaries read the
neighbours' power, and experiment 581's bank shifts all 16 Primaries in one direction, which feeds the
Candidate a regional move that no real switch makes. A real transfer moves the donor the opposite
way. The all-shifted world therefore inflates the Candidate's follow share and biases T1 and L1
towards zero. The host-only world removes the bias:

- The 16 Primaries are split into sets in which no member is an electrical neighbour of another
  member. The neighbour lists are those of `c650.world()`, which `e00400_blend.nged_neighbours`
  builds from the interconnection table, so the lists are symmetric. A greedy colouring by degree
  gives 4 sets, and the triangle D10, D12, D15 means that no colouring needs fewer than 3. The
  implementer confirms the count on the CPU in check KP.
- For each fold month, seed, pass, and set, the new module builds the panel in which only the set's
  members carry their shifts of that pass, rebuilds the Candidate's lag columns and signal block from
  that panel, and forecasts the test rows of the set's members with the saved weights.
- The host-only forecast of a pass is the union of the sets' files, with one forecast of every
  Primary, and that Primary the only shifted reader of its own neighbourhood.

**The Candidate in the all-shifted world is report-only.** The module also runs the panel of
`e00581_bank.contaminate` unchanged, at seeds 0 and 1, on all 8 fold months and all 24 passes. That
panel compares with experiment 709, whose bank shifted every input of the Candidate in the same way.
Its prediction is that the Candidate follows more in the all-shifted world than in the host-only
world.

**The Candidate's test rows are cut to the rows that the scorer reads.** The module keeps the origins
with `d` from 0 to 20 and lead times up to 240 hours, and in the host-only world keeps the rows of
the set's members. The cut roughly halves the disk (an uncut file is about 12.9 MB, so 960 uncut files
would be about 12.4 GB) and the Candidate's time per file. The check worlds forecast every test row.

**`M1`, `K`, and `E` run in the all-shifted world only.** For each fold month, seed, and pass, the new
MSA command standardises the all-shifted panel with the fold's stored mean and standard deviation,
swaps the panel into the source as the history power (as experiment 713's `cmd_bank` does), and
forecasts all 16 Primaries. `M1` reads no neighbour, so its forecast of a Primary is the same in both
worlds, and check KM4 shows it.

**Experiment 713 clips the standardised history power at 5 standard deviations, and a shift of 0.50
of capacity reaches the clip.** In the coldest fold month the standard deviation of a Primary's
cleaned power is 0.07 to 0.21 of capacity, so a shift of 0.50 is 2.4 to 7.0 standard deviations. The
clip caps the follow share of `M1` mechanically, and the Candidate's lag columns are not clipped in
that way, so the clip works against T1 and L1. A shift of 0.50 also leaves the physical range at some
Primaries. Three measures answer both effects:

- The MSA command prints, for every job, the share of shifted history half-hours beyond 5 standard
  deviations, by depth and sign.
- The scorer prints the share of shifted half-hours above capacity or below zero, by group and sign.
- T1 and L1 on the shifts of depth 0.15 and 0.30 alone are a pre-registered sensitivity reading,
  which the 0.50 sign rule of the decision rule uses.

**The scorer joins four frames on `time_series_id`, `power_fcst_init_time`, and `valid_time`:** the
Candidate on the shifted panel, the Candidate's stored forecast, the MSA arm on the shifted panel, and
the MSA arm's stored forecast. The MSA command reads its origins from the Candidate's stored forecast
on the unshifted panel, so the MSA jobs do not wait for the Candidate's jobs. A key that any frame
lacks is dropped for every network, and the scorer prints the dropped count per fold month, pass, and
slice and asserts five properties: every network of a contrast holds the same key set; no origin
lies before the month start; every valid time lies before the holdout months; every forecast
quantile is finite; and no in-month origin with `d` from 0 to 20 is dropped.

## One module for the Candidate, which also serves F4's reading S6

**F4's reading S6 and this experiment both need the Candidate's forecast on an injected panel, so the
module is written once and reviewed once.** F4 (experiment 715) forecasts MSA arms on experiment
709's bank and on the second event generator of [experiment
714](../00714-msa-era-identifiability-audit-and-frozen-probe/result.md), and reads each arm minus the
Candidate on experiment 709's bank only. This experiment owns S6's "minus `C`" on the second event
generator, and F4's code stays untouched, because an edit to F4's scorer would join two review cycles
and two merges. The module writes the Candidate's forecasts for the second generator's world on the 4
bank folds at seeds 0 and 1. The scorer joins those files with F4's stored MSA forecasts on F4's
keys, and computes the reading with F4's `event_crps` and `pass_bootstrap_ci`.

**The scorer imports F4's `event_crps`, `pass_bootstrap_ci`, and the second generator read-only from
one pinned commit, `ac3e768e4`.** That commit is the head of `origin/msa-f4-neighbour-rows` at which
F4's gate ran, and the last commit that changed `e00715_era.py`, which holds `event_crps`,
`pass_bootstrap_ci`, and `g2_panel`. The local, unpushed head `3bce9f1c7` is 3 commits ahead and adds
only the waive-gate file and `result.md`. The folder of the second generator
(`e00714_generator2`) is identical at `ac3e768e4` and on `main`. Before the S6 reading, the
implementer re-checks that F4's stored second-generator MSA forecasts carry F4's gate hash and that
`e00715_era.py` has not changed since `ac3e768e4`, and re-pins if either check fails. The log of every
job records the pinned commit.

**The new files, all under `ml-experiments/00726-msa-candidate-follow-share-on-injected-shift-bank/`:**

| File | Does | Source |
|---|---|---|
| `e00726_candidate_panel.py` | Runs one injected panel through the Candidate's saved weights and writes one forecast file per world, fold month, pass, and seed. `--world` selects the panel builder and `--root` the weights' root | A copy of `run_inject` and its helpers from experiment 667's `prevailing/p667_inject.py`, with the source lines and the commit named in the docstring. The changes are the panel builder, the paths, the removal of every table-building call, and the cut of the test rows |
| `e00726_msa_panel.py` | Runs the all-shifted, `zero`, and host-only panels through the stored stage 3 state of `M1`, `K`, `E`, or `N` | A copy of the loop of `cmd_bank` in experiment 713's `e00713_era.py`, importing `load_model`, `batch_for`, and `standardised_quantiles` from the frozen checkout |
| `e00726_score.py` | The key join, the slices, the follow share, the contrasts, the bootstrap, the verdict table, and the report-only readings | Imports `e00581_bank.event`, reimplements the phase rule of `e00581_tails.event_table`, and imports F4's two functions from the pinned commit |
| `e00726_checks.py` | The preflight and the known-answer checks | New |
| `e00726_figures.py`, `run.sh` | Figures, and a copy of F4's wrapper with a folder snapshot added | F4's `run.sh` |

**Each world has two panels, one for the lag columns (`power_lag`) and one for the signal block and
the neighbour readings (`power_sig`).** In the `e581_host` world (with `--set`) and the `e581_all`
world both are the same panel. The `zero` world is the unshifted panel. The `e650` world is the two
panels of experiment 667's `contaminate_year`, the `bank709` world is the split of
`e00709_cand.py:576-579`, and the `g2` world is the split of the second generator, imported from the
pinned F4 commit.

**Three traps in the copied code are a gate of the first code review.**

- `power_for` caches the panel keyed on the pass `r` alone. In the host-only world the key must be
  `(r, set)`, or every set after the first reuses the first set's panel without raising.
- `FoldSignals` asserts `n_test == len(test.x)` against the stored table. The module cuts the test
  rows at the subset passed to `test_block`, never at `test`.
- `table_file` reads `tables/predict/` only when `E00667_PHASE=predict`. The module must assert
  `E00667_PHASE=predict`, because without it the module reads the training-phase tables.

**A literal copy of `run_inject` would write into experiment 667's folder and skip any file that
exists there, so the module has path rules of its own.** `p667_inject.py` hard-codes experiment 667's
data folder and skips a target that exists. Experiment 667's folder already holds passes 0 to 11 at
seeds 0 and 1, written with experiment 650's events. A copy that kept those paths would print "done"
for those passes without forecasting, and the scorer would read experiment 650's forecasts as
experiment 581's. The rules:

- **Every output path lies under this experiment's data folder and carries the world in the name,**
  with `_k<set>` after the world in the host-only world. The module refuses to start if an output path
  resolves under the data folder of experiment 581, 650, 667, 709, 713, 714, or 715.
- **Tables, models, step-2 weights, the chain, and the statistics are read-only.** The module asserts
  that each file exists and never builds one, and holds no call to `ensure_tables`.
- **`--root` redirects every read path together:** `--root e667` for experiment 667's folder (seeds 0
  and 1) and `--root seeds713` for experiment 713's `candidate_seeds/` (seeds 2 to 4). The module does
  not read `CHAIN667` from the environment, and refuses a seed that does not belong to the chosen root.
- **`run.sh` takes a folder snapshot before and after every job,** of every path, size, and
  modification time under the folders of experiments 667 and 713 that the module must not change,
  plus the hashes of the stored fits. A job whose two snapshots differ fails, and the implementer
  stops and reports to the manager.

**The MSA command runs from experiment 713's frozen checkout (`worktrees/frozen-713-7c29114a8`).**
The caller sets `EXTRA_PYTHONPATH`, and `run.sh` holds no absolute path. Before loading a state, the
command asserts that `e00713_arms.code_hash()` equals `e85b84e2ea04b3cb` (the hash in all 1,296
`code_hash` entries of experiment 713's logs), that the checkout's HEAD is `7c29114a8`, and that the
checkout is clean. The log of every job also records the hash of each recipe module as `importlib`
resolves it.

## Decision rule and verdict

**The two counted contrasts are T1 and L1, each at the six signed-MW Primaries and at the eight
MVA-only Primaries.** T1 is the follow share of `M1` minus the host-only follow share of `C` in the
after-switch slice, and L1 is the same difference in the lasting-step slice, both at seeds 0 to 4.
Each contrast is the mean over events of the paired difference, with every fold month weighted
equally. The interval is a percentile bootstrap that resamples the 8 fold months and then the 24
passes within each fold month, with 20,000 draws, at 98.75%. A positive T1 means that `M1` carries a
finished shift forward more than the Candidate, and a positive L1 means that `M1` carries a standing
shift forward more than the Candidate. A percentile bootstrap with 8 clusters tends to give intervals
slightly too narrow, by about a factor of the square root of 7/8 in standard error. The write-up
states the factor and applies no correction.

**The smallest difference that counts is 0.05 of the depth, and the stored follow shares of `M1` show
that the interval may be wider than that.** The 16 stored score files of `M1` give a mean follow
share of 0.181 over 8 fold months and 2 seeds, a standard deviation of 0.066 across the 8 fold-month
means (a standard error of 0.023), and a root mean square difference of 0.038 between seed 0 and seed
1 within a fold month. These files cover passes 0 to 11 and origins from day 4 to the end, including
some before the month start, so they guide scale only. A paired contrast removes part of the
month-to-month spread, so the expected half-width of a counted interval lies between the seed noise
of about 0.007 and about 0.07.

**The reading "no detectable difference" may be unreachable, and then `not supported` is too.** That
reading needs an interval inside +-0.05, and the expected half-width runs up to 0.07. A result with no
difference would then read `inconclusive`.

**The split-seed guard stays.** Before any contrast, the scorer prints, for `C` and for `M1`
separately, the difference between the follow share from seeds 0 and 1 and the follow share from
seeds 2 to 4. A counted cell whose own split-seed difference exceeds 0.025 in absolute value reads
"cannot tell". The guard confounds seed noise with the weights' root, because seeds 0 and 1 of the
Candidate come from experiment 667's chain and seeds 2 to 4 from experiment 713's `candidate_seeds/`
chain. The Candidate's failed seed stays in, as in experiment 713 (seed 0 at the eight MVA-only
Primaries in the first fold month), and the scorer prints every counted cell with that fold month
left out as a report-only reading. The guard can fire by chance in roughly 1 cell in 20.

A cell is one counted contrast in one group, and a cell's reading is the first row that holds. D is
the point estimate and [L, U] the 98.75% interval. A single-seed D pairs seed `s` of `M1` with seed
`s` of `C`.

| Order | Cell reading | Condition |
|---|---|---|
| 1 | cannot tell | The split-seed rule fires, or the cell would otherwise read row 2, 3, or 4 and D changes sign when the 0.50 shifts are left out |
| 2 | `M1` follows more | L > 0, D >= +0.05, and D is positive for at least 4 of the 5 single seeds |
| 3 | `M1` follows less | U < 0, D <= -0.05, and D is negative for at least 4 of the 5 single seeds |
| 4 | small difference | The interval excludes zero, and the sign holds for at least 4 of the 5 single seeds |
| 5 | no detectable difference | L > -0.05 and U < +0.05 |
| 6 | cannot tell | Any other case |

**The sign rule of row 1 applies only to cells that would otherwise read rows 2 to 4.** A "no
detectable difference" cell has a point estimate near zero, and noise flips its sign whether or not
the 0.50 shifts are present, so the rule would turn nearly every such cell into "cannot tell".

| Order | Verdict | Condition |
|---|---|---|
| 1 | `refuted` | "`M1` follows less" in both groups of T1 or in both groups of L1 |
| 2 | `supported` | "`M1` follows more" in all four cells |
| 3 | `partly supported` | "`M1` follows more" in at least one cell, and "`M1` follows less" in no cell |
| 4 | `small gain`, written as "`M1` follows slightly more" | "small difference" with a positive D in all four cells |
| 5 | `not supported` | "no detectable difference" in all four cells |
| 6 | `inconclusive` | Any other case |

**The `refuted` row comes first on purpose,** because the hypothesis names both slices: the row fires
on "`M1` follows less" in both groups of one contrast even when the other contrast reads "`M1`
follows more".

**What each outcome would mean.**

- **`supported`, `partly supported`, or "`M1` follows slightly more":** the MSA network is more
  exposed than the Candidate to a switch that has ended, and tracks a standing shift further. The size
  of T1 times a typical depth gives the bias in points of capacity that a real switch leaves in the
  week after the switch ends.
- **`not supported`:** the two networks carry the same share of a shift. The CRPS difference of +1.28
  on experiment 709's bank then comes from the width of the bands, from the mirrored neighbours that
  the Candidate reads, or from differences between the two banks.
- **`refuted`:** the Candidate follows more. The audit's explanation of the lasting-step gain would
  need another cause, and audit trigger 2 of the `ml-experiment` skill fires.

**Every reading carries experiment 713's disclosure.** Experiment 713's known-answer check KA3 failed
and its gate was waived, so no reading of the MSA line shows that the network can learn to follow a
lasting step. The follow share here is measured on fitted states and needs no such learning, and the
write-up states both facts.

## Report-only readings, read at 95%

**None of the readings below changes the verdict, and the write-up gives the count of report-only
readings beside any of them that it quotes.** The one pre-registered link to the verdict is the 0.50
sign rule, which applies only to cells that would otherwise read rows 2 to 4.

1. Absolute follow shares of `C`, `M1`, `K`, and `E` in each slice and group, with the day-4-to-end
   slice beside experiment 581's published table and experiment 713's stored 0.181.
2. Attribution: `M1` minus `K`, `E` minus `M1`, `K` minus `C`, and `E` minus `C`, at seeds 0 and 1 on
   the 4 bank folds. Prediction: `M1` minus `K` lies inside +-0.05.
3. Every counted cell by sign, depth, and length, with the two diagnostics of the clip and of the
   physical range.
4. The follow share by origin day, from `d` = 0 to 20, as a curve per network and length.
5. By fold month, by Primary, for D15 with D18, and with the first fold month left out.
6. The lead bands of 120 to 240 hours and 24 to 240 hours.
7. The follow ratio at q05 and q95 and the share of unshifted readings on the safe side of q95 or q05,
   as experiment 581's page reports.
8. CRPS against the unshifted power in both slices, `M1` minus `C`, and in the lasting-step slice
   also against the shifted power for target half-hours before the end.
9. The Candidate in the all-shifted world: T1 and L1 with the Candidate's all-shifted forecasts at
   seeds 0 and 1. Its difference from the host-only follow share measures how much of the follow
   share comes from shifted neighbours.
10. Experiment 709's true level-step eras, on the CPU from stored files: the follow share of `M1` minus
    `C` and of `K` minus `C` at origins 7 to 28 days after the onset, on the 4 bank folds at seeds 0
    and 1. The reading needs no GPU, because every forecast it uses is stored.
11. F4's reading S6 on the second event generator of experiment 714: each arm minus `C` after
    temporary moves, on F4's keys, with the Candidate's second-generator files of this experiment.

## Known-answer checks

**Every check below passes before any follow share is printed.** A failed check stops the run, and the
implementer reports to the manager. No threshold is changed after a check is run. Checks KM2, KM3, and
KC6 print pass or fail only, and their values go to a file that nobody reads before phase 1 starts.
The tolerance of every forecast comparison is 1e-4 of capacity, which is 15 times below the resolution
that the follow share needs.

| Id | Check | Passes when |
|---|---|---|
| KP, preflight on the CPU | Every input file of every job, every output path, the colouring of the neighbour graph, and the three checkout assertions | Every input exists; no output path resolves under another experiment's folder; no set holds two neighbours and every Primary is in one set; the code hash, HEAD, and clean state match |
| KC1, zero shift through the Candidate | World `zero`, every test row, against the stored forecast, on three fold and seed combinations that cover both roots | Identical key sets, and a largest difference of at most 1e-4 |
| KC2, experiment 650's events through the Candidate | World `e650`, every test row, two fold months, passes 0 and 7, seeds 0 and 1, written by this check's own run | Identical key sets and a largest difference of at most 1e-4 against experiment 667's stored files, which shows the copy is experiment 667's inject path |
| KC3, experiment 709's bank through the Candidate | World `bank709`, two fold months, passes 0 and 5, seeds 0 and 1 | At most 1e-4 against experiment 709's stored forecasts on the shared keys |
| KC4, the panel | The all-shifted and host-only panels of every pass and fold month | Equal to the cleaned power outside each shift's window and to the cleaned power plus the signed depth inside it; NaN where the cleaned power is NaN; in a host-only panel, every Primary outside the set equals the cleaned power |
| KC5, the Candidate reads the shifted panel | The day-old lag column at an origin inside a window | Moves by one constant per Primary times the shift of the lag index, to a float32-relative tolerance of 1e-5 (absolute when the constant is below 1), in every in-window row and in at least one row; the lag-time columns are identical. The design asked for 1e-6; the lag column is stored in float32 at a magnitude of a few units, so the ratio of the movement to the shift carries a relative error of a few times 1e-7 per row |
| KC6, no cross-talk in the Candidate | One Primary shifted, every test row; and one host-only set with the test rows cut to its members | Every Primary that does not read the shifted one as an in-neighbour is unchanged to 1e-4; the shifted Primary moves; the cut and uncut member forecasts agree |
| KM1, zero shift through the MSA command | World `zero`, two fold months, seed 0, arms `M1`, `K`, `E`, and `N` | Identical key sets and a largest difference of at most 1e-4 |
| KM3, the MSA command reads the shifted panel (runs before KM2) | `M1` on pass 0 of the first fold month | The history power inside a window equals the standardised shifted panel clipped at 5 standard deviations to 1e-6, and the follow share of the 0.50 shifts is above 0.02 |
| KM2, a network that ignores history | Arm `N`, and `M1` with every history row masked, on passes 0 and 12 | The follow share is 0 to within 1e-6 in every slice |
| KM4, `M1` reads no neighbour | `M1` on one host-only panel against `M1` on the all-shifted panel of the same pass | Equal for every member of the set to 1e-4 |
| KS1, the scorer | Synthetic frames in which the shifted median equals the stored median plus 0.37 times the signed shift | Returns 0.37 in every slice for a rising and for a falling pass, and 0 when the frames are equal |
| KS2, the slices | Every event of passes 0 and 13 of one fold month | The origin days of each slice equal the slice table |
| KS3, equal keys | Every contrast | The five assertions hold, and the dropped counts are printed |
| KW, weather at the origin | The gate statement of experiment 713 printed for both commands | The MSA target row reads the ensemble 00Z run of the origin day, issued at least 9 hours before the native origin; the Candidate's origin-time weather assertions pass; no history weather is stamped after the origin |

**KM3 runs before KM2, because KM2 cannot fail if the MSA command never swaps the panel.** Arm `N` and
the masked `M1` score 0 whether the shifted panel reaches the network or not. **KC2 must write its own
files:** with experiment 667's paths, the copied `_done` would make KC2 pass by skipping.

**Departures of the code from the design, stated before any check runs.**

- KC5 uses the float32-relative tolerance in the table above, not 1e-6.
- KM2 compares each arm's all-shifted forecast with its own unshifted forecast from the same
  command: the `zero` world of arm `N`, and of `M1` with `--mask-history`, at seed 0 on the first fold
  month and one more. A second row compares the masked `M1`'s zero world with experiment 713's
  stored forecast at 1e-4.
- Experiment 715's `g2` forecasts exist for fold 2025-11 and seed 0 only, so reading S6 covers one
  fold and one seed.
- Reading 8 uses the project's quantile score (twice the pinball loss over the 11 levels, as in
  F4's scorer), in points of capacity. Reading 7's safe-side half is implemented, and its coverage
  half is not yet. Reading 3 has no breakdown by length, and the clip diagnostics are not yet
  joined to it.
- Phase 5 (the attention extraction) is cut. It is the first cut in the order of cuts below, and
  no code for it exists. The write-up therefore has no attention maps from this experiment. The
  figures script is still to be written before the write-up.

## Leakage and disclosures

**No forecast reads power stamped later than 24 hours before the native origin, and no weight was
trained on the bank.** The target-row weather of the MSA arms comes from the origin's own ensemble
run, and check KW prints the statement, because experiment 404's scores above 24 hours were voided for
a weather look-ahead.

**The write-up states nine limits.**

- The MSA arms read ERA5 and CAMS history weather, a lab shortcut of experiment 713 at history ages of
  2 to 5 days, which does not touch the power path that the bank shifts.
- Both neural networks were fine-tuned with experiment 581's synthetic shifts, so L1 measures how far
  each learned to ignore a shift of 7 to 28 days' standing.
- The counted contrasts shift one host with unshifted neighbours, and no world includes the opposite
  move of a donor.
- Every shift ends 14 days after a month starts, and the lasting-step slice stands in for a lasting
  step.
- At origin days 14 and 15 no input shows that the shift has ended, so T1 mixes following with letting
  go.
- Experiment 713 clips the history power at 5 standard deviations, and a shift of 0.50 reaches the
  clip and can leave the physical range.
- Shifts are constant in megawatts across the day, which matches the working assumption for a load
  transfer and ignores the shape of the transferred load.
- A percentile bootstrap with 8 clusters gives intervals slightly too narrow.
- Experiment 713's KA3 disclosure.

## Compute, order, and machine rules

**The estimate is about 3.5 to 4.6 GPU-process-hours, with a reporting ceiling of 5.** It is built from
stored file times: experiment 667's inject job wrote one file every 1.2 seconds after about a
minute of loading, and experiment 713's bank job wrote one MSA pass every 1.6 seconds after about a
minute of loading. A pass of this design is 3,360 MSA forecasts, so an MSA job of 24 passes takes
about 2.3 minutes (unverified).

| Phase | Jobs | GPU-process-hours |
|---|---|---|
| 0. Preflight KP on the CPU; checks KC1 to KC6, KM1 to KM4, and KW; timing and memory peak of one job of each command | About 25 | 0.4 |
| 1. `C` on the 8 fold months and 24 passes: the host-only world at seeds 0 to 4, and the all-shifted world at seeds 0 and 1 in the same processes | 40 | 0.9 to 1.3 |
| 2. `M1` on the all-shifted world: 8 fold months, seeds 0 to 4 | 40 | 1.3 to 1.7 |
| 3. `K` and `E` on the all-shifted world: the 4 bank folds, seeds 0 and 1 | 16 | 0.5 to 0.7 |
| 4. `C` on the second generator's world for S6: the 4 bank folds, seeds 0 and 1, 32 passes | 8 | 0.2 to 0.3 |
| 5. Attention extraction of `M1` on shifted and unshifted history, for the figures | 6 | 0.1 to 0.2 |
| **Total** | **About 135** | **About 3.5 to 4.6** |

**The plan is fixed before phase 0: 5 seeds of `C` and `M1`, 8 fold months, and 24 passes, with `K`
and `E` on the 4 bank folds at seeds 0 and 1.** Phase 0 prints pass or fail and the time projection
and prints no follow share, so no later choice can follow a treatment score. After phase 0 the
implementer multiplies the measured job times by the job counts and reports the projection to the
manager. If the projection is above 5, the implementer stops and asks before phase 1.

**Order of cuts, if the projection is above the ceiling:** drop phase 5; then the all-shifted world of
the Candidate; then restrict the lead days to the 24 to 120 hour band; then the two-seed plan, which
keeps seeds 0 and 1 of `C` and `M1`, all 8 fold months, and all 24 passes. Under the two-seed plan both
single seeds carry the sign of D, and no split-seed guard applies. No fold month and no pass is cut
from a counted reading, because the 8 fold months are the resampled unit and the 24 passes balance
depth, length, and sign. The manager approves any cut before phase 1.

**The host-only fallback fires only after the first two cuts.** The colouring gives 4 sets, so a count
trigger cannot fire. The remaining trigger is a projection of phase 1 above 1.5 GPU-process-hours
after phase 5 and the all-shifted Candidate run have been cut. The fallback would swap the counted
world for the biased all-shifted world to save about half an hour, so it comes last. Then the counted
Candidate run is the all-shifted world at seeds 0 to 4, the host-only world runs at seeds 0 and 1 on
all 24 passes as a guard, a cell reads "`M1` follows more" only if the host-only contrast at seeds 0
and 1 has the same sign, and the write-up names the fallback in its first paragraph.

**No seed, fold month, or pass is added or dropped after a follow share is printed.** The experiment
has no screen and no stop rule on a score, because prediction is short and every fold month is needed
for the interval.

**Timing.** The run starts straight after F4's counted fits, on the manager's word. Phase 0 can run as
soon as the second code review passes. The S6 reading waits for F4's stored MSA forecasts on the
second generator. Before phase 1 the implementer books the GPU slots with the manager against the
streams of F4 and F7, and sends "predictions starting" with the frozen checkout's commit to the
manager before phase 0 and again before phase 1.

**Machine rules, all through `run.sh`:**

- Every command runs under `systemd-run --user --scope` with `MemoryMax=12G`, `MemorySwapMax=0`, and
  `OOMPolicy=continue`.
- One thread for every numerical library (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `POLARS_MAX_THREADS`, and `RAYON_NUM_THREADS` are 1, and
  `torch.set_num_threads(1)`).
- The wrapper calls the virtual environment's Python directly after `uv sync --group gpu`, and never
  `uv run` inside the scope.
- At most two GPU processes run at once, each under 6 GB of GPU memory, and the wrapper refuses to
  start when under 25 GB of host memory is available.
- One process per job, atomic writes, and a finished job is skipped only when its own output file
  exists under this experiment's data folder.
- The Candidate's seeds 2 to 4 run one seed per process, because three seeds in one process reached
  the 12 GB cap in experiment 713.
- No download, no new data source, no spend, and no change under `src/` or `packages/`. The experiment
  writes nothing into the data folders of experiments 581, 650, 667, 709, 713, 714, or 715, and edits
  no file of experiment 715.

## Method

**Architecture.**

- `M1`, `K`, `E`, and `N` are the stage 3 states of experiment 713's axial-attention network over a
  matrix of the last 14 days by 48 half-hours, unchanged, and run from the frozen checkout.
- The Candidate is experiment 667's demand network (arm `P7+D50`), which reads a day-old lag, weekly
  lags, and electrical-neighbour summaries, run from its saved weights `models/P5/<month>_d0_s<seed>.pt`.
- The shift enters the power inputs only, through the same visibility rule as the real power in both
  networks, and the targets are never shifted.

**Training regime.**

- No network is trained, refitted, or fine-tuned. GPU time is prediction only, about 3.5 to 4.6
  GPU-process-hours.
- The stored weights were fitted by experiments 667 and 713 on a monthly refit, with experiment 581's
  synthetic shifts in fine-tuning.

**Measurement.**

- The bank is experiment 581's, 24 passes over 8 fold months, and the counted rows are the origins
  with `d` from 0 to 20 at leads from 24 to under 120 hours.
- The score is the follow share of the median forecast, and the counted contrasts are T1 and L1 by
  group, by the decision rule above.
- The counted Candidate run shifts the host only; the all-shifted run is report-only.

## Reviews

An Opus agent reviewed the design twice and a re-check followed. The re-check's should-fix items are
applied in this page: the 0.50 sign rule applies only to cells that would otherwise read rows 2 to 4,
the host-only fallback fires only after phase 5 and the all-shifted Candidate run are cut, the
`production_status` departure is stated, the three traps in the copied code are a gate of the code
review, the host-only sets number 4, and F4's commit is pinned at `ac3e768e4`. Still to come: an Opus
code review of each new script before any run, a second review of the fixed code, a result review, an
audit if a trigger fires, a numbers review, and a prose review.

## Files

- `data/ml-experiments/briefs/msa_architecture/followshare_design.md` and
  `followshare_design_review.md` (read, private): the design that this page condenses, and its review.
- Read from experiment 713 ([pull request 402](https://github.com/openclimatefix/nged-ml-research/pull/402)):
  the stored stage 3 states and forecasts of `M1`, `K`, `E`, and `N` under its data folder, its
  `candidate_seeds/` folder, and the code of `e00713_era.py`, `e00713_arms.py`, and its layout and
  batch modules, from the frozen checkout.
- Read from [experiment 581](../00581-robust-multiweek-lag-encoder-x3/result.md): `e00581_bank.py` and
  `e00581_tails.py`. Read from [experiment
  667](../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md): the saved
  weights, tables, and `prevailing/p667_inject.py`. Read from [experiment
  650](../00650-extended-shift-departure-head-against-decayed-recent-error/result.md): the event
  generator and `c650.world()`. Read from [experiment
  709](../00709-era-signal-in-the-candidate-part-1/result.md): the bank and its stored forecasts. Read
  from [experiment 714](../00714-msa-era-identifiability-audit-and-frozen-probe/result.md): the second
  event generator.
- Read from experiment 715 (F4) at commit `ac3e768e4`: `e00715_era.py` for `event_crps`,
  `pass_bootstrap_ci`, and `g2_panel`, and its stored second-generator MSA forecasts.
- To be written by this experiment: the scripts named above under
  `ml-experiments/00726-msa-candidate-follow-share-on-injected-shift-bank/`, and forecast files under
  `data/ml-experiments/00726-msa-candidate-follow-share-on-injected-shift-bank/candidate/<world>/<month>/`
  and the MSA equivalents, never committed. No code has been written.
