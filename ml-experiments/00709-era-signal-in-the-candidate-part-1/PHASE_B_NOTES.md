# Experiment 709, phase B: notes for the reviewer

**Phase B has five scripts and no fit has run.** `e00709_signal.py` builds the era columns,
`e00709_model.py` holds the network, `e00709_cand.py` fits and scores one fold, `e00709_checks.py`
gates the chain, and `e00709_chain.sh` runs the folds. The checks that train a tiny network on
random inputs (`--synthetic`) have not been run, because the brief forbids any training before an
Opus review. The forward-only checks (identity at initialisation, anchor, double counting, O-enc off
on the lag side, translation, as-of alignment) were run and pass.

## Files and the code each one reuses

- **`e00709_signal.py`** reads phase A's `e00709_classify`, `e00709_stage0`, `e00709_inject`, and
  `e00709_features`. It writes `signals/<arm>.parquet` and `signals/<arm>_episodes.parquet` under
  the experiment's data folder.
- **`e00709_model.py`** subclasses `PermLagMdn` from
  `ml-experiments/00613-nra-roster-contrasts-with-intervals-five-seeds/e00613_nbmodel.py`. It uses
  `widen_first_layer` from `ml-experiments/00696-*/e00696_model.py`. The anchor follows the
  `OutputOffsetDrift` pattern of experiment 565 and reads the offset knots of `LagOffsetMdn`
  (`ml-experiments/00572-*/e00572_model.py`).
- **`e00709_cand.py`** is a copy-and-adapt of
  `ml-experiments/00667-*/e00667_cand.py`, with the injected-bank scoring idea of
  `ml-experiments/00693-*/e00693_cand.py`. It imports `e00574_fit` (the fit loop, `train` copied
  into `train_curriculum`), `e00667_fold` (the fold patches), `e00667_export.contract_rows`, and
  `studies.experiment_store.write_predictions`.
- **`e00709_chain.sh`** follows `e00696_chain.sh` (the `checks.json` gate) and the `gpu.sh` of
  experiment 667 (the systemd-run wrapper).

## Ambiguities in the plan and the reading chosen

- **Anchor mode on the cut day.** An era whose onset falls on the fold's cut day or later is
carried by no offset knot, so the rule is `onset >= cut` (mode 1), not `onset > cut`. A learned arm
uses mode 2 when `onset < cut <= confirm`. The plan's clause about a row's valid time falling after
the newest knot is read as already covered by the onset rule, because stage 3 rows end at the cut.
- **Mode 2 formula.** The anchor is `clip(full - (offset(newest knot) - offset(onset knot)), min(0,
full), max(0, full))`. The offset term is detached from the graph, so the anchor does not train the
knots.
- **Curriculum "half of the passes".** Read as the odd epochs of the stage 3 loop. Even epochs read
  the plain tensors and odd epochs read tensors with the injected world (`ALT`). Validation always
  reads the plain rows.
- **Curriculum world.** One world per fold and seed, which merges up to 4 phase A passes with
  disjoint touched Primaries (up to 16 tries), world `curriculum`, pass indices
`(month_index * MAX_SEEDS + seed) * 16 + try` with `MAX_SEEDS = 10`. Onsets are 120, 100, 80, and 60
days before the cut. The plan says onsets lie 120 to 40 days before the cut, and the injector adds a
random offset inside its window, so the realised onsets may fall later than 60 days before the cut.
- **Curriculum signals.** Learned arms overlay the injected passes' signals on the real signal for
  the touched Primaries. The oracle, the placebo, and oracle-enc see the real record's events plus
  the injected events in the alternate epochs, and the placebo permutes that union.
- **Curriculum targets.** Era hosts shift both the target (`z` plus the era change over the target
  standard deviation) and the lags. Switching events shift the lags and the neighbours only, and the
  target stays clean (the NRA target excludes a transfer).
- **Lag translation reach.** Only the first 14 days after an era's onset, or the whole switching
  event up to 120 days, are translated, in blocks of 7 days that read the band changes at the first
  day of the block (entry of the day before, so the translation is causal). Era translation reaches
  back 60 days before the onset. The neighbours' summaries and the exponentially weighted columns
  are rebuilt from the changed own-site power, so the translation is partial for the neighbour
  columns that read other Primaries' power.
- **Soft arm.** The anchor is `p * level`, the unknown flag is 0, an era translation uses `p`, a
  switch translation uses `1 - p`, and `p` is rounded to a quarter so that rows share a translation.
- **Rows held out.** An episode whose alarm fell before the cut is held from its onset to the day
  before its first `era` entry when the class is `era`, and over its whole open span when the class
  is `unknown` or `switch`. The mask is applied by the row's valid-time day to
  both training and validation rows. Rows from before the current era's onset keep weight 0.1.
- **Placebo permutation.** Era switches and switching events are permuted separately. Each keeps its
class, vector, and duration and takes a random eligible Primary (never D8, D15, or D18) and a random
onset day at least 28 days from every true event and from the other permuted events of the same
Primary. The placebo draws a fresh permutation in each curriculum world.
- **Era-id table.** 129 rows: slot 0 is shared by every id first seen at or after the cut and by the
  curriculum's rows. The table is zero-initialised and has an L2 penalty of 0.01 on its mean square
  (the weight decay of the plan), added in `penalty()`.
- **Penalty.** `LagOffsetMdn.penalty` is unchanged (plan section 2), so the override only adds the
  id-table term. There is no break in the first difference.
- **Encoder initialisation.** The era encoder has ordinary random initialisation. A zero last layer
  would give zero gradient to both the encoder and the zero-initialised widened mixer columns, so
  the era path would never train. Identity with `PermLagMdn` holds because the widened mixer columns
  and the O-enc layers start at zero.
- **Identity tolerance.** 1e-6 at initialisation, 1e-5 after training on `none` rows, because the
  widened columns add zero products in a different summation order.
- **Predictions.** One prediction set per arm, fold, and seed, named `<arm>-<month>-s<seed>`, in
experiment 709's store, with row set `experiment_677_monthly_refit_rows_valid_2025_11_to_2026_06`.
That `row_set_id` is the plan's own string and is unchanged. It means: contract origins at 12:00 UTC
on each day of the fold's month, 14 days of half-hours, 16 series, `valid_time` before 2026-07-01. A
set that exists is left unchanged. `write_contract` drops only the rows whose `valid_time` is on or
after 2026-07-01.
- **Bank.** 16 passes per scored month with the BANK seeds and `pass_index = month_index * 16 + j`,
  onsets on the fold's first day, scored on the touched Primaries and their neighbours only. Each
pass writes the forecast quantiles, the injected truth in capacity share, and the pass's truth table
under `bank/<arm>_<month>_s<seed>/`. The stage 3 model is the saved fit and is never refitted.
- **Pilot.** `--pilot` runs 4 epochs on 10% of the rows each epoch, forecasts the first 3 origin
  days, and runs 1 bank pass, all in a folder whose name ends in `_pilot`. It writes no prediction
  set. It prints the fit, forecast, and bank timings, which the chain keeps in the logs.

## Resolution of the Opus review (B1, B2, S1 to S16, minor findings)

All findings held and were applied. None was rejected.

- **B1.** The bank builds `lag_x` from the injected panel (`e.block_of` on the injected lag power,
  as in `e00693_cand.py:971-978`) and passes it to `forecast_window`. `e00709_checks.py --alignment`
  checks that a host's lag column 37 equals the clean lag before the onset and differs after it.
- **B2, S16.** Hard arms show an era vector only for the states unknown and era. A switching row
  gets no era vector and no anchor, and the soft arm weights by p(era). "No era" has its own
  forced-zero slot (`SLOT_NONE = 1`), the shared unseen-era slot is 0, and real slots start at 2, so
  no world id can map onto a real Primary's slot.
- **S1.** A row of an injected event takes the cut `onset - 14` days.
- **S2, S3, S4.** Described under the curriculum signals above. The oracle's injected era vector is
  the band mean of the injected change over 28 days from the onset, which reads nothing after the
  cut. Both translations of a (Primary, onset, block, p) group apply to one panel.
- **S5.** Bank truth is NaN on or after 2026-07-01, and a `clean_share` column is stored.
- **S6.** The weather check asserts, from the `build_test` run, that the run is 00:00 UTC of the
  origin day, that its demand frame has that `nwp_init_time`, and that its summary file exists.
- **S7.** The curriculum world index uses `MAX_SEEDS = 10`, so seeds do not collide. Seeds above 1
  stay unavailable, because experiment 667 has stage 1 and stage 2 weights for seeds 0 and 1 only.
  Seeds 2 to 4 need new stage 1 and stage 2 chains, and the manager decides about them.
- **S8.** Logs hold `peak_gpu_gib` (`torch.cuda.max_memory_allocated`) and `peak_rss_gib`. The
  curriculum holds only the columns it changes (37, 40, the signal block, the era block) and never a
  second copy of the matrix. The chain checks the free GPU memory with `nvidia-smi` (`GPU_NEED_MIB`,
  default 6000 per job) and defaults `--mem-cap` to 16G.
- **S9.** The chain changes to the repository root before it sources the environment file.
- **S10.** Rebuilt translated rows get `add_training_noise`.
- **S11.** `preflight()` raises one `FileNotFoundError` that lists every missing input. It creates
  no file.
- **S13.** `hold_mask` holds the whole span of a withdrawn (reversed) event.
- **S14.** The logs and the pilot report `alt_passes` and `alt_eras`. Later passes take more eras so
  that the curriculum reaches 8 eras per fold and seed, within the host-separation rule.
- **S15.** World signals are cached under `worlds/` (curriculum by month and seed, bank by month and
  pass).
- **Minor.** The era block uses `np.where` and `torch.where`. `check_signal_arm` fails when no
  sampled row hits a test day, and its dead oracle branch is gone. The pilot shortens the `none` arm
  and tags its logs. Placebo seeds are `PLACEBO_SEED_BASE[world] + index`. The shared feature helper
  is `sg.pass_features`. Constants are `Final`.
- **Penalty.** The penalty on the widened columns is unchanged from the plan: `ID_L2 * table**2`
  added to the base penalty.

## Reads and writes outside the experiment's folder

- Experiment 667's stage 1 and stage 2 weights, tables, and inputs are only read. `require_tables`
  raises if a table is missing, so this experiment never builds a table.
- `fold.install` calls `e00667_extend.redirect()`, which can write a one-row stub frame under 667's
  `inputs/frames` when a run has no frame. Those frames already exist for the scored months.

## What could not be verified

- **No fit, no pilot, no bank pass has run.** Memory, speed, and the translation cost are unknown.
  `sig.signal_columns` rebuilds whole-grid arrays on each call, so a fold with many translated
  groups could be slow. The real record has about 100 groups, and an alternate world about 50.
- **GPU memory.** The alternate tensors double the training matrix (about 5 million rows by about
  120 columns in fold 2026-03), so the fit needs roughly 5 GB more than experiment 667's. The
  alignment check built that fold's rows on the CPU within the 8 GB cap and passed
  (`alignment_passed` is true in `checks.json` for 2026-03; the other two gates are not set).
- **The signal tables do not exist yet.** `python e00709_signal.py` builds them from the frozen
stage 0 models on the CPU. `e00709_checks.py --signals` then checks the cut behaviour on each table.
- **The training checks have not run.** `--synthetic` trains tiny networks on random inputs
  (identity after training, O-enc liveness, frozen encoders).
- **Attribute names.** `test.valid`, `test.origin`, `test.origin_idx`, and `test.target_idx` come
  from the use in experiment 693's script. The forecast frame's `time_series_id` dtype is assumed to
  cast to Int64.
- **Real-record oracle.** The era table's rows without an era index (the D8 fade, the D8 row with no
  dates) are skipped. The verified events of experiment 580 include both donor and recipient rows.
- **Alternate world on the extended grid.** The clean stack is longer than the panel (121,248
  half-hours against 119,328), so every injected change is added to the first panel-length slice.
