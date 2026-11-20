# Notes for the reviewer: choices where the design is ambiguous

Each line is the simplest reading that was taken. Not content to publish.

1. **`clean` power is the observed power.** The injection adds shifts to
   `common.series_and_arrays()` power
   (NaN stays NaN), not to experiment 581's `corrected` series, so an empty world equals the cached
   lags bit
   for bit (test 5).
2. **Rows of the one-stage arms.** Training rows are the non-held-out, non-purged rows of window F
   inside the
   stage's window. `V_sel` is the held-out rows with age under 730 days (`recent_held`). Only
   `S_era`,
   `S_ens`, and `M3` select on all held-out rows (`all_held`), as 707 does.
3. **Fine-tune rows come from window-F rows with age under 730 days**, not from a second
   `build_fold`. Test 4
   asserts the same rows, order, and weights as `build_fold` with `W24`. `build_fold` is still
   called once per
   fold to fit the standardiser, target statistics, and spread table on unshifted rows.
4. **Step cap.** Checked per batch (training stops inside an epoch when it is reached) and counts
   the optimiser
   steps of one stage. `m2` = 40 x steps/epoch of window F + 40 x steps/epoch of the last 730 days,
   from this
   world's training-row counts.
5. **Schedule `S`.** Early stopping and "best" selection count from epoch index 5 (epoch 6). Epochs
   1 to 5 never
   select. `CYC` is fixed at 12 epochs and selects the lowest `V_sel` among epochs 9 to 12 (index 8
   to 11).
6. **Cosine restarts** are per step, with `cycle_epochs * steps_per_epoch` steps per cycle; the rate
   ends at a
   tenth of the start rate. AdamW state is kept across cycles (one optimiser).
7. **Sampler.** An epoch has `ceil(n_train / 16384)` batches of exactly 16,384 draws. Weighted
   shares are
   logged from the sampling weights.
8. **EMA** averages parameters only (the network has no buffers), starts as a copy of the `S_ens(F)`
   state, and
   is the network that is validated, selected, and saved.
9. **Placebo.** The age of a row of month `j` is `refit - (start(pi(j)) + offset)`, so ages still
   fall as valid
   time rises inside a month. The offset is clipped to the last half-hour of month `pi(j)` and to
   the refit. The
   design says "age of the first half-hour plus offset", which would make ages rise inside a month.
   `P` uses the
   half-life `--h-t` (required for Phases S, C, and SC) and writes to the arm folder `P-i`. A rerun
   with another
   `--h-t` stops, because the log of each stored `P-i` fit records `h_t` and `run_group` compares
   it.
10. **`PP`** reads its half-lives from a JSON file (`--pp-half-lives`, keyed by site label, `"inf"`
    allowed).
    `e00710_score.py --tuning --write-pp-half-lives PATH` writes the file (the lowest mean CRPS on
    folds 2024-10
    and 2025-01 among `W-h0.5-i`, `W-h1-i`, `W-h2-i`, and `M3-i`, which is `inf`).
11. **Shifted-lag rows** (bank scoring) are rows where any of the three lags reads a shifted
    half-hour
    (`origin_lags` on the shift series). In-event rows are those with origin inside the event,
    after-event rows
    those with origin in the next 7 days; both need a shifted lag. The NRA actual is the observed
    power.
12. **Real events** are forecast as bank pass 100 (corrected power) for months where an event
    overlaps the month
    or the 7 days before it. December onsets are labelled "real-event result"; every other onset is
    labelled
    flagged.
13. **Tuning folds** score the cached ENS rows whose origin is in the month with lead up to 240 h,
    dropping
    flagged exact-zero targets; no bank passes. The tuning score file holds `lead_hours` and `y`.
14. **Parsimony hyper-parameter count** is a convention in `hyper_parameter_count` (half-life 1,
    floor 2,
    schedule 3, per-Primary 4, sampler 1, EMA 1, cosine 3, a trailing window 1).
15. **Intervals on absolute metrics** use the half-hour mean (not the equal-weight mean over
    Primaries) with
    20,000 month-block resamples at 95%. Contrasts use 100,000 at the family level via
    `metrics.month_bootstrap_contrast`. Slump contrasts (I4 to I6) resample events by calendar month
    with
    `studies.bootstrap._resample_bounds` (private function).
16. **Known-answer marker.** `work/known_answers_passed.json` holds one entry per passed test and
    records the
    SHA-256 of the experiment's scripts, so any code change makes the fit refuse until the tests are
    re-run. `e00710_fit.py` needs all seven entries
    (the pilot and `--no-marker` skip the check).
17. **Not implemented:** `e00710_checks.py` and `e00710_tune.py` are folded into
    `e00710_known_answer.py` and
    `e00710_score.py --tuning`; the `-i`/`-n` slump per lead band and the real-event plots are not
    produced.
18. **Nothing was run.** Only import checks, `ruff`, `ty`, and a tiny random-data smoke of the loop
    features
    (flat, sampler, EMA, cosine, step cap, last-epoch selection, per-epoch rates) on the CPU.

## Review 1 triage

Verified against the code, then applied or rejected. File: `briefs/ens_one_shot/code_review_1.md`.

- **B1 applied.** `e00710_arms.job_tier` and `plan_jobs` (sorted by tier, seed, fold; Phase `SC`
  joins S and C);
  `e00710_fit.py --tier`, `--dry-run` (prints tier, seed, fold, state), `--gpu-hours` (planned and
  finished
  GPU-hours per tier from the logs' `seconds_total`, then the folds both arms of each contrast hold
  after
  `--budget-hours`, default 10); `e00710_score.py --interim` (own folds and seeds per contrast,
  `INTERIM:` banner
  on every table, `INTERIM, no verdict`). Interim prints contrasts, the two arms' CRPS, and the
  convergence
  table only; the slump tables and real events wait for the full run. Window-F rows are rebuilt per
  group.
- **M1 applied.** `is_candidate` removes the reference, the placebo, and any `S_era` arm from `B*`
  and `O*`;
  `M1-i`'s tuning score and the two finalists (plus `O*` if outside them) print.
- **M2 applied.** `M3` has `tuned=True`; it forecasts from the `ensF` state of `M2` in the same
  group.
- **M3 applied.** `work/tuning_per_primary.parquet`, a per-Primary table, and
  `--write-pp-half-lives`.
- **M4 partly applied.** `--dry-run` prints the variant list and one frame's size, and the forecast
  frames are freed
  after scoring. The two-pass scoring is not written, because the 33-frame estimate is unmeasured;
  measure the
  peak RSS on a 2-fold run first.
- **M5 applied.** A `None` guard blocks `supported` (except the clear-sky-index slice); the `B*
  minus M2` guard is
  `None` when absent.
- **M6 applied.** `claim_state` takes a per-state `.lock` file (`O_CREAT | O_EXCL`, stale locks
  removed by pid) and a
  waiting stream loads the state that the first stream saved; state, JSON, and forecast temporary
  names carry the
  pid. `save_fit` writes to a per-job folder, so two streams must still never share a job. Streams
  may therefore
  split by arm.
- **M7 applied.** `--shift-share-range LOW HIGH` (required outside the pilot) stops after any
  `i`-state stage outside
  the range; `tuning_scores` prints `STOP:` if `M2-i` is above `M1-i` or `M2-n` by more than 0.10
  points.
- **m1 applied.** `relag` adds `lag(world) - lag(observed)` (zero where not finite) to the cached
  lag values.
- **m2 partly applied.** The tautological asserts are removed and the example prints in the first
  group of every
  run. Carrying `nwp_init` into the print is rejected: `prep.assert_origin_after_run` is the guard
  and the print is
  informational.
- **m3 applied.** Phase C adds `B*-n` only if `needs_new_n_fit(B*)`.
- **m4 applied.** `M3` is in the minimal Phase S plan.
- **m5 applied.** `--h-t` is required for Phases S, C, and SC, and `run_group` stops if a stored
  `P-i` log has another
  `h_t`. The folder name is unchanged.
- **m6 applied.** `tuning_rows` keeps only valid times before the first scored fold's refit.
- **m7 partly applied.** The secondary contrasts (`S1`, `S2`) print with their own verdict line, and
  every
  contrast of C1 to C7 gets a t test. The other descriptive readings (without 2026-01, per lead
  band, D15 and D18,
  the load-shift scenario) are rejected for now and need adding before the write-up.
- **m8 applied.** `lag_shift_mask` takes the observed power and ignores a lag that reads a missing
  observation.
- **m9 partly applied.** Real-event rows use `lag_shift_mask`, and the shifted-minus-corrected error
  prints beside the
  NRA error. Labelling by (transfer, Primary) is rejected, because the D5 and D16 transfer
  identifiers are not in
  the code.
- **m10 applied.** Absolute intervals resample the 12 months' sums and counts (20,000 draws, fixed
  seed).
- **m11 applied.** The marker records the SHA-256 of the scripts; the fit and the tests compare it.
  No marker file
  exists yet, so none was deleted.
- **m12 rejected.** Test 3 on one fold is acceptable. Extending test 6 to `bank_rows` needs
  synthetic frames and the
  real Primary table, and cannot be checked without running it.
- **m13 partly applied.** `--finalists` has `choices`; `build_fold` no longer holds `x_shift` twice
  without a
  world; the tuning scorer uses `average_files`. `--bank-only` is rejected: a job whose forecast
  exists is
  never refitted, and `--no-bank` is for the pilot only.
- **Design text applied.** The placebo age sign and the 5e-4 rate of `M1`'s `S_ens` stage are
  corrected in `design.md`.
- **Not run.** Only `ruff`, `ty`, a planner dry-run with `--gpu-hours`, and synthetic checks of
  `parsimony_pick`.

## Review 2 triage

- **N1 applied.** `plan_jobs(phase="T")` leaves `PP` out unless `include_pp` is set, which the fit
  sets when `--pp-half-lives` is given. `run_group` also skips a pending `PP` with a printed line
  when no file is given, so it never crashes. Second pass: score with `--tuning
  --write-pp-half-lives FILE`, then run `--phase T --pp-half-lives FILE` (finished jobs are
  skipped). A `PP` finalist in Phase S, C, or SC without the file stops at the start.
- **N2 applied.** `is_one_shot` selects the `O*` pool by stage name (a single `ONE` or `CYC` stage),
  which excludes `M3`. Checked on the synthetic ranking with `M3` best: `O*` is `W-h1`, `B*` is
  `M3`.
- **N3 applied.** `plan_jobs` adds `O*` to the `i` arms of Phases S and SC (not Phase C, whose arms
  the design fixes). The scorer's `--tuning` writes `work/tuning_choices.json` (`b_star`, `o_star`,
  `h_t` as the best `W-h` arm, `finalists`). The fit (Phases S, C, SC) and the full scorer read it
  for every flag not given and stop if it is missing, so no value is hard-coded. The choice is
  rewritten each time `--tuning` runs, so run it again after the `PP` pass.
- **N4 applied in part.** The fold scale is cached for every fold (no `clear`), so 12 builds, and
  each build's seconds go to `logs/month_scale_seconds.json`; `--gpu-hours` adds the builds still to
  do. The window-F and ERA rows are still rebuilt per group, because they depend on the world and
  seed; that rebuild falls inside the first job's `seconds_total`, so it is counted. Writing the
  scale to `.npz` is rejected: the cache suffices within a stream.
- **N5 applied.** `write_bank` has a required `overwrite` argument, which `forecast_job` sets, so a
  refitted job replaces every bank pass. `run_stored` keeps the skip (its networks are fixed).
- **n1 applied.** A lock is claimed by `os.link` from a pid file, so it is never empty. An
  unparsable lock older than 60 s is stale. A stale lock is renamed to a private name before
  removal, and put back if it proves live. A waiter prints `waiting for <lock>` once.
- **n2 applied.** `--shift-share-range` is required outside the pilot whatever `--no-marker` says.
- **n3 rejected for now.** Recommendation for the operator: after the pilot, use a wide range such
  as 0.02 0.5 rather than one tight around fold 2026-01; the range is printed with the choices only
  through the command line. Computing the 60 worlds' coverage first needs a run.
- **n4 applied.** `--tuning` writes `work/stop_checks.json`; `--phase S|C|SC` refuses unless both
  checks exist and passed (a dry run and `--gpu-hours` do not need it, and `--no-marker` does not
  skip it). The minimal Phase T now keeps the `n` state of `M2`.
- **n5 applied in part.** The origin-time example prints in the first group with pending work; the
  interim convergence table has an `INTERIM` heading; the per-Primary parquet and the `PP` JSON are
  written atomically. The `relag` check of cached lags against the observed series is rejected for
  now, because it needs a run: the pilot's log does not print it.
