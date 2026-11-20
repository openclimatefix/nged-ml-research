# Pre-registered extension E1 and E2: commands and job counts

Run these after the minimal plan (Phases T, S, and C) and its review. Each command goes through
`run_phase.sh`, which sets the one-thread BLAS and Polars variables, the 16 GB memory cap, and
`PYTHONPATH`. Set `DATA_PATH_INTERNAL`, `NGED_S3_BUCKET_URL`, `NGED_S3_BUCKET_ACCESS_KEY`, and
`NGED_S3_BUCKET_SECRET` first (see `CLAUDE.md`). Add `--dry-run` to any fit command to list the jobs.

Write `M="2025-07 2025-08 2025-09 2025-10 2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06"`
for the 12 scored folds.

## E1: `M1h90-i`, 36 fits

`M1h90` has the stages `S_era`, `S_ens` (from `S_era`), and a fine-tune with a 90-day half-life. The
first two stages have the state keys `era` and `era+ensF`, the same as `M1`, so the fit loads
`pretrained/era/i/<fold>_s<seed>.pt` and `pretrained/era+ensF/i/<fold>_s<seed>.pt` that `M1-i`
stored, and trains only the fine-tune. If `M1h90` ran first, it would train and store those two
states, and `M1-i` would then load them. The `i`-state world depends on the fold and seed only, so
the two arms see the same synthetic events.

```bash
ml-experiments/00710-ens-only-one-shot-training-recipe/run_phase.sh e00710_fit.py \
    --arms M1h90 --state i --months $M --seeds 0 1 2 --device cuda --shift-share-range 0.02 0.5
```

## E2: the `-n` states for I3 to I6, 72 jobs

**(a) `FTh90d-n`: 36 fits with bank passes.** The stage `S_ens(F)` has the state key `ensF`. In the
`-n` state of a scored fold with seeds 0 to 4, the fit loads experiment 707's stored
`pretrained/ensF_ftw730` (the same state that `M2-n` and `M3-n` use) and trains only the fine-tune.
The fit writes the bank passes of every fitted job.

```bash
ml-experiments/00710-ens-only-one-shot-training-recipe/run_phase.sh e00710_fit.py \
    --arms FTh90d --state n --months $M --seeds 0 1 2 --device cuda --shift-share-range 0.02 0.5
```

**(b) `M1-n`: 36 bank passes, no fit.** `M1-n` is experiment 707's stored `A_F` network, so its
forecast exists and only the bank passes of seeds 0 to 2 are new. The scorer pairs `M1-i` with
`M1-n` on the seeds whose bank passes exist in both states.

```bash
ml-experiments/00710-ens-only-one-shot-training-recipe/run_phase.sh e00710_fit.py \
    --stored M1-n --months $M --seeds 0 1 2 --device cuda
```

`M2-n` also needs bank passes, which the minimal plan did not write (no `work/bank/M2-n` folder
exists). Run the fourth command, so that I5 can be read (36 passes, no fit):

```bash
run_phase.sh e00710_fit.py --stored M2-n --months $M --seeds 0 1 2 --device cuda
```

If `FTh90d` is not `B*`, I3 and I6 read `B*` and so need `B*-n`; E2 as approved covers
`FTh90d` and `M1` only.

## Reading the results

```bash
ml-experiments/00710-ens-only-one-shot-training-recipe/run_phase.sh e00710_score.py --interim
ml-experiments/00710-ens-only-one-shot-training-recipe/run_phase.sh e00710_score.py
```

`--interim` prints E1 on the folds and seeds that both arms hold, and the full run prints E1 beside
C1 and I1 to I6 (I3 and I6 read `B*-n`).
