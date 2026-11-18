# Experiment 702: order of the stages

**Run the stages in this order, from the repository root, and stop at the first failure.** Every
command needs the environment below. The plan is `docs/ml-experiments/00702-*/result.md`. No stage
after step 6 starts until the reviewer has read the output of steps 1 to 6.

```bash
export DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data
export NGED_S3_BUCKET_URL=https://example.com NGED_S3_BUCKET_ACCESS_KEY=dummy NGED_S3_BUCKET_SECRET=dummy
source ml-experiments/00524*/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
D=ml-experiments/00702-per-substation-versus-global-training-of-the-modular-demand-network
py() {  # every Python command: its own scope, 16 GB host cap, no swap
  systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 -p OOMPolicy=continue \
    uv run --no-sync python "$@"
}
```

**Every Python command runs in its own systemd scope with a 16 GB cap and no swap, and with one
thread per maths library.** `py` above does this for the CPU commands, and `gpu.sh` does it for
the GPU commands. The thread variables are exported before any Python process starts, so no worker
pool starts with more threads.

**Source `env.sh` once per shell.** It prepends every experiment folder to `PYTHONPATH`, about 107
KB, and a second copy passes Linux's 128 KB limit on one environment string, at which every new
process fails with "Argument list too long". `gpu.sh` empties `PYTHONPATH` and sources `env.sh`
itself, so a GPU command runs the same from a shell that has sourced `env.sh` and from one that has
not.

Every CPU command is `py $D/<script>.py ...`. Every GPU command goes through
`$D/gpu.sh`, which the chain scripts call. Logs are in `$DATA_PATH_INTERNAL/ml-experiments/00702-*/logs/`.

**1. Spans and shifted-row hashes of the pooled world, for the pilot fold and the four test months.**
`chain_arm.sh` and `chain_weather.sh` run this when the file is missing; running it first shows a
disagreement with `core.fold_span` before any fit. The command asserts the span formula.

```bash
for M in 2026-05 2025-11 2026-01 2026-04 2026-07; do
  E00677_ROOT=$DATA_PATH_INTERNAL/ml-experiments/00702-per-substation-versus-global-training-of-the-modular-demand-network \
  E00677_SPAN=1 E00677_MONTH=$M E00677_PHASE=predict $D/gpu.sh e00702_fit.py split3 spans --month $M
done
```

**2. Rescore experiment 677's stored forecasts** (CPU only). It must agree within 0.01 points.

```bash
py $D/e00702_score.py match-stored
```

**3. Control refit on the pilot fold, then the refit match check.**

```bash
SEEDS="0" SUFFIX=_pilot $D/chain_global.sh 2026-05
py $D/e00702_export.py global_pilot 2026-05 --seeds 0 --name global_pilot
py $D/e00702_score.py match-refit
```

**4. Bank match check: this experiment's bank on experiment 677's stored global networks.**

```bash
SEEDS="0" $D/chain_global.sh 2026-05 bank-stored
py $D/e00702_score.py match-bank --months 2026-05 --seed 0
```

**5. Truncated-archive weather test, for the pilot fold and then for each test month before that
month's first scored fit.** The run time is unmeasured, so a first run can pass `D15,D18` as the
second argument.

```bash
for M in 2026-05 2025-11 2026-01 2026-04 2026-07; do $D/chain_weather.sh $M; done
```

**6. First run: the four arms at seed 0 on the pilot fold, at the three Primaries of the plan's rule.**
Print the three Primaries with `py -c "import e00702_score as s;
print(s.pilot_primaries())"` (the list below assumes `D14,D2,D15`). Split-at-1 runs its `isolation`
check inside the chain. The `_b8192` runs are the batch diagnostic.

```bash
P=D14,D2,D15
for ARM in split3 split2 split1 hybrid3; do SEEDS="0" $D/chain_arm.sh 2026-05 $ARM $P; done
for ARM in split1 split3; do SEEDS="0" $D/chain_arm.sh 2026-05 $ARM $P --batch 8192 --suffix _b8192; done
for ARM in split3 split2 split1 hybrid3; do
  py $D/e00702_export.py $ARM 2026-05 --seeds 0 --primaries $P --name pilot_$ARM
done
for ARM in split1 split3; do
  py $D/e00702_export.py $ARM 2026-05 --seeds 0 --primaries $P \
    --suffix _b8192 --name pilot_${ARM}_b8192
done
py $D/e00702_score.py pilot --files global_pilot pilot_split3 pilot_split2 pilot_split1 pilot_hybrid3 pilot_split1_b8192 pilot_split3_b8192
```

The phase logs (`arms/<arm>/2026-05/<label>/work/logs/phases_*.json`) hold the fixed-epoch phases, the
validation loss, the wall-clock time, the peak memory, and the floored columns. Every number from this
step is interim.

**7. Test months, for each M in 2025-11 2026-01 2026-04 2026-07.**

```bash
$D/chain_global.sh $M                                  # control pair, seeds 3 4 5
for ARM in split3 split2 split1 hybrid3; do $D/chain_arm.sh $M $ARM; done   # seeds 0 1 2, all 16 Primaries
for ARM in split3 split2 split1 hybrid3; do py $D/e00702_export.py $ARM $M; done
py $D/e00702_export.py global_pair $M --seeds 3 4 5
```

Export writes `backtests/<arm>.parquet` from every month it is given, so export all four months in one
call per arm when the four folds have finished: `e00702_export.py $ARM 2025-11 2026-01 2026-04 2026-07`.

**8. Scores.**

```bash
py $D/e00702_score.py score
py $D/e00702_score.py bank
```

**Not implemented:** the load-shift scenario slices, the example-forecast and cold-night figures,
and the plan's real-data isolation test (random values for the other 15 Primaries). `isolation`
replaces the last with hash equality of every Primary's cleaned row, shifted rows, and extra-column
blocks against the pooled world, plus a guard on the file readers.
