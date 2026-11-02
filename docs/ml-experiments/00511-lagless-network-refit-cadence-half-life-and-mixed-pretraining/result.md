---
id: "00511"
slug: lagless-network-refit-cadence-half-life-and-mixed-pretraining
title: >-
  How often should the lag-free demand network be refitted, which recency half-life suits it, and
  does pretraining on ERA5, CAMS, and ENS rows together help?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted network weights, window forecasts, and stale-model forecasts, under this
  experiment's private data folder only; experiment 510's models and forecasts are read, never
  written. Serving path: not touched (this repository holds none). Degradation rule: not touched.
  Designs: more than one defensible (what a staleness curve means for a daily refit, how a warm start
  is defined, how a mixed pretraining weights its sources), so the plan fixes each before any score
  is read. Callers: this folder's scripts import experiments 397, 425, 428, 448, 470, and 510 and the
  `studies` evaluation code. One Sonnet code review before the first fit, one Sonnet results review,
  and two Sonnet write-up reviews.
hypothesis: >-
  Refitting the lag-free demand network more often than weekly (twice weekly or daily) closes more
  of the 0.13 to 0.16 point gap that experiment 510's best arm L-Ww (weekly, 90-day half-life) left
  to K2-S1, because the gain from weekly over monthly refits grew through the month and so is a
  staleness effect. A 30-day or 60-day half-life closes part of the gap at the price of a larger
  leak of an injected load shift. Pretraining on ERA5 and CAMS rows together with ENS rows,
  with a source flag, separates the two jobs of the network (learning the ERA5-to-ENS difference and
  learning recent level shifts) and so improves the weekly fine-tune.
prediction: >-
  Written before any score is read. P1: L-3d beats L-Ww by 0.03 to 0.08 points of CRPS and L-1d beats
  L-3d by a further 0.02 to 0.06, at both groups, so L-1d ends 0.05 to 0.12 behind K2-S1 and no arm
  closes the gap completely. P2: L-Ww-30 beats L-Ww by under 0.05 points but its injected-shift leak
  is more than 1 point of capacity above L-Ww's. P3: L-mix beats L-Ww by less than 0.05 points
  (inside the minimum detectable effect).
decision_rule: >-
  See "Pre-registered rules". A more frequent refit, or a shorter half-life, or the mixed
  pretraining, is worth porting only if its paired contrast against L-Ww has a Bonferroni interval
  excluding zero and a point estimate of at most -0.05 at the six or at the eight, with q99
  pinball loss within +0.02, cold winter nights no worse than +0.05, and (for refit cadence and
  half-life) an injected-shift leak no more than 1 point of capacity above L-Ww's. The -0.05 bar
  sits below the minimum detectable effect (0.164 at the eight and 0.123 at the six, experiment
  510's two-seed values), so a pass reads "interval excludes zero" and never "at least 0.05".
benchmark: >-
  L-Ww (experiment 510, weekly, 90-day half-life, stored) for every contrast; K2-S1 and A-pool
  (experiments 470 and 425, stored) for the gap to the lagged network; BG2, climatology, and
  `manual_heuristic` for scale.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity, q01, q05, q95, and q99 coverage and pinball loss,
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the signed-MW six.
---

## Plan (status: planned; no score has been read)

**This plan was written before any fit.** The take-home message, the verdict, and the results are
added when the experiment finishes.

## Question

Experiment 510 found that refitting the lag-free network weekly rather than monthly scored 0.17 to
0.19 points of CRPS better, and that the gain grew through the month (zero in week 1, about 0.35 by
week 4). That pattern is a staleness effect. We ask whether refitting twice a week or every day
closes more of the remaining gap to K2-S1, whether a 30-day or 60-day half-life does, and whether a
pretraining that sees ENS rows as well as ERA5 and CAMS rows helps the weekly fine-tune. A fourth
question, how stale the ERA5 and CAMS pretraining may be, is optional (L-1d-oldPT, below).

## Arms

All arms are lag-free (37 non-lag inputs, 3 mixture components, widened 1.15 at scoring), restart
every fine-tune from the fold's pretrained weights (seeds 0 and 1), and train on ENS rows with valid
time at most the cutoff minus 3.5 hours.

| Arm | Refit cadence | Half-life | Seeds | Note |
|---|---|---|---|---|
| L-3d | twice weekly: days 0 and 3 of each 7-day block, 00:00 UTC (45 cutoffs over 5 folds) | 90 days | 0, 1 | |
| L-1d | daily, 00:00 UTC (154 cutoffs) | 90 days | 0, 1 | seed 1 is run unconditionally and reported alone as the replication |
| L-Ww-30, L-Ww-60 | weekly (25 cutoffs) | 30, 60 days | 0, 1 | take the slot experiment 510 reserved for a 30/180-day sweep |
| L-mix | weekly | 90 days | 0, 1 | mixed ERA5, CAMS, and ENS pretraining (below) |
| L-1d-warm | daily | 90 days | 0 | gated and report-only (below) |
| L-1d-oldPT | daily | 90 days | 0 | optional: only if L-1d beats L-Ww; not in the fit count |
| References, not refitted | | | | L-M, L-W, L-Ww (experiment 510); K2-S1, A-pool (experiments 470, 454, 425) |

**Scoring window.** Every model scores only the forecast origins from its own cutoff to the next
cutoff of the same cadence (one day for daily, 3 or 4 days for twice weekly, one week for weekly,
the whole month for monthly L-M). The windows tile the month, so each origin has one scoring model.
The `leak` command prints the check, before any fit, that the windows tile each month and that no
scored origin lies before its own cutoff.

**Reuse.** A fit whose cutoff day (0, 7, 14, 21, or 28) coincides with one of experiment 510's
weekly L-Ww fits uses 510's saved weights, because the rows, the half-life, the seed, and the
pretrained weights are identical. The window forecast is produced afresh. That saves 50 of L-3d's 90
fits and 50 of L-1d's 308.

**L-mix.** Pretrain on ERA5 and CAMS rows (source flag 0) and the fold's ENS rows (flag 1, rows from
2024-04) together, with each batch half from each source (4,096 + 4,096), so the fewer ENS rows are
not swamped. Arm A already has a lead input: the `lead` and `lead_sqrt` columns, and the ERA5 and
CAMS pseudo-runs of experiment 418 carry real leads of 0.5 to 351 hours. So the plan adds only the
source-flag column (a 38th input) and keeps the existing lead inputs for both sources. Early
stopping uses the mean of the held-out negative log-likelihood of the two sources. The fine-tune
then uses ENS rows only (flag 1), weekly, with the 90-day half-life. The flag-1 pretraining rows
must satisfy the same cut as the fine-tune rows (valid time at most the pretraining cut minus 3.5
hours, origin at most the cut); the `pretrain-mix` command prints and asserts the check before
fitting. If L-mix passes its rule, it is repeated daily (L-mix-1d).

**L-1d-warm (defined before running).** Continue from the previous day's weights for one epoch (20%
of the training rows, as in experiment 510's harness) at the fine-tune learning rate, with a fresh
optimiser each day and no early stopping. The first cutoff of a fold starts from the pretrained
weights and is the full L-1d fine-tune; nothing chains across folds. Run it only if L-1d beats L-Ww
(seed 0, report-only).

**L-1d-oldPT.** Optional; run only if L-1d beats L-Ww. Experiment 494's PretrainOnce arm already
tests stale pretraining. If run, it ships the old fold's standardiser and per-Primary target
statistics with the old weights.

## Fit counts and cost

- **New fine-tunes:** L-3d 40, L-1d 258, L-Ww-30 and L-Ww-60 100, injected-shift stress 80, L-mix
  50: 528 fine-tunes, plus 10 mixed pretrainings.
- **Timed cost:** a daily fine-tune takes 2.5 to 3 seconds of training (the first one 7.8 seconds)
  and 0.15 seconds of row selection, because each month's rows are built once and held on the GPU.
  About 1 GPU-hour in all, with the staleness inference. One GPU process under 8 GB.
- **Gated, not counted:** L-1d-warm up to 154 fits and L-mix-1d 154 per seed.

## Scoring and contrasts

- **Staleness curve.** Score each origin with the model refit `k` days earlier (k = 0 to 7), for
  every arm. This is inference only, and works for daily refits. Beside it, report experiment 510's
  L-M minus K2-S1 by week of the month (0.46, 0.37, 0.41, 0.37, 0.16), which does not grow with the
  week, so staleness is not isolated, and 510's L-Ww minus L-W of +0.07 in weeks 1 and 2 at the
  eight. Say which horizon bands a daily refit can affect: the newest daily rows reach only short
  leads.
- **Contrasts (paired, same seeds and pretrained weights).** L-3d, L-1d, L-Ww-30, and L-Ww-60
  against L-Ww (4); L-1d against L-3d (1); each of the four against K2-S1 and A-pool (8); L-mix
  against L-Ww (1). That is 14 contrasts, at the eight and the six, so 28 CRPS tests (56 with q99
  pinball loss). Bonferroni count: 28 for CRPS. Absolute scores are reported beside every contrast.
- **Minimum detectable effect.** Experiment 510's two-seed values: 0.164 at the eight and 0.123 at
  the six (from its `mde_k2.csv`).
- **Rows.** Rows with no labelled event (NGED's event labels are incomplete), days 1 to 10, the 12
  to 36 hour band, and the horizon bands day 1, days 2 to 3, and days 4 to 10.
- **Injected-shift leak.** Experiment 510's stress run (a 7-day shift of 3 times the standard
  deviation of a Primary's 7-day mean target, both signs) for L-1d and L-Ww-30 only, one seed, five
  folds (about 80 fits). For L-Ww-30 the block ends at the day-14 cutoff as in 510. For L-1d the
  block is fixed (days 7 to 14 of the month) and the daily refits at days 14 to 20 are compared with
  their uninjected fits, so the leak is reported by the age of the block (0 to 6 days). Descriptive:
  510's L-Ww leak ranges from 3.4 to 14.5 points across folds, so the 1-point limit cannot be tested
  and is reported as a trade-off beside the score gain of every arm.
- **Other slices.** Rows inside labelled events against NRA-corrected actuals (sensitivity),
  Christmas week 22 to 31 December 2025 and 1 to 14 January 2026 per arm (noise floor 0.26 at the
  six and 0.27 at the eight for one Christmas), cold winter nights, q01, q05, q95, q99 coverage and
  pinball loss, the widening sweep 1.0 to 1.15, avoidable-curtailment cost at f = 0.9, 0.75, 0.5 at
  the signed-MW six. D15 and D18 are report-only.
- **Production cost per arm.** Fits per day, seconds per fit, stored ensemble rows, pretraining
  cadence, and networks per serving pass (1 for every arm).

## Pre-registered rules

- **More frequent refit worth porting:** contrast against L-Ww with a Bonferroni interval excluding
  zero and a point estimate of at most -0.05 at the six or the eight, q99 pinball loss within +0.02,
  cold winter nights no worse than +0.05. The leak is reported as a trade-off (its 1-point limit
  cannot be tested).
- **Shorter half-life worth porting:** the same terms, against L-Ww.
- **L-mix adopted:** L-mix minus L-Ww with an interval excluding zero and a point estimate of at
  most -0.05 at either group, and q99 pinball within +0.02.
- **Stale pretraining acceptable (if L-1d-oldPT is run):** not shown worse than L-1d by 0.03.
- **No arm within 0.1 of K2-S1:** say so and state the distance; the choice between a simpler first
  release and the 4-network lagged mixture is the maintainer's.

## Standing practices followed

Cleaned data and the D15 and D18 cut before 2025-03-01 (through experiment 428's rows); stored
references reused (L-M, L-W, L-Ww, K2-S1, A-pool, BG2, climatology, `manual_heuristic`) on the same
rows; seed 0 and 1 averaged by quantile as in 510; July 2026 unused; no XGBoost work; widening 1.15.
Schedule: the refit cadences are the experiment's subject. Titles say "load-shift", never
"switching".

## Implementation details (deleted when this ships)

Code in `ml-experiments/00511-.../`: `e00511_common.py` (cadences, the month's rows on the GPU),
`e00511_fit.py` (`leak`, `time`, `pretrain-mix`, `fit`, `assemble`, `stale`). Experiment 510's
harness is imported; its weekly offsets are hard-coded, so this folder adds cadence-aware cutoffs,
windows, and a `build_month` that extends rows to the last daily cutoff.
