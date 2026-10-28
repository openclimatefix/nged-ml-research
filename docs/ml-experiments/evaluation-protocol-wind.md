# Evaluation protocol for wind farms, version 1

**This page fixes how every experiment on wind farms is scored.** It is provisional: it copies the
demand protocol's structure and changes only what a wind farm needs. An experiment records
`protocol: wind-v1` in its front matter. A change to any metric, fold, series list, slice, interval,
or tolerance makes a new version, and the superseded text moves to `evaluation-protocol-archive/` as
`wind-v1.md`. The wind protocol is due a fuller review in a separate session.

| Version | Date | Change |
|---|---|---|
| `wind-v1` | 2026-10-28 | First version, provisional |

## What is scored

- **Series:** wind farms W1 to W3 and the public wind farms of experiment 288, each on its own row.
  NGED's metered generators are never named, and a farm's time series is never published with its
  name or identifier. Public wind farms may be named.
- **Rows:** `fold_a_tuesdays_v1`, 53 Tuesday runs. The Tuesday row set stays for wind until a later
  version moves wind to contiguous-month folds. The demand protocol's change of folds does not
  retire the Tuesday set.
- **Lead range:** 0.5 hours to 15 days is scored, with true days 1 to 10 reported beside it. Days 1
  to 10 is the headline band.
- **Forecasts compared:** the forecast under test, `climatology`, and the XGBoost forecast without the
  offset table. `manual_heuristic` and BG2 are not scored for wind farms.

## Slices

**Each metric is reported over the slices below, and every slice row shows its half-hour count.**
The demand-only slices do not apply.

| Kind | Slices |
|---|---|
| Lead time | Day 1, day 2, days 3 to 4, days 5 to 7, days 8 to 10, days 11 to 15 |
| Condition | All half-hours; by month; by wind-speed band |
| Tail | Pinball loss and coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles per farm; the upper tail is reported with the head's calibration |
| Series | Every farm on its own row |

**No MVA or signed-MW split applies.** The farms report megawatts.

## Metrics, reporting, and adoption

**The metrics, interval method, reporting rules, adoption conditions, and known-answer tests are
those of [`demand-v1`](evaluation-protocol-demand.md)**, with the changes this page states. They are
repeated by reference and not copied, so a future change to the demand wording does not silently
change this protocol. Where the two disagree, this page governs wind. The tolerances start at the
demand values (0.02 points for the headline, 0.05 for the tails and other slices) and are tweaked
only by a new version of this protocol.

**A family with few series and few runs will rarely pass the gain test, and the rule does not lower
the bar for it.** A change that cannot reach the bar stays a candidate, and the maintainer may judge
it by a descriptive report instead.

## What this page does not cover

- A physical lock that stops research sessions reading the held-out actuals.
- Logging scores to MLflow, which a follow-on pull request adds.
