# Calibration and uncertainty: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [368](../../00368-d8-conditional-band-head-on-gam-stack-residuals/result.md) | Does one band head on the GAM stack's residuals, pooled across Primaries, improve D8's uncertainty without hurting other Primaries? | 1 | Nothing | Planned |
| [P14](#p14-coupling) | Which coupling step, ECC-Q or the Schaake shuffle, gives better coherence scores at short and long leads? | 5 | The scoring harness and experiment 00016 | Deferred; no consumer needs coherent members yet |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### P14: coupling

**The coupling step turns calibrated quantiles back into 51 coherent members.** The arms are ECC-Q
with the interpolation, tail, and tie rules of the [staged forecast roadmap
page](https://openclimatefix.github.io/nged-substation-forecast/roadmap/staged-probabilistic-forecast/),
and the Schaake shuffle, scored by the
variogram score and the CRPS of summed solar farms in a weather cell, by lead band.
