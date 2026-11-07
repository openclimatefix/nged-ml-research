# Calibration and uncertainty: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 483 | Does a widening or table P make the tails of the first-release demand candidates honest? | 1 | Experiment 470 | Planned |
| 492 | Can two constants per metering group replace table P for the first-release demand candidates? | 2 | Experiment 483 | Planned |
| 646 | Does running the demand networks once per ensemble member and pooling the 51 outputs beat giving them the ensemble mean and spread? | 2 | Nothing | Planned |
| 630 | Is the roster networks' upper demand tail too low on ordinary days, and does a bias correction fitted on earlier months remove the excess? | 1 | Nothing | In progress |
| 625 | How few member paths do NGED's power-flow studies need, and does a Schaake shuffle of anomaly paths help the Primaries? | 3 | Experiments 434 and 436 (finished) | In progress |
| 629 | Do coherent Primary scenarios, summed to a bulk supply point, give a trustworthy tail for the flow through that point? | 4 | Experiments 625 and 626 (finished) | In progress |
| 631 | Once the metered farms beneath a bulk supply point are in the sum, does a coherent scenario sum give a usable tail for the flow through that point? | 5 | Experiment 629 (finished) | In progress |
| [P14](#p14-coupling) | Which coupling step, ECC-Q or the Schaake shuffle, gives better coherence scores at short and long leads? | 6 | The scoring harness and experiment 00016 | Deferred; no consumer needs coherent members yet |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### P14: coupling

**The coupling step turns calibrated quantiles back into 51 coherent members.** The arms are ECC-Q
with the interpolation, tail, and tie rules of the [staged forecast roadmap
page](https://openclimatefix.github.io/nged-substation-forecast/roadmap/staged-probabilistic-forecast/),
and the Schaake shuffle, scored by the
variogram score and the CRPS of summed solar farms in a weather cell, by lead band.
