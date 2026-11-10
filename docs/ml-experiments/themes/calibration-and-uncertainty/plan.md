# Calibration and uncertainty: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 483 | Does a widening or table P make the tails of the first-release demand candidates honest? | 1 | Experiment 470 | Planned |
| 492 | Can two constants per metering group replace table P for the first-release demand candidates? | 2 | Experiment 483 | Planned |
| 630 | Is the roster networks' upper demand tail too low on ordinary days, and does a bias correction fitted on earlier months remove the excess? | 1 | Nothing | Finished (`partly supported`) |
| 625 | How few member paths do NGED's power-flow studies need, and does a Schaake shuffle of anomaly paths help the Primaries? | 3 | Experiments 434 and 436 (finished) | Finished (`partly supported`) |
| 629 | Do coherent Primary scenarios, summed to a bulk supply point, give a trustworthy tail for the flow through that point? | 4 | Experiments 625 and 626 (finished) | Finished (`not supported`) |
| 631 | Once the metered farms beneath a bulk supply point are in the sum, does a coherent scenario sum give a usable tail for the flow through that point? | 5 | Experiment 629 (finished) | Finished (`not supported`) |
| [P14](#p14-coupling) | Which coupling step, ECC-Q or the Schaake shuffle, gives better coherence scores at short and long leads? | 6 | The scoring harness and experiment 00016 | Answered for the farms and the Primaries by experiments 434, 436, and 625; open for coherence with the direct forecast of the bulk supply point's meter (see the [delivery recommendation](delivery-recommendation.md)) |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### P14: coupling

**The coupling step turns calibrated quantiles back into 51 coherent members.** The arms are ECC-Q
with the interpolation, tail, and tie rules of the [staged forecast roadmap
page](https://openclimatefix.github.io/nged-substation-forecast/roadmap/staged-probabilistic-forecast/),
and the Schaake shuffle, scored by the
variogram score and the CRPS of summed solar farms in a weather cell, by lead band. Experiments
[434](../../00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts/result.md),
[436](../../00436-joint-scenarios-across-demand-wind-solar-and-leads/result.md), and
[625](../../00625-coherent-members-from-quantiles-k-sweep-and-path-checks/result.md) have since
compared the couplings: a Schaake shuffle of past observed paths scored 34% below independent draws
across the wind and solar farms, and no coupling beat independent draws across the 16 Primaries.
[Experiments 629 and
631](../../00631-bsp-group-sum-with-metered-generators-against-direct-bsp-forecast/result.md) found
that summed node scenarios did not give a usable tail for the flow through a bulk supply point. The
[delivery recommendation](delivery-recommendation.md) states the current position.
