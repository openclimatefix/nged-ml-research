# Calibration and uncertainty: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 483 | Does a widening or table P make the tails of the first-release demand candidates honest? | 1 | Experiment 470 | Finished (partly supported): The first-release demand candidates are too wide on the lower tail and too narrow on the upper tail, so a widening of both tails makes the lower tail worse while it repairs the upper one. See [experiment 483](../../00483-uncertainty-rerun-on-the-first-release-demand-candidates/result.md). |
| 492 | Can two constants per metering group replace table P for the first-release demand candidates? | 2 | Experiment 483 | Finished (partly supported): At the eight MVA-only Primaries, one lower constant and one upper constant per metering group, fitted on earlier months, recalibrate the demand forecast as well as table P (20 factors per metering group) does, so table P can be dropped there. See [experiment 492](../../00492-two-constant-tail-widening-in-place-of-table-p-for-demand/result.md). |
| 630 | Is the roster networks' upper demand tail too low on ordinary days, and does a bias correction fitted on earlier months remove the excess? | 1 | Nothing | Finished (`partly supported`) |
| 625 | How few member paths do NGED's power-flow studies need, and does a Schaake shuffle of anomaly paths help the Primaries? | 3 | Experiments 434 and 436 (finished) | Finished (`partly supported`) |
| 629 | Do coherent Primary scenarios, summed to a bulk supply point, give a trustworthy tail for the flow through that point? | 4 | Experiments 625 and 626 (finished) | Finished (`not supported`) |
| 631 | Once the metered farms beneath a bulk supply point are in the sum, does a coherent scenario sum give a usable tail for the flow through that point? | 5 | Experiment 629 (finished) | Finished (`not supported`) |
| 689 | Does a width scale fitted on a fixed lookback remove the steps in band width at the model refits, or does seed averaging matter? | 2 | Experiment 683 (finished) | Finished |
| [P14](#p14-coupling) | Which coupling step, ECC-Q or the Schaake shuffle, gives better coherence scores at short and long leads? | 6 | The scoring harness and experiment 00016 | Answered for the farms and the Primaries by experiments 434, 436, and 625; open for coherence with the direct forecast of the bulk supply point's meter (see the [delivery recommendation](delivery-recommendation.md)) |
| [627](../../00627-scenario-subset-selection-from-51-coherent-paths/result.md) | Which 3, 5, or 10 of the 51 coherent member paths should NGED's power-flow studies use, with no label attached to any path? | Done | Nothing | Finished (partly supported): Picking 5 of the 51 forecast paths by cutting them into 5 equal groups on one summary and taking one path from each covers their spread better than picking 5 at random, in every month, but still misses the tails: the stratified rules leave a gap of 11% to 12% of the range at the 5th percentile and 13% to 14% at the 95th. |
| [628](../../00628-quantile-form-client-side-scenario-rehydration/result.md) | Can NGED rebuild coherent scenarios from stored quantiles alone, and which tail rule should the client use? | Done | Nothing | Finished (partly supported): NGED can rebuild the marginal forecasts and a Schaake-shuffle coupling from 11 stored quantiles per site and half-hour plus a small dependence table. 1,000 client-drawn scenarios score within 0.3% of the 51 scenarios the pipeline builds on CRPS, and a Schaake shuffle of 51 stored past paths matches the pipeline's shuffle on the variogram score. |
| [642](../../00642-roster-upper-tail-correction-out-of-sample-leave-one-month-out/result.md) | Can a correction fitted on other months bring the upper demand tail of K1 and the Candidate within 1.5 percentage points of nominal without raising CRPS by more than 1%? | Done | Nothing | Finished (partly supported): Shifting each stored demand forecast to the level of the previous 28 days, and then widening its upper range, brings the share of readings above the forecast 90th percentile (q90) to between 9% and 12.5% in every test month at 14 of the 16 primary substations, against 10% to 16% before, but it makes the continuous ranked probability score (CRPS, the score of the whole forecast distribution) about 2% worse. |
| [646](../../00646-per-ens-member-neural-demand-networks-pooled-quantiles/result.md) | Does running the demand networks once per ensemble member and pooling the 51 outputs beat giving them the ensemble mean and spread? | Done | Nothing | Finished (not supported): For K1 and K3 on the five development months, giving the demand networks the mean and spread of the 51 ensemble members matched running them once per member and pooling the 51 answers, once the bands had equal width. |
| [688](../../00688-winter-normal-running-coverage-median-error-and-events-part-1/result.md) | Do the 90% bands of the normal-running Primaries miss in winter because of median error and load transfers, or because the bands are too narrow? | Done | Nothing | Finished (supported): The 90% bands of the 16 Primary substations' normal-running forecasts miss in winter mostly because the median forecast is off by a different amount each month, and partly because of load transfers, not because the bands are too narrow. |

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
