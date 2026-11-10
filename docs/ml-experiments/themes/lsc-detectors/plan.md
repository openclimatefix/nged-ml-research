# LSC detectors: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [SW2](#sw2-switching-event-detection) | Which change to the switching detector raises the share of moved MW found at an acceptable alarm rate? | Not set | Nothing | Iterations SW2-5 to SW2-9 are done (experiments 68, 61, 60, 55, and 62); further iterations are proposals; experiment 181 found that a block detector finds 0 of 17 unlabelled block days at RO-NC's alarm budget, and the candidate label table awaits NGED review |
| 672 | Does a graph neural network find the donor of an injected load transfer better than a search at connected Primaries? | High (maintainer approved 2026-11-09) | Nothing | Finished (not supported): the graph network pairs 0.08 against E1-J's 0.29 in the long and deep cell; the graph line is closed at 15 Primaries |
| 675 | Why does the search of experiment 670 leave most recipient megawatts unpaired, and does a pretrained encoder find donors and recipients better than the event table's features? | High (maintainer asked 2026-11-09; reopens the graph line closed by experiment 672) | Nothing | Finished (not supported): the donor end holds 0.87 of E1-J's found-but-unpaired recipient megawatts, and masked pretraining finds 0.28 of long and deep donor megawatts against 0.59 for a random-init encoder |
| 678 | Does feeding a probe's donor alarms into E1-J's recipient search pair more injected transfer megawatts than E1-J alone? | High (graph Discuss session 2026-11-10) | Nothing | Finished (not supported): A donor probe feeding E1-J's recipient search pairs a little more than E1-J (0.36 against 0.29 of long and deep megawatts), not the +0.10 that was predicted, and far below the 0.60 that would have made it the event table. See [experiment 678](../../00678-donor-posterior-hybrid-for-load-transfer-pairing/result.md). |
| 690 | Does the slow-level state track the July to August level change at Primaries 13 and 16? | High (maintainer approved 2026-11-10) | Nothing | Finished (partly supported): the masked slow level follows the change and halves the median absolute error at both Primaries in hindsight; two control Primaries moved by more than 5% |
| 309 | Can the gap between a no-lag forecast and a forecast with lags tell NGED which one to use, without knowing the truth? | High (maintainer approved 2026-10-23) | Nothing | Finished (not supported): NGED cannot pick between the forecast with lags and the forecast without lags from the gap between them alone. See [experiment 309](../../00309-lsc-two-forecast-gap-choosing-without-truth/result.md). |
| 186 | Are D8's multi-day drops an embedded gas peaker running (spark spread, triads, flat blocks)? | High (NGED answer relayed 2026-10-22) | Nothing | Finished (partly supported): The evidence fits a gas engine below D8 that runs when electricity is dear relative to gas, but does not prove it. See [experiment 186](../../00186-switching-d8-gas-peaker-spark-spread-and-triads/result.md). |
| [653](../../00653-edge-state-event-table-penalised-segmentation/result.md) | Can a penalised segmentation of the Primaries' residuals build a table of load-shift events with partners, start times, and end times? | Done | Nothing | Finished (not supported): At 15 Primary substations, a penalised segmentation of the demand residuals finds about two thirds of the injected load-shift megawatts, but it did not say which neighbouring Primary the power went to: only 18% of injected transfer ends were paired correctly, and on the real record a shuffled neighbour graph gives half as many pairs as the real graph (2 against 4, too few to be sure). |
| [657](../../00657-event-table-with-explicit-solar-term-against-plain-arm/result.md) | Does subtracting an explicit solar term from the Primaries' residuals reduce false events at sunny midday without losing injected events? | Done | Nothing | Finished (not supported): Subtracting an estimate of the unmetered solar output from the demand forecast's errors changed nothing, because the forecast already allows for the sunshine at these six substations. |
| [658](../../00658-edge-state-event-table-neural-network/result.md) | Does a neural network that reads all 15 Primaries' demand residuals at once find and pair load-shift events better than the penalised segmentation of experiment 653? | Done | Nothing | Finished (not supported): A neural network that reads all 15 Primaries' demand residuals together did not find or pair injected load-shift events better than the penalised segmentation of experiment 653: it found 0.48 to 0.52 of the injected megawatts against 0.64, and paired 0.16 to 0.17 of injected transfers against 0.18. |
| [660](../../00660-shared-generator-of-injected-load-transfer-events/result.md) | Can one shared generator of injected load-transfer events look real enough to train and test detectors on? | Done | Nothing | Finished (supported): We built one shared tool that adds fake load transfers between neighbouring Primary substations to real power history. |
| [665](../../00665-event-table-recall-by-length-and-depth/result.md) | How much of an injected load-shift event does each event-table method find, split by how long the event lasts and how deep the step is? | Done | Nothing | Finished (partly supported): Where an injected load shift lasts more than 7 days and steps 8 noise units or more, the event table of experiment 653 finds 0.67 of the injected megawatts and the neural network of experiment 658 finds 0.65. |
| [676](../../00676-pretrained-graph-model-for-load-transfers-part-2-edge-state-network/result.md) | Does a neural network with an explicit state for every link between two Primaries, and a recipient lag, pair more injected load-transfer megawatts than the donor-probe hybrids of experiment 678? | Done | Nothing | Finished (not supported): A neural network that learns the state of every link between two Primaries, and the delay before the recipient follows, pairs no more transfer megawatts than the donor-probe hybrids or the search of experiment 670 (paired share of long, deep transfer megawatts: 0.32 for the new network, 0.29 for the search, 0.35 and 0.36 for the hybrids). |
| [680](../../00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/result.md) | Does a search that also starts from each unpaired rise, and looks for the matching fall at a neighbouring Primary, pair more injected load-transfer megawatts than the one-way search of experiment 670? | Done | Nothing | Finished (partly supported): Searching from each unpaired rise for the matching fall at a neighbouring Primary pairs 0.38 of the injected transfer megawatts in the long and deep cell, against 0.29 for the one-way search, a gain of 0.09. |
| [682](../../00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model/result.md) | Does choosing edge flows by a fixed cost per change pair more injected load-transfer megawatts than the searches of experiments 670 and 680, without adding more false transfers? | Done | Nothing | Finished (not supported): Explaining pairs of steps by one flow on an edge, chosen by a fixed cost per change, does not find the donor and recipient of an injected load transfer better than the simple searches: it pairs 0.33 of the long and deep megawatts against 0.29 for the one-way search and 0.38 for the two-way search. |
| [685](../../00685-event-table-on-the-final-test-window-all-primaries/result.md) | Which load-shift candidates does the fixed event table find at the 15 Primaries from 2026-07-01 to 2026-10-31? | Done | Nothing | Finished (inconclusive): From 2026-07-01 to 2026-10-31 the event table finds 13 candidate events (14 rows) at the 15 Primaries and no transfer between time_series_id 13 and 16 in July or August, nothing at id 11 in September, and one open fall at id 18 from 2026-10-04. |
| [686](../../00686-event-table-transfer-timing-and-false-solar-growth-events/result.md) | Do the donor and the recipient of a load transfer start together, and what kind of false event does solar growth cause in the event table? | Done | Nothing | Finished (partly supported): The two verified transfers give no sign of a lag between the substation that lost demand and the one that gained it: the D3/D22 steps coincide within noise, and the D5/D16 steps are 2.5 hours apart at the end and too noisy at the onset to say. |

**Experiment 00116 (differentiable physics, NRA) also bears on this theme.** It fits flows on the
interconnection edges between Primary substations and scores them on injected transfers; see the
[Network research plan](../differentiable-physics-network/plan.md).

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### SW2: switching-event detection

**The SW2 line improves the switching-event detector first, scored by event-level F1 against a
labelled set of visible events, and leaves the use of the detector until the detector works.** The
detector serves two purposes later. The first is a lower forecast error, if switching periods act as
unexplained noise for the XGBoost model and corrupt its lagged power inputs. The second, and the
more important to NGED, is a forecast under the normal running arrangement (NRA), into which NGED
can add its own switching. The [switching
events](https://openclimatefix.github.io/nged-substation-forecast/roadmap/switching-events/) roadmap
page
proposes building that forecast by excluding switching periods from the training target. An event
table for NGED is a third purpose, at lower priority. The production detector has to run on power
alone, because the switching logs exist only for the trial area, so the logs are used here for
evaluation only.

**The line reopens a question that experiment 00038 closed, because experiment 00038 tested a
different use of the detector.** Experiment 00038 masked detected switching from the history of the
error correction that runs after the XGBoost model, and the mask made the forecast worse. No
experiment has yet masked switching from an XGBoost training target or from a lagged input.
Experiment 00037 confirmed 37 of 612 logged closures of 2 hours to 60 days as visible in its own
residual, and scored its detectors on the 18 of them in the validation months. That reference set is
small, about 6 of the 37 are expected by chance, and the label was built from the residual the
detectors use, so a detector that shares the residual's blind spots is flattered by the label. The
line therefore starts by building a reference set independent of the detectors.

The full labeller design is in the private file `data/ml-experiments/briefs/sw2_labeller_design.md`.

#### SW2-4 to SW2-9: detector iterations

**Each iteration tests one change to the detector against the [experiment
43](../../00043-sw2-1-detector-scoreboard/result.md) scoreboard, tuned on injections and odd months,
scored on even months, with recall weighted by the MW moved as the headline from SW2-7 onwards.**
The order below is a proposal, by expected gain per hour of work, for the plan of each iteration to
confirm:

- **SW2-5 (done, [experiment 68](../../00068-sw2-5-train-detect-retrain-loop/result.md)):** three
  rounds of refitting the lagless XGBoost model without the detected intervals changed the share of
  the moved MW found by +0.011 and the forecast error by -0.175%, and a random mask did as well.
- **SW2-6 (done, [experiment 61](../../00061-sw2-6-paired-step-intervals/result.md)):** pairing the
  onset and return steps into one alarm score finds less than D0M at the same alarm rate, but taking
  the end of D0M's interval from the strongest opposite step in the next 14 days finds 0.19 more of
  the moved MW.
- **SW2-7 (done, [experiment 60](../../00060-sw2-7-profile-weighted-step-detector/result.md)):** a
  step statistic that weights each half-hour by the expected power found the same transfers as the
  constant step on five injection shapes, so the step's shape does not limit the 24-hour detector.
  About a third of the transfers with a donor step of 6 noise units or more have the right onset and
  the wrong end, which SW2-6 (paired-step intervals) addresses.
- **SW2-8 (done, [experiment
  55](../../00055-sw2-8-neighbour-veto-single-series-detector/result.md)):** a single-series
  detector first, with neighbour balance as a sense check, not a gate. Every single-series alarm
  stands. A metered neighbour's opposite step raises the alarm's confidence and names the receiver,
  a same-sign step on the neighbours vetoes the alarm as a common-mode weather or calendar miss, and
  an alarm with no metered neighbour keeps its single-series confidence. Experiment 00043 found the
  neighbour-balance detector D1, which uses the neighbour test as a gate, finding almost none of the
  injected transfers without a metered receiver. The sum of the neighbourhood's residuals and a
  subset-sum over neighbours for a split feed the confidence score.
- **SW2-9 (done, [experiment 62](../../00062-sw2-9-whitened-residual-step-detector/result.md)):**
  whitening, experiment 00037's proposed F4, an autoregressive model of order 1 per series and hour
  of day fitted to the residual. The filtered detector found less of the moved MW (-0.097 at 24
  hours), because the filter shrinks a day-long step faster than the noise of the 24-hour step.
