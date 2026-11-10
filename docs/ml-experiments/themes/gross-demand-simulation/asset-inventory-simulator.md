# An asset-inventory simulator

**An asset-inventory simulator is worth one small test on data whose answer is known, as a source of
an asset list per substation and of scenarios, mostly at long horizons, and not as a better
short-range forecast.** The simulator would fit, for each Primary substation (a substation that
steps 33 kV down to 11 kV), a list of the assets behind its meter and a few parameters per asset.
Examples are a solar capacity, the size of each unit of a gas engine and how long the engine usually
runs, and a battery's power. The simulator would then run in the forward direction, from assets to
power, to give scenarios. The forward direction needs a model with real operating logic for each
asset type, described under [What the forward direction means](#what-the-forward-direction-means).
Fitting a fixed list, rather than explaining every half-hour, drops the step that
failed in the gas-engine experiments read for this page: deciding when each asset is on.

**The maintainer's answers make the asset list a deliverable and long-horizon scenarios the main
use.** NGED hold nothing beyond the Embedded Capacity Register about what sits under each
substation, so no ground truth exists to check the list against, and the list is shown as inferred. See
[Maintainer's answers (2026-11-10)](#maintainers-answers-2026-11-10).

**Two conditions, drawn from the merged experiments, decide whether the idea can work.** The asset
list has to start from NGED's Embedded Capacity Register, because in the four experiments read for
this page that let a fit decide on its own whether a generator exists, the fit reported generators
that are absent or unconfirmed. And the output has to be judged on known-answer tests, because
the half-hourly power the meter records (demand minus the generation below it) cannot separate
several assets of one type.

**The smallest test would add real metered generator output to the readings of the six Primaries
metered in megawatts (MW), and give the fit an asset list that includes one asset known to be
absent.** The other Primaries are metered in apparent power (MVA). The test is described under [The
smallest test](#the-smallest-test) and is an option for the maintainer, who has not yet chosen it;
nothing has been started.
The test would show cheaply whether the fit recovers the capacities and running-time parameters of
assets it is told about, and whether it gives near-zero capacity to a listed asset that is absent.

**This page answers from merged pages and stored files, using data before 2026-07-01 only.** The
discussion that led to it is recorded on [The simulator discussion of
2026-11-09](simulator-discussion-2026-11-09.md). The register counts below come from
`scripts/register_coverage.py`, beside this page, run on one core.

## Terms used on this page

- **Primary:** a primary substation. The 16 Primaries in the study area are D2 to D5, D8 to D12, D14
  to D16, D18, D19, D21, and D22, the anonymised labels of the
  [glossary](../../glossary.md#d-codes).
- **Net series:** the half-hourly power a Primary's meter records, which is demand minus the output
  of the generators connected below it.
- **Strict disaggregation:** splitting every half-hour of the net series into the contribution of
  each asset, for example dating each start and stop of a gas engine.
- **Asset inventory:** a list of the assets behind one Primary, with a few fitted parameters per
  asset and no half-hour-by-half-hour state.
- **Register:** NGED's Embedded Capacity Register (ECR), August 2026 release, which lists connected
  and accepted generation and storage of 50 kW and above. It is read by `studies.ecr_reader`.
- **Register name:** the text in the register's Primary column. The 15 register names that match the
  study area cover the 16 Primaries, because Wrangle T1 (D15) and Wrangle T2 (D18) share one.
- **Slot:** one technology at one registered site; a site can hold up to three.
- **Known-answer test:** a test that adds a known asset's output to a real reading, so a fit can be
  scored against the truth.
- **Noise unit:** a Primary's usual difference between successive 6-hour means, as defined in
  [experiment 653](../../00653-edge-state-event-table-penalised-segmentation/result.md).
- **Recovery:** the share of a known, injected asset's capacity that a fit finds again; 1 is all of
  it.
- **NGED's metered generators** are anonymised as Wind A to C, Solar A to F, Battery A, and Gas A.

## Maintainer's answers (2026-11-10)

- **The asset list is itself a deliverable.** NGED do not know exactly what sits under each
  substation, and they have nothing beyond the register, so no ground truth exists. A confirmed
  asset list per substation is one of the main benefits, together with a forward simulator of
  demand.
- **NGED cannot confirm the lists beyond the register.** The known-answer test is therefore the main
  validation, and each list is shown as inferred.
- **Scenarios matter much more at long horizons.** They are also interesting at days 1 to 10 for
  what-if questions. One example: a generator known to exist that has been off for months turns on
  during the 10-day horizon, and the question is what happens to the substation's demand.
- **A forward model means real operating logic.** A battery model implements what a battery
  scheduler would do given the available price signals and the state of charge, with several
  scheduling profiles, because there are several ways to optimise. The same holds for other
  price-driven generators. A gas generator is a physical model written as equations that capture its
  ramping time, its minimum on-duration, and similar limits.
- **Training the forecaster on simulated series is conceivable if the simulator becomes good
  enough.** The maintainer calls training on simulated series "a bit sketchy".
- **Not yet answered:** whether the decoy rule should apply to solar too (see [Open
  question](#open-question-for-the-maintainer)).

## What the forward direction means

**The forward direction needs, for each asset type, a model that decides what the asset does from the
signals it would see.** A fitted profile of the asset's past output is not enough. The table says
which parts of the research repository (`nged-ml-research`) already hold such a model.

| Asset type | Forward model wanted | In the repository today |
|---|---|---|
| Battery | A scheduler with the price signals available and the state of charge, in several profiles | Partly. `studies.battery_dispatch` holds a daily linear-programme schedule for a price taker, with a state of charge, an energy limit, a cycle cap, and a rank-rule schedule; `studies.battery_templates` holds tariff-window, charge-only, merchant, Agile-tariff, and home-battery schedules; `studies.battery_state_space` holds the differentiable estimator that mixes them. The estimator fits the schedules to substation data. We found no module that runs the schedules forward for a registered battery |
| Other price-driven generator | An optimiser that follows the price signal within the plant's limits | No. We found none in `src/studies` or in the experiments read for this page |
| Gas engine | Plant equations with ramping time, minimum on-duration, and similar limits | No. The engine detector is a hidden Markov model of whole-number units and spell lengths, fitted to the residual (experiments 196 and 291), and has no ramping or minimum-on-duration equation |
| Solar | Output from sunshine and panel geometry | Yes: the physics solar plant curve (experiment 282), with the assumed tilt fitting worse than the farms' best |
| Wind | Output from each ensemble member | Yes: the physics wind plant run on each ensemble member, followed by a mixture-density head (experiment 347) |
| Demand under the assets | Calendar, temperature, heating, holidays, and the night step | Yes: the demand GAM (`studies.dp_gam`) |

**The demand GAM, the solar and wind plants, and the battery schedules exist. The gas-engine
equations and the other price-driven generators do not.** The only recorded gas-engine output
we read for this page is Gas A's, which covers 44 days.

## Strict and relaxed disaggregation identify different quantities

**Strict disaggregation asks the net series for the state of every asset at every half-hour, and the
gas-engine experiments did not get that state reliably.** The relaxed formulation asks for a
per-asset capacity and a distribution of running times. Both quantities are averages over many
events, so each event only has to be found often enough to estimate the average.

**The D8 gas-engine line is the closest precedent for the relaxed formulation, and it shows both
what works and what does not.** The best-supported account of D8 is that a 6 MW gas engine below it
stopped being added back into D8's reading on 2023-08-15. After that date D8's power falls well
below what a fitted model expects whenever the engine runs ([experiment
194](../../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md)).
NGED has not confirmed this account. A hidden Markov model of up to three units of about 1.75 MW
detects when the engine is running, when the engine's recorded output is added to D8's reading, but
raises too many false alarms at three of six other Primaries ([experiment
196](../../00196-ps2-quantised-engine-detector-on-d8-with-known-answer-gate/result.md)). The
detector reports the engine as running during 27 of the 28 multi-day drops at D8 after 2023-08-15
([experiment 291](../../00291-d8-drops-engine-spells-load-shift/result.md)), but the drops and the
detector are both computed from the same residual.

**Knowing the engine's state helped the forecast only when the state was read at the half-hour being
forecast.** With the state of the last settled day, the engine term made the forecast 10% worse
([experiment 294](../../00294-regime-aware-model-at-d8-engine-state-term/result.md)). Experiment 294
did not try to forecast the state, so whether the state can be forecast is open. The relaxed
formulation sidesteps the question by modelling how often and how long the engine runs. The unit
size is not yet estimated well: experiment 294's fitted unit is 1.4 to 1.5 MW against a recorded
1.97 MW.

### What the net series can identify

**The net series identifies large, long, or slowly changing quantities.**

- **The total solar capacity behind a Primary, at season-year resolution.** The capacity fit finds a
  median 0.78 to 0.86 of a real solar farm's output added to a Primary ([experiment
  129](../../00129-differentiable-physics-pv-capacity-known-answer-test/result.md)). The
  neural-network probe reads 2.4 to 3.9 MW of unmetered solar at each of the six MW Primaries, after
  dividing by the share of an injected real farm that the probe finds ([experiment
  649](../../00649-pv-capacity-table-season-year-from-stored-probes/result.md)).
- **A step in solar capacity, when the step is large and the capacity is free to change each
  month.** A known step was found to within 2% of its size and within one month of its date
  ([experiment 207](../../00207-dp-meter-regime-steps-e2-step-capacity-known-answer/result.md)).
- **A recorded engine's running half-hours, when the engine's output is injected** and its units are
  large against the noise (experiment 196). The same detector does not give the engine's state or
  unit size reliably in the real series (experiments 196, 199, and 294).
- **Long, deep steps of any cause.** For injected load shifts that last more than 7 days and step 8
  noise units or more, the event table finds 0.67 of the injected megawatts ([experiment
  665](../../00665-event-table-recall-by-length-and-depth/result.md)).

### What the net series cannot identify

**The net series does not identify small, short, or numerous assets, or the cause of a step.**

- **Several assets of one type behind one meter.** The net series carries their sum, and the
  solar-capacity methods above estimate one total. The register lists 77 connected solar slots
  across the 15 register names, with two or more at 12 of them, and solar below 50 kW is not listed
  at all. A per-site split would have to come from the register, not from the data.
- **Assets that never make a visible step.** Real batteries whose output was added to NGED's metered
  power were mostly not detected or sized, a provisional finding, and the estimator reports about
  2.6% of peak flow where no battery exists ([the unmetered battery capacity
  study](../../../studies/unmetered-battery-capacity.md)). Solar growth makes no clean step either;
  the event table reads solar growth as false events, half of which last 3 days or less ([experiment
  686](../../00686-event-table-transfer-timing-and-false-solar-growth-events/result.md)).
- **Short events.** For load shifts of a day or less, the event table finds 0.16 to 0.26 of the
  injected megawatts (experiment 665). Gas A ran in 0.170 of half-hours, about 4 hours a day, in
  spells with a median of 2 hours, over its 44 recorded days (experiment 194). Spells that short sit
  in that class unless the units are large against the noise.
- **The cause of a step.** A load transfer, a generator starting, and a change in demand all look
  like a step in the net series. At D8 the drops and the engine detector are both computed from the
  same residual, so experiment 291 could not call the engine proved. D8's night load also stepped
  after March 2025 for a reason no tested explanation covers ([experiment
  199](../../00199-ps2b-engine-detector-v2-ramp-night-price/result.md)).

### Fits left free to add generators report generators that are absent or unconfirmed

**In the four experiments read for this page that let a fit decide whether a generator exists, the
fit reported generators that are absent or that could not be confirmed.** A generator-finding
component reported a generator at every substation tested, and was switched off ([experiment
110](../../00110-differentiable-physics-5-embedded-generation-and-capacity/result.md)). The engine
mixture of [experiment 264](../../00264-dp-gas-engine-asymmetric-bands-at-d8/result.md) fitted a
running engine at two substations that have none. In a year with no solar steps, a monthly solar
capacity invented steps above 0.5 MW at 2 of 5 Primaries (experiment 207). A scan of the real series
found about as many solar steps as the same scan invents where none exists, so none of the steps
could be called real ([experiment
215](../../00215-dp-pv-step-scan-real-series-register-match/result.md)). The relaxed formulation is
likely to avoid this failure only if the list of asset types per Primary comes from outside the net
series.

## What the register can supply

**The register gives, for each connected asset of 50 kW and above, its technology, its registered
MW, its connection date, its connection voltage, and its register name.** The register also carries
storage MWh and duration, a combined heat and power flag, a flexible-connection flag, export and
import limits, and, for assets accepted but not yet connected, a target energisation date. The
register does not carry running hours, a dispatch rule, or an output profile, so the running-time
parameters of a relaxed fit must come from the data.

**At the 16 Primaries the register lists 98 connected slots, most of them small solar.** Of the 98
slots, 93 (57.0 MW) connect at 11 kV or below, behind the Primary's meter, and 5 (0.43 MW) give no
voltage. Of the 98 slots, 97 carry a connection date.

| Asset class | Slots | Registered MW | Register names with one or more |
|---|---|---|---|
| Solar | 77 | 33.2 | 15 (all) |
| Bioenergy (biomass, biogas, landfill gas) | 8 | 11.1 | 6 |
| Gas or oil engine | 2 | 7.0 | 2 (D14 6.0 MW, D5 1.0 MW) |
| Type not given | 7 | 4.3 | 6 |
| Wind | 1 | 1.2 | 1 (D9) |
| Storage | 3 | 0.65 | 3 |

**The register also lists 16 slots (3.7 MW) accepted to connect at these Primaries:** 13 solar (3.0
MW), 2 storage (0.4 MW), and 1 wind (0.28 MW). Those slots are the new connections a planning
scenario could add before they appear in the net series.

**Three limits make the register a starting estimate that the fit may overrule, and not the truth.**

- **The register omits assets below 50 kW, and the other registers do not fill the gap.** At 8 of
  the 12 Primaries with a probe interval, the probe's interval lies above the solar capacity that
  three registers list together, the Feed-in Tariff report and the Renewable Energy Planning
  Database included ([experiment
  673](../../00673-pv-capacity-for-nged-per-primary-with-register-cross-check/result.md)). The
  registers together therefore miss part of the solar capacity. The Feed-in Tariff report, which
  does list small sites, ends in March 2025.
- **The register name and the meter that records an asset can differ.** The 6 MW gas engine that
  appears in D8's readings is registered under D14, and experiment 194 warns against reading the
  register's Primary column as the substation whose meter records the asset. In that case, an asset
  list read from the Primary column alone would put the engine at the wrong Primary.
- **Some rows cannot be placed.** Across the whole register, 655 connected rows carry no coordinate
  and 203 carry a coordinate inflated by a power of ten (experiment 673), and 7 slots at the study
  Primaries give no technology.

**NGED's metered generators are the cases where the true output is known.** The three wind farms and
the six solar farms are metered directly ([Final models for the production
port](../../final-models.md)), and so are a battery and a biofuel generator ([D
codes](../../glossary.md#d-codes)). Gas A's own readings give a known answer for only 44 days, from
2023-07-03 to 2023-08-15 (experiment 194).

## What the repository already has

**The ten parts below exist in the repository, and each has a known weakness.**

| Part | What it gives | Status and weakness | Source |
|---|---|---|---|
| Demand generalised additive model (GAM) with explicit terms (`studies.dp_gam`) | A convex baseline of demand from calendar, temperature, heating, holidays, clock changes, and the night step of storage heating | Used in the best-guess stack; on data with realistic noise the heating slope is poorly determined (interval about 38% of the slope wide) | [Experiment 113](../../00113-differentiable-physics-n5-convex-identification-at-realistic-noise/result.md), [experiment 123](../../00123-differentiable-physics-e7-storage-heating-night-step-estimation/result.md) |
| Solar capacity fit and probe | Unmetered solar MW per Primary, season-year resolution | Recovery 0.78 to 0.86 of an injected farm; reads a cloud-following load as solar | Experiments 129, 649, 673; [Unmetered PV capacity: what we know](../solar-disaggregation/unmetered-pv-capacity.md) |
| Physics solar plant curve | Solar output from sunshine and panel geometry | Within 5 points of capacity at 3 of 6 metered farms; the assumed 35 degree tilt fits worse than the farms' best 15 to 20 degrees | [Experiment 282](../../00282-pv-plant-curve-audit/result.md) |
| Physics wind plant | Wind farm output from each ensemble member, with a small neural-network correction | The current research-default wind forecast; its upper quantiles are too narrow | [Experiment 347](../../00347-wind-physics-hybrid-plant-then-mdn/result.md) |
| Engine detector | Running state of a gas engine with whole-number units | Finds a recorded engine when injected; too many false alarms elsewhere | Experiments 196, 291 |
| Battery estimator | Battery power and energy from a price-driven schedule | Simulated batteries above 10% to 20% of peak flow are found; real ones mostly not (provisional) | [Unmetered battery capacity study](../../../studies/unmetered-battery-capacity.md) |
| Injected-event generator | Load transfers added to real power history | A classifier cannot tell its events from 49 real windows (area under the curve 0.55) | [Experiment 660](../../00660-shared-generator-of-injected-load-transfer-events/result.md) |
| Event table | Steps per Primary, with start, end, and size | Finds 0.67 of long, deep injected megawatts and 0.16 to 0.26 of events of a day or less | Experiments 653, 665 |
| Substation cards | Fitted per-Primary properties: solar capacity, daily profile, heating response, holiday effects | Point estimates without intervals; provisional | [Experiment 104](../../00104-differentiable-physics-2-simple-structural-model/result.md) |
| Injection harness | A real metered farm's output added to a Primary's reading, with the recovery measured | The known-answer route of experiments 129, 231, 269, and 474 | [Experiment 231](../../00231-pv-capacity-audit-where-recovery-goes/result.md) |

**The missing part is a joint fit that holds a register-constrained list of assets and estimates
their parameters together.** Each part above was fitted alone, with the other assets left as
unexplained variation or ignored. An asset-inventory fit would put the solar term, the engine term,
and the battery term into one fit, with the register deciding which terms exist at each Primary.

## Candidate uses, ranked

**The two uses that rank highest are an asset list per Primary, shown as inferred, and scenarios
at long horizons for curtailment and planning.** The ranking is a judgement of the value to
NGED against the effort of building and testing, and the ranking puts the asset list and the
long-horizon scenarios first because the maintainer gave them most weight. The values are our
estimates. Each use carries a prediction that could fail and the measurement that would show it.

| Rank | Use | Value to NGED | Effort to build and test | Prediction that could fail | How it would be measured |
|---|---|---|---|---|---|
| 1 | An asset list per Primary, with parameters, beside the register | High: a stated main benefit; shows where the register name and the meter disagree, as for the D8 engine registered under D14 | Low: reuses the parts above | On injected known answers, capacity within 20% for every listed asset that steps 8 noise units or more, and under 0.5 MW for a listed asset that is absent | The smallest test below; the list is then shown as inferred |
| 2 | Scenarios at long horizons (months ahead), for curtailment and planning | High: a stated aim of the forecast, and the horizon the maintainer says matters most | Medium to high: needs a running-time model per asset, forward models with real operating logic, and a demand scenario model | Issued a month ahead, between 0.5% and 2% of held-out half-hours fall above the scenarios' 0.99 quantile, and likewise below the 0.01 quantile, with a narrower 0.01-to-0.99 band than a block bootstrap of the Primary's own residual history | Coverage and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles on held-out months of 2026-01 to 2026-06 |
| 3 | A what-if at days 1 to 10, such as a dormant generator that turns on | Medium to high: a question the maintainer named | Medium: needs a gas-engine plant model, which does not exist | Replaying a known start or stop, such as the D8 engine's change on 2023-08-15, the simulated change in the net series is within 30% of the observed change (a proposed threshold) | Error of the simulated step against the observed step on the known events |
| 4 | A new connection added from the register before it appears in the data | Medium to high for planning; needs NGED to say which connections matter | Medium | For assets connected in 2024 to 2026, a scenario that adds the registered asset at its connection date moves the post-connection error of the solar or step term by at least half the asset's registered MW | Retrospective: hold out the data after each connection date and compare with and without the registered asset |
| 5 | A fallback when telemetry is missing or wrong | Medium: the service must degrade, not fail | Low | The forward simulator scores better than climatology at days 1 to 10 when no recent power is available | CRPS and tail pinball loss against climatology and the lag-free forecast on the same rows |
| 6 | Simulated series for a known-answer test bed, and conceivably for training the forecaster | Medium: helps the evaluation problem behind the discussion; the maintainer calls training on simulated series "a bit sketchy" | Low once use 1 exists | Existing forecasting methods keep their ranking on the simulated series (Kendall rank correlation of at least 0.8 with their real scores, where 1 means the same ranking) | Rank transfer, as proposed in the discussion |
| 7 | Counterfactuals ("the load without the engine") | Low until NGED asks | Low | None measurable: no truth exists | Not measurable |

**Use 5 is unlikely to matter, because a lag-free forecast already covers missing telemetry.** The
fallback would have to beat that forecast, not climatology, to change the service. Onboarding a
Primary with a short history would be another use. The maintainer has paused onboarding work, so
onboarding is not ranked here.

## The smallest test

**A known-answer test of the inventory fit on the six MW Primaries, with real metered output added
and an asset list that is partly wrong, is the cheapest test that could stop the idea.** The test is
an option for the maintainer, not a planned experiment.

- **Rows:** the cleaned half-hourly readings of the six MW Primaries from 2024-07-01 to 2026-06-30.
- **Injected assets, one set per Primary:** a metered solar farm's output at 50% of the Primary's
  solar capacity, as in experiment 231; Battery A's recorded output; and an engine of two 2 MW units
  whose running spells are drawn from a fixed distribution with Gas A's recorded median of 2 hours.
- **The list given to the fit:** the injected solar farm, the battery, and the engine, plus one
  decoy engine (an engine that is listed but not injected). At one Primary in three, the injected
  battery is left off the list.
- **The fit:** the demand GAM, one solar capacity term, an engine term with a unit size, a unit
  count, and a log-normal spell-length distribution, and a battery term from the battery estimator.
  No half-hourly state is reported.
- **Predictions that could fail:**
    - the fit recovers each listed, injected asset's capacity within 20%, and the engine's median
      spell length within 30%;
    - the decoy engine gets under 0.5 MW at 5 or more of the 6 Primaries;
    - the residual of the fit shows the unlisted battery as a daily cycle.
- **Stop rule:** stop if the decoy gets 1 MW or more at 3 or more of the 6 Primaries. A fit that
  finds a listed engine wherever it is told to look would give an asset list that cannot be trusted.
- **Effort:** CPU only, about a day of one-core fits. The figure is a guess, scaled from the size of
  the solar-capacity fits.
- **Option for later:** a language-model loop that matches the registers to the signals and edits
  the simulator, to be compared with this fit; see
  [option 6 of the plan](plan.md#option-6-in-detail-a-language-model-loop).

## Risks

- **Circularity.** A simulator fitted to the metered record and tested against the same record
  passes any test that shares its assumptions. The test above uses real metered output as the truth,
  so the injected asset's shape is not the simulator's own. The demand beneath the injected asset is
  still the fitted GAM's own output, so scenario calibration must be scored on months the fit has
  not seen.
- **Diversity against calibration.** A scenario generator can be made so wide that it covers every
  month and is no longer informative. Each scenario test must report the band width beside the
  coverage, and compare with a block bootstrap of the Primary's own history, which needs no asset
  inventory at all.
- **What NGED could still say.** NGED's records may cover the running hours of the larger engines,
  which Primaries have export limits or flexible connections, and whether any metered generator's
  output is still added back into a Primary's reading. We have not asked.
- **Validation of the asset list.** NGED cannot confirm which assets sit behind which meter beyond
  the register, so only known-answer tests validate the list. The D8 engine registered under D14
  shows that agreement with the register is not proof.
- **Bands may do the work.** At D8, widening the bands from past residuals cut the quantile score by
  9% to 19% and lifted the share of readings inside the 90% band from 42% to between 80% and 87%
  (experiment 264). A scenario use must beat that simpler widening before an engine term is worth
  adding.

## Open question for the maintainer

- **Should the decoy rule apply to solar too?** Every register name lists solar, so a decoy can only
  be tested for engines, batteries, and wind. If solar matters most, the existing solar-capacity
  work already covers the inventory's solar term, and the test can drop the solar term.
