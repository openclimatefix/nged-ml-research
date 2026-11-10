# Options for proposing the simulator to NGED (advisor analysis)

**This page records an AI advisor's analysis of 2026-11-10 as a set of options; nothing on it is
agreed work.** The advisor answered whether a "Lego replica" simulator of each NGED substation,
built from forward models of the assets behind it and used for scenario planning, is a better
proposal to NGED than extending the forecast. It wrote from the merged pages of this repository and
of the production repository, two NGED planning documents, and about 17 web searches. No fit was
run. Statements about NGED's planning tools, about Ofgem and the Department for Energy Security and
Net Zero, and about published papers rest on external pages the project has not independently
verified, except where a line below says the page was checked. Numbers about this repository are
quoted from the merged page linked beside each. Where a statement is the advisor's judgement and not
a sourced finding, the sentence says so. NGED's metered generators are anonymised (Wind A to C,
Solar A to F, Battery A, Gas A), with no coordinates and no dates on a generator's series. Nothing
below uses the final-test window 2026-07-01 to 2026-10-31. The effort figures and the phases are the
advisor's guesses, and the page commits the project to none of them.

## Terms used on this page

- **DFES:** Distribution Future Energy Scenarios, NGED's published scenarios of connected demand and
  generation up to 2050, by electricity supply area.
- **ESA:** electricity supply area, the area supplied by one primary substation below 33 kV.
- **NDP:** NGED's Network Development Plan.
- **PSS/E:** the commercial power-flow program that NGED's Switch-Level Analyser uses as its engine.
- **Load set:** the half-hourly demand, generation, and storage values the load-flow tool reads for
  each node and representative day.
- **Known-answer test, register, Primary:** as defined on [the asset-inventory
  page](asset-inventory-simulator.md#terms-used-on-this-page).
- **Decoy:** an asset, such as an engine, that the register lists for a Primary but that is absent
  from the real readings, added to a known-answer test to see whether the fit gives it capacity.
- **BSP and GSP:** bulk supply point and grid supply point, the nodes where NGED's network meets the
  transmission network.
- **Fitted or assumed:** the label the advisor proposes for each simulator block, saying whether its
  parameters were estimated from the net series or taken from the register and public operating
  knowledge.

## Take-home

**Propose it as a complement to the forecast, scoped as a "per-substation asset list and behaviour
profile" product that plugs into the planning chain NGED already run, and start it only after the
production port has shipped.** Three findings drive that recommendation. First, NGED already have
the scenario-planning machinery the idea assumes they lack: the Distribution Future Energy Scenarios
(DFES) produce volumes per primary-substation supply area, a customer-behaviour step turns the
volumes into half-hourly profiles per technology, and a bespoke tool runs those profiles through
PSS/E load flow for every half-hour, representative day, outage, year, and scenario (see [question
2](#2-what-a-dno-needs-for-scenario-planning-and-what-nged-already-have)). What that chain does not
have is substation-specific knowledge of what sits behind each meter and how it behaves, which is
exactly what a fitted per-substation model can supply. Second, the repository's evidence says the
part of the idea that fits the data is the inventory (a total solar capacity at season-year
resolution, large long steps, and a check on register discrepancies), while the part that sounds
most attractive, forward models with real operating logic for batteries and engines, is the part the
net series cannot identify (see [question 3](#3-the-nwp-analogy-tested)). Third, a stand-alone
"simulator" would be read by NGED as a rival to PSS/E and to the digital-twin projects already
funded, whereas a profile-and-inventory product that writes into their load set is not a rival (see
[question 8](#8-counter-arguments-and-risks)).

**The recommendation would be wrong under any of five conditions.**

- NGED's planners say the clustered DFES profiles are adequate for primaries and they do not want
  substation-specific profiles. The proposal then has no customer, and only the asset list survives.
- The known-answer test on the asset-inventory page meets its stop rule (a listed-but-absent engine
  given 1 MW or more at three or more of the six MW Primaries). The list cannot then be trusted and
  the product has no first deliverable.
- NGED can already produce the list. The `MPAN to Substation Number.csv` file NGED supply
  "associates each Embedded Capacity Register (ECR) generator to the substation it connects to",
  with all trial-area generators present ([data
  sources](https://openclimatefix.github.io/nged-substation-forecast/roadmap/data-sources/)). The
  inventory's value is therefore the unregistered share and the behaviour parameters, not the
  registered sites, and if NGED hold running-hour records for the large engines as well, the value
  shrinks further.
- The January 2027 v1.0 release slips. The requirements page puts the forecast first and everything
  else as support for it
  ([requirements](https://openclimatefix.github.io/nged-substation-forecast/background/requirements/)),
  and a proposal that competes for the same people before the forecast ships costs the forecast.
- Ofgem's central asset register arrives on time. Ofgem consulted in December 2025 on new licence
  requirements for DNO asset registers (the consultation page does not describe a central register
  as one of its options; that part is not confirmed) ([Ofgem, Enhancing asset
  visibility](https://www.ofgem.gov.uk/consultation/enhancing-asset-visibility-distribution-network-operator-options)),
  and published a draft impact assessment of the central-register proposals in October 2026 ([Ofgem
  draft impact
  assessment](https://www.ofgem.gov.uk/sites/default/files/2026-10/Draft-impact-assessment-central-asset-register-proposals.pdf)).
  A statutory register would make an inferred list a cross-check rather than a product.

## 1. Separate extension, complement, or not at all

**A complement, framed to NGED as "what is behind each of your substations and how does it behave",
and not as a simulator.** The maintainer's three motivations map onto three different products, and
only one of them is well supported.

- **The asset list** is the deliverable with the strongest evidence and the clearest customer. The
  Department for Energy Security and Net Zero reports that DNOs "estimate that less than half of
  small-scale distributed energy assets are registered with them" ([DESNZ call for
  evidence](https://www.gov.uk/government/calls-for-evidence/improving-the-visibility-of-distributed-energy-assets/improving-the-visibility-of-distributed-energy-assets-call-for-evidence)),
  and the Energy Networks Association's response to Ofgem is reported to argue that "applying
  machine learning to consumption data can identify unregistered assets" (this quotation is not
  re-checked: the PDF could not be opened, [ENA response, February
  2026](https://www.energynetworks.org/assets/images/2026/260226ena-response-to-enhancing-asset-visibility-consultation-(feb-2026).pdf)).
  In the trial area the probe's interval lies above the solar capacity that three registers list
  together at 8 of the 12 Primaries with an interval ([experiment
  673](../../00673-pv-capacity-for-nged-per-primary-with-register-cross-check/result.md)), so the
  registers miss part of the solar capacity, which is the gap an inferred list fills.
- **Long-horizon scenarios** are the DFES's job, and the DFES already run from "an April 2025
  baseline to 2050" at supply-area level ([DFES 2025 methodology, page
  18](https://dso.nationalgrid.co.uk/downloads/15126/nged-dfes-2025-methodology-.pdf)). What a
  fitted model adds is the substation's own half-hourly behaviour (its heating response, its night
  step, how often its engine runs) in place of a cluster's profile. That is a profile product, and
  the place to deliver it is the load set NGED's load-flow tool reads.
- **Days 1 to 10 what-ifs** (a dormant generator turns on) are a forecast feature, not a simulator.
  The forecast already has a lag-free output, and the asset-inventory page ranks a forward simulator
  as a fallback at use 5 and says it "is unlikely to matter, because a lag-free forecast already
  covers missing telemetry" ([asset-inventory simulator](asset-inventory-simulator.md)). A what-if
  that adds a registered engine's output to the forecast needs the engine's plant model, which does
  not exist (same page, "What the forward direction means"), and a 44-day record of Gas A to fit it
  on.

**"Separate extension" is the right contract form but the wrong framing.** The requirements page
says the Network Innovation Allowance project's "single hard requirement is that the project gives
NGED new information about forecasting for their assets"
([requirements](https://openclimatefix.github.io/nged-substation-forecast/background/requirements/)).
A planning product sits outside that, so a separate agreement is needed. Framing the agreement as a
simulator invites comparison with PSS/E and with the digital twins in [question
2](#2-what-a-dno-needs-for-scenario-planning-and-what-nged-already-have); framing it as a data
product (an inventory table and a profile table per substation) invites comparison with the DFES
baseline and profiles, which is a comparison the project can win.

## 2. What a DNO needs for scenario planning, and what NGED already have

**NGED's planning chain already runs scenarios through a load-flow model at half-hourly resolution,
so the gap is not the machinery but the inputs that describe each substation.** The Network
Development Plan (NDP) 2026 methodology describes the chain ([NDP 2026 introduction and
methodology](https://dso.nationalgrid.co.uk/downloads/15146/ndp-2026-introduction-methodology.pdf),
page numbers are approximate and not re-checked; they may be one page off the printed footer):

- **Scenarios.** Four DFES scenarios, assessed "across the next 10 years" (page 31), out to 2050 in
  the headroom report (page 34).
- **Unit.** An Electricity Supply Area (ESA), "the geographical area supplied by a primary
  substation that supplies electricity at a voltage below 33 kV or a customer directly supplied
  at..." ([DFES 2025 methodology, page
  10](https://dso.nationalgrid.co.uk/downloads/15126/nged-dfes-2025-methodology-.pdf)). The DFES
  baseline comes from "NGED connection agreement data" (whether public registers are also named was
  not re-checked), and sites "with an accepted network connection offer are individually researched"
  (page 20).
- **Profiles.** A customer-behaviour step turns volumes into "a load set of MW/MVAr values", using
  "the latest published LTDS Table 3 data for the starting load assumptions" (page 16). The load set
  includes "half-hourly profiles for each type of demand, generation and storage and representative
  day" (page 31). The 2024 customer-behaviour report replaced the earlier demand profiles with more
  classes "to better model the variety of primary substation demand patterns across the network",
  built with machine-learning clustering of domestic profiles, and NGED publish the profiles on
  their open-data portal ([DFES
  Profiles](https://connecteddata.nationalgrid.co.uk/dataset/dfes/resource/bc4bfcb3-6ec3-400e-a4c3-3f73e3627802)).
- **Representative days.** Five: winter, summer, intermediate-cool, and intermediate-warm peak
  demand with minimum coincident generation, and summer peak generation with minimum coincident
  demand (page 17, Table 6).
- **Load flow.** "The Switch-Level Analyser tool is a bespoke power system analysis program written
  in Python 2.7. It uses PSS/E version 34 as its core analysis engine" (page 31), developed since
  2016, run for "approximately 3.5 billion individual load flow studies for each scenario and year
  combination" (page 20), returning MVA branch flows, voltage exceedances, lost load, and group load
  per GSP, BSP, and primary group (page 31).
- **What the tool does not model.** Active network management, because "the existing analysis tools
  do not replicate the ANM logic", and procured flexibility (page 32).
- **Headroom.** Since 2024 a sensitivity method adds "a 1 MW load" at each boundary node under each
  contingency and assumes linearity (page 34).
- **Network model.** The Long Term Development Statement is published as a Common Information Model
  from 132 kV down to 11 kV, and Ofgem's form of statement says the models are sufficient to support
  steady state power flow calculations ([LTDS
  CIM](https://connecteddata.nationalgrid.co.uk/dataset/ltds-common-information-model); [Ofgem form
  of
  statement](https://www.ofgem.gov.uk/sites/default/files/2024-01/Form%20of%20Long%20Term%20Development%20Statement.pdf)).
  The production repository records that no Flexpectation code reads the LTDS and that the join to
  `time_series_id` is uninvestigated ([data
  sources](https://openclimatefix.github.io/nged-substation-forecast/roadmap/data-sources/)).

**Three digital-twin efforts already surround NGED, and none of them fits a substation to its own
telemetry.** Fractal Flow, Northern Powergrid's Strategic Innovation Fund project with every GB DNO
and NESO as Beta partners, runs September 2026 to September 2028, trials on historic data in three
control rooms none of which is NGED's, and forecasts "up to 24-hours ahead" ([cost-savings
metrics](https://openclimatefix.github.io/nged-substation-forecast/roadmap/cost-savings-metrics/#tier-2-full-power-flow-modelling)).
National Grid and Atos launched Triton in February 2026, a planning tool "built from thousands of
datasets made available by distribution network operators and transmission owners" ([Electrical
Review](https://electricalreview.co.uk/2026/02/03/atos-national-grid-launch-triton-digital-twin-tool-for-network-planning/)).
NGED's PRIDE project (Planning Regional Infrastructure in a Digital Environment) is building a
digital twin for regional energy planning on the LAEP+ platform ([National Grid project
page](https://nationalgrid.co.uk/innovation/projects/planning-regional-infrastructure-in-a-digital-environment-pride-discovery);
this page was found by search and has not been read). The advisor's search found no GB
distribution-operator project that infers per-substation asset inventories or dispatch behaviour
from the substation's net series; that is a statement about one search of the open web, not about
the field.

**Where a fitted per-substation model would sit: it replaces two inputs of the chain and leaves the
load flow alone.** The first input is the baseline: the register-based count of what is connected,
which misses the unregistered share. The second is the profile: a cluster's half-hourly shape per
technology, which a substation-specific fit replaces with the substation's own heating slope, night
step, holiday effects, and engine running-time distribution. Everything downstream, the PSS/E load
flow, the contingency analysis, and the headroom sensitivities, stays NGED's. In that framing the
deliverable is a load set NGED can read, not a simulator NGED must learn.

**The value is in the inventory and the substation-specific profile; the explanations and the
forward scenarios are second-order.** The explanations matter to NGED's trust in the inventory (a
planner needs to see why the fit says 3 MW of solar), and in the advisor's judgement an
operator-visible feature such as the Economy 7 step helps NGED trust the fit. The forward scenarios
matter only where NGED's own profile step is weaker than ours, which is at the assets the net series
cannot identify (batteries and engines, [question 3](#3-the-nwp-analogy-tested)). A worked version
of the scenario question already exists in the delivery recommendation: NGED's power-flow study
"needs node injections", summing node scenarios missed the BSP export tail (33.9% to 79.0% of actual
export flows fell below the sum's q05 against a nominal 5%), and the direct forecast of each BSP
meter came closest to nominal (4.9% and 7.2% below q05, 6.7% and 9.9% above q95) ([delivery
recommendation](../calibration-and-uncertainty/delivery-recommendation.md)). A simulator's scenarios
would face the same coherence test, and the recommendation's conditional stress vectors remain
untested.

## 3. The NWP analogy, tested

**The analogy holds in form and fails in the ratio of observations to unknowns.** A numerical
weather prediction (NWP) model combines a forward model of known physics with an estimate of the
state from dense observations, and runs an ensemble for uncertainty. The simulator would combine a
forward model of each asset with parameters estimated from one net series per substation. The table
says where each element of the analogy stands.

| Element of data assimilation | Holds here | Breaks here |
|---|---|---|
| Forward model from known physics | Solar output from sunshine and panel geometry; wind output from an ensemble member's wind field; both exist ([asset-inventory simulator](asset-inventory-simulator.md)) | Demand "physics" is behaviour: the demand GAM is a statistical fit, so a simulator built on it is circular when used to test anything that shares its assumptions (same page, "Risks") |
| Observation operator | The meter is a known sum: demand minus generation below it. At the eight MVA Primaries the operator is the absolute value, a nonlinearity the probe work has to model with an unknown reactive floor ([unmetered PV capacity](../solar-disaggregation/unmetered-pv-capacity.md)) | One scalar per half-hour per substation. NWP assimilates millions of observations of many state variables; here several assets of one type behind one meter "carries their sum", and the register lists 77 solar slots with two or more at 12 of 15 register names (asset-inventory page) |
| Parameters against state | Slowly varying parameters (a season-year solar capacity, a unit size, a spell-length distribution) are what the net series identifies | The state (is the engine on now; is the battery charging) is what NWP assimilates and what the net series does not give: knowing the engine's state helped the forecast only when read at the half-hour forecast, and with the last settled day's state it made the forecast 10% worse ([experiment 294](../../00294-regime-aware-model-at-d8-engine-state-term/result.md)) |
| Ensembles for uncertainty | Ensemble weather already drives the wind and solar plants; a block bootstrap of the substation's own residual history is a cheap reference ensemble (asset-inventory page, "Risks") | Spread from weather members does not cover parameter uncertainty: the heating slope's interval is about 38% of the slope wide on realistic noise ([experiment 113](../../00113-differentiable-physics-n5-convex-identification-at-realistic-noise/result.md)), and the capacity intervals were honest only with a cloud-load term ([experiment 474](../../00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals/result.md)) |
| Stationarity | Solar and wind physics do not drift | Demand drifts: D8's night load stepped after March 2025 ([experiment 199](../../00199-ps2b-engine-detector-v2-ramp-night-price/result.md); the asset-inventory page says the reason is covered by no tested explanation); solar capacity drifts 0.1 to 0.7 MW per year, overlapping the within-month minimum detectable capacity of 0.05 to 0.35 MW ([unmetered PV capacity](../solar-disaggregation/unmetered-pv-capacity.md), where both ranges are marked as from a brief and are not re-checked) |

**The published identifiability literature says the same in general terms.** Odunlami and Netto
check the observability of an aggregated distributed-energy-resource model before running a Kalman
filter, and call observability a necessary but not sufficient condition for identifiability ([arXiv
2510.10892](https://arxiv.org/html/2510.10892v1)). A PowerTech paper on thermostatically controlled
loads finds that estimating non-identifiable parameter sets gives estimates "biased by the values
used to initialize the iterative estimation algorithm" ([Parameter identifiability of
TCLs](https://aminer.org/pub/654dc45f939a5f4082bc3102/parameter-identifiability-and-estimation-of-thermostatically-controlled-loads)).
A variational data-assimilation paper on load-model calibration warns that "the paucity of
informative data makes overfitting a real danger" ([arXiv
2311.07676](https://arxiv.org/pdf/2311.07676)). None of the three is a substation-level net-load
study, and the advisor's search found no identifiability analysis of behind-the-meter disaggregation
as such.

**The repository's evidence bounds how far the fit can go, asset by asset.**

- **Solar.** The convex capacity fit recovers a median 0.78 to 0.86 of a real farm's output added to
  a Primary ([experiment
  129](../../00129-differentiable-physics-pv-capacity-known-answer-test/result.md)); the neural
  probe reads 0.68 of an injected farm raw and, divided by its own recovery factor of 0.68,
  estimates 2.4 to 3.9 MW of unmetered solar at each of the six MW Primaries ([experiment
  649](../../00649-pv-capacity-table-season-year-from-stored-probes/result.md)). A known step was
  found within 2% of its size and one month of its date, but in a year with no steps the monthly fit
  invented steps above 0.5 MW at 2 of 5 Primaries ([experiment
  207](../../00207-dp-meter-regime-steps-e2-step-capacity-known-answer/result.md)). No candidate
  shape passed the placebo test of experiment 474. Season-year capacity is identifiable with a
  stated calibration; monthly steps are not.
- **Engines.** A hidden Markov detector finds a recorded engine when its output is injected but
  raises too many false alarms at three of six other Primaries ([experiment
  196](../../00196-ps2-quantised-engine-detector-on-d8-with-known-answer-gate/result.md)). The
  fitted unit is 1.4 to 1.5 MW against a recorded 1.97 MW (experiment 294). Gas A ran in 0.170 of
  half-hours in spells with a median of 2 hours over its 44 recorded days ([experiment
  194](../../00194-ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census/result.md)),
  and for load shifts of a day or less the event table finds 0.16 to 0.26 of the injected megawatts
  ([experiment 665](../../00665-event-table-recall-by-length-and-depth/result.md)). Spells that
  short are in the class the net series does not identify unless the units are large against the
  noise.
- **Batteries.** Simulated in-family batteries are flagged in 58% of sums at a 10% share and 94% at
  20%; real public batteries added to NGED flows were flagged in 0 of 736 sums at 5% and 10% shares
  and 50 of 736 (6.8%) at 40%; Battery A put back into its bulk supply point at up to 11 times its
  metered output was flagged in 0 of 20 blocks; the estimator reports about 2.6% of peak flow where
  no battery exists ([unmetered battery capacity](../../../studies/unmetered-battery-capacity.md),
  all provisional). The register lists 3 connected storage slots totalling 0.65 MW at the 16
  Primaries (asset-inventory page), so a battery block has almost nothing to fit to.
- **Fits left free to add generators.** In the four experiments the asset-inventory page read that
  let a fit decide whether a generator exists, the fit reported generators that are absent or
  unconfirmed: a generator finder reported one at every substation ([experiment
  110](../../00110-differentiable-physics-5-embedded-generation-and-capacity/result.md)), an engine
  mixture fitted a running engine at two substations that have none ([experiment
  264](../../00264-dp-gas-engine-asymmetric-bands-at-d8/result.md)), and a real-series step scan
  found about as many solar steps as it invents where none exists ([experiment
  215](../../00215-dp-pv-step-scan-real-series-register-match/result.md)). The asset types per
  Primary therefore have to come from the register, and the register's Primary column can name the
  wrong meter: the 6 MW engine that appears in D8's readings is registered under D14
  (asset-inventory page).

**What this means for the proposal: the fit can deliver a register-constrained inventory with
parameters at season-year resolution, and cannot deliver fitted operating logic.** The battery
schedules and engine equations the maintainer describes would therefore enter the simulator as
assumptions taken from the register and from public operating knowledge (the four GB battery
scheduling algorithms on the [battery scheduling
page](https://openclimatefix.github.io/nged-substation-forecast/background/gb-battery-scheduling/);
a vendor claim that one make of gas engine (INNIO Jenbacher) ramps to full load in under five
minutes, [Clarke Energy](https://www.clarke-energy.com/gas-engines/peaking-stations/)), with their
parameters checked against the net series only where the asset is large against the noise. Assumed
blocks are still useful for a planning scenario, provided the output labels each block "fitted" or
"assumed".

## 4. The block library

**Six blocks matter, and they split cleanly into physics blocks that exist, a demand block that
exists, and two operating-logic blocks that do not.** The table ranks them by value to a planning
scenario and by effort, using the asset-inventory page's "What the forward direction means" and
"What the repository already has" tables as the source of what exists; the ranking is the advisor's
judgement.

| Rank by value | Block | Physically well defined? | Exists in the repository? | Effort to make it a forward block | Value to scenario planning |
|---|---|---|---|---|---|
| 1 | Demand under the assets (calendar, temperature, heating, holidays, clock changes, night step) | No: behavioural, but convex and explicit | Yes, the demand GAM `studies.dp_gam`; the heating slope's interval is about 38% of the slope wide on realistic noise (experiment 113) | Low: needs a weather-scenario driver (ERA5 or a climatological year) in place of a forecast | High: the substation's own profile replaces a cluster's profile in NGED's load set, and the night step is the operator-visible feature |
| 2 | Solar (rooftop fleet and registered sites) | Yes | Yes: the physics plant curve (experiment 282) and the capacity fit and probe (experiments 129, 649, 673) | Low: scale the curve by the fitted season-year capacity; a register-anchored yearly trajectory is the cheapest January option on the [unmetered PV page](../solar-disaggregation/unmetered-pv-capacity.md) | High: every register name lists solar (77 slots, 33.2 MW), and the summer-peak-generation day is one of NGED's five representative days |
| 3 | New connections from the register (16 accepted slots, 3.7 MW: 13 solar, 2 storage, 1 wind) | Yes for solar and wind | Partly: the physics plants exist; no code adds a registered asset at its energisation date | Low to medium | Medium to high: this is the DFES pipeline step done per substation; needs NGED to say which connections matter |
| 4 | Gas engine (unit size, unit count, ramping, minimum on-duration, running-time distribution) | Yes for the plant; no for the dispatch rule | No: the detector is a hidden Markov model with no ramp or minimum-on equation; Gas A's record is 44 days | Medium: the equations are simple, the dispatch rule (price, Triad, Capacity Market) is not, and the fit is weak (unit 1.4 to 1.5 MW against 1.97 MW) | Medium: 2 registered gas or oil slots (7.0 MW) and 8 bioenergy slots (11.1 MW) at 6 register names; the D8 engine is the one large case |
| 5 | Battery (scheduler with price signals and state of charge, several profiles) | Yes for the plant; the schedule is a choice among optimisers | Partly: `studies.battery_dispatch` holds a daily linear programme and a rank rule; `studies.battery_templates` holds five tariff and merchant schedules; no module runs them forward for a registered battery | Low to medium to run forward; not fittable (real batteries mostly undetected) | Low today: 3 connected slots, 0.65 MW, and 2 accepted, 0.4 MW; rises if NGED's storage pipeline grows |
| 6 | Wind | Yes | Yes: the physics plant on each ensemble member (experiment 347); capacity estimation was refuted by experiment 617's known-answer test (the decision to park it is not on a merged page and is not re-checked) | Low | Low: 1 connected slot, 1.2 MW, at D9 |

**Two blocks the maintainer did not name belong in the library because the net series contains
them.** Load transfers between Primaries (each Primary spends roughly 10% of its time in an abnormal
arrangement, [network
page](https://openclimatefix.github.io/nged-substation-forecast/background/network/)) are the
largest non-asset feature of the series, and the shared injected-event generator of [experiment
660](../../00660-shared-generator-of-injected-load-transfer-events/result.md) cannot be told from 49
real windows (area under the curve 0.55), so a transfer block exists. Meter regimes (false zeros,
stuck values, MVA folding, commissioning ramps) are the other; a scenario simulator should produce
the clean series, so these matter only to the fitting side.

**The minimum set that makes a scenario useful is blocks 1 to 3: demand, solar, and new
connections.** With those three a scenario can answer the questions NGED's representative days ask
(winter peak demand with minimum generation, summer peak generation with minimum demand) for the
substation's own profile rather than a cluster's, and can add a pipeline connection at its
energisation date. Blocks 4 and 5 add value only at the Primaries whose register lists such an
asset, and only as labelled assumptions until a known-answer test says otherwise.

**External libraries supply the load flow, not the behaviour.** pandapower, PyPSA, OpenDSS, and
GridLAB-D are power-flow and optimisation tools (the PyPSA comparison page names OpenDSS and
GridLAB-D without describing them, [PyPSA comparable
software](https://docs.pypsa.org/v0.29.0/references/comparable-software.html)); NGED already have
PSS/E for that layer. The bottom-up demand models that exist, the CREST one-minute domestic model
validated on 22 East Midlands dwellings ([Richardson et
al.](https://repository.lboro.ac.uk/articles/journal_contribution/Domestic_electricity_use_a_high-resolution_energy_demand_model/9573941))
and NREL's dsgrid, whose documentation reports that "the final residuals were significant,
especially at higher geographic and temporal resolution" ([dsgrid
dataset](https://www.osti.gov/dataexplorer/biblio/dataset/1823248)), need appliance, occupancy, and
tariff data the project does not hold per substation, which is why the discussion session rejected a
bottom-up build first ([simulator discussion](simulator-discussion-2026-11-09.md)). Elexon's
settlement profile classes (the count of eight was not re-checked) give generic half-hourly shapes,
including a switched-load split for Economy 7 ([Elexon load
profiles](https://bscdocs.elexon.co.uk/guidance-notes/load-profiles-and-their-use-in-electricity-settlement)),
and are a reasonable prior for the demand block but not a substitute for the fitted GAM.

## 5. The fitting method and the LLM's role

**Divide the labour by what each tool is good at: convex fits estimate parameters, data analysis
decides which parameters are identifiable, and the LLM proposes register matches and model edits
inside a budget, and never touches the objective.** The order matters because the failure modes of
an LLM in the loop are the failure modes the repository has already met in free fits.

- **Convex fits (the demand GAM, the solar capacity term, the engine term, the battery estimator)
  own every number.** They are reproducible, their intervals can be computed, and the known-answer
  harness already scores them. The missing piece is the joint fit the asset-inventory page names:
  one fit holding a register-constrained list of assets and estimating their parameters together.
- **Data analysis owns the identifiability gate.** Before a parameter is reported, a known-answer
  test and a placebo say whether the net series can carry it. The existing results already give the
  gate's values: season-year solar capacity passes with a calibration factor; monthly solar steps,
  engine state, and batteries fail.
- **The LLM owns two bounded tasks.** First, matching: reading a register row (technology, MW,
  connection date, voltage, register name, free-text site name) against the signals the analysis
  found (a step of X MW on date Y, a daily cycle, a midday dip) and proposing which register row
  explains which signal, with a stated confidence. Second, proposing edits: given the residual
  diagnostics of the current fit, proposing one of a fixed menu of moves (add a listed asset type,
  change a spell-length prior, flag a meter regime), each of which the convex fit then evaluates.
  The matching task is fuzzy text work an LLM does well; the edit task is a search an LLM can
  propose but must never score.

**The failure modes are known from this repository and from the one precedent for LLM judgement on
these series.**

- **Inventing assets that match statistics.** A fit left free to add generators did exactly that in
  four experiments (question 3). An LLM proposing edits is a fit left free with a richer vocabulary.
- **Overfitting the statistics.** A simulator whose moves are judged on the months it was fitted to
  will match them.
- **Non-reproducibility.** Two runs of an LLM loop give two inventories unless the moves are logged
  and replayable.
- **Miscalibrated judgement.** In the one precedent, a Sonnet reviewer judging event-table
  candidates reached precision 0.45 against a pre-registered gate of 0.80, and 0.75 after a post hoc
  exclusion of recurring candidates (the page says five in its headline and 22 in its `best_arm`
  field, so the count is not re-checked), still under the gate ([experiment
  692](../../00692-event-table-reviewer-calibration-gate/result.md)). Battery-Sim-Agent, the nearest
  published LLM-in-the-loop calibration, keeps the agent in a closed loop with the simulator and
  judges each move against target data with quantitative error metrics (most of its benchmark
  targets are simulator outputs with perturbed parameters, and only its real-world validation uses
  experimental cell data) ([arXiv 2605.29560](https://www.alphaxiv.org/abs/2605.29560.md)); the
  multi-agent parametrisation of Xia et al. is not about power systems ([arXiv
  2405.18092](https://www.emergentmind.com/papers/2405.18092)). The advisor's search found no
  power-system example.

**The controls are the ones the brief lists, and each maps onto an existing repository mechanism.**

- **Register constraints.** The LLM may propose an asset only from the register's rows for that
  register name and its neighbours (the D8 and D14 case says neighbours are needed), and a solar
  term is always present.
- **Held-out months.** Every move is scored on months the fit has not seen, as the asset-inventory
  page already requires for scenario calibration.
- **Known-answer tests with a decoy and an unlisted asset.** The smallest test on the
  asset-inventory page already has both: a listed-but-absent engine, and a battery left off the list
  at one Primary in three. The LLM loop runs the same test, and the stop rule (decoy at 1 MW or more
  at 3 or more of 6 Primaries) applies to it.
- **A fixed edit budget and logged moves.** A ceiling on the number of edits per Primary, every edit
  written to a log with its diagnostic, its proposal, its score before and after, and the held-out
  score, so the inventory can be rebuilt without the LLM.
- **A plain fit as the comparator.** The LLM loop has to beat the register-constrained convex fit on
  the known-answer test before any of its inventories is shown to NGED. The LLM-option brief in this
  folder puts that dependency first, and the advisor judges the ordering right.

**What must be explainable, and how an operator checks it.** For each asset on the list: its
technology and register row (or "unregistered"); its fitted parameters with intervals; the signal
that supports it (a figure of the residual with and without the term, on held-out months); the
known-answer recovery of the method that produced it; and the move log that led to it. An operator
checks an explanation by three questions the list must answer: does the register or the connections
team know this site; does the figure show the signal on a day the operator can look up in the
telemetry; and does the same method find the asset again when the operator chooses the held-out
months. An explanation an operator cannot check against something they already hold is a story, and
the D8 engine shows that register agreement alone is not a check.

## 6. Evidence and validation NGED would accept

**No ground truth exists below the Primary, so validation has four legs: known-answer injection,
held-out months, NGED's own records, and a comparison with the planning inputs NGED use today.**

- **Known-answer injection with real metered output.** The injection harness of experiments 129,
  231, 269, and 474 adds a real metered farm's output to a Primary and measures recovery; the
  smallest test extends it with Battery A's record and a drawn engine, plus a decoy and an unlisted
  asset. The predictions that could fail are already written: capacity within 20% for each listed
  injected asset, median spell length within 30%, the decoy under 0.5 MW at five or more of six
  Primaries, and the unlisted battery visible as a daily cycle in the residual (asset-inventory
  page, "The smallest test"). Those are the figures to put in front of NGED.
- **Held-out months for scenarios.** Scenario calibration scored on 2026-01 to 2026-06 with the fit
  trained before, reporting coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles and the band width
  beside a block bootstrap of the Primary's own history (asset-inventory page, use 2). Nothing from
  2026-07-01 onwards may be used.
- **Retrospective new connections.** For assets connected in 2024 to 2026, hold out the data after
  the connection date and show that adding the registered asset moves the post-connection error by
  at least half its registered MW (asset-inventory page, use 4). This is the test closest to what
  the DFES pipeline step does.
- **NGED-supplied checks.** The asset-inventory page lists what NGED may still hold and records no
  request for: running hours of the larger engines, which Primaries have export limits or flexible
  connections, and whether any metered generator's output is still added back into a Primary's
  reading. Add to that list: the `MPAN to Substation Number` mapping as a check on register-name
  placement, the "Reverse Power Flow?" column of `Monitor Direction.csv` (which conflicts with the
  telemetry at D8, D12, and D22, [state at the January
  pause](../lsc-detectors/state-at-january-pause.md)), and the LTDS Table 3 loads the NDP starts
  from.
- **Comparison with the DFES inputs.** A comparison a planner is likely to make unprompted is
  between our fitted profile and the DFES cluster profile for the same substation and representative
  day, and between our inventory and the DFES baseline for that supply area. Both DFES inputs are
  published ([DFES
  Profiles](https://connecteddata.nationalgrid.co.uk/dataset/dfes/resource/bc4bfcb3-6ec3-400e-a4c3-3f73e3627802)),
  so the comparison can be made before NGED are asked anything. A fitted profile that explains more
  of the held-out telemetry than the cluster profile is the evidence a planner understands.

**The known limits must be stated rather than scored over.** The delivery recommendation's list of
what cannot be validated with the pilot's data applies here too: q99 and beyond, any element that is
not metered, combined rare events, and the split of a stress scenario across nodes ([delivery
recommendation](../calibration-and-uncertainty/delivery-recommendation.md)). For the inventory the
equivalent limits are: several assets of one type behind one meter are reported as one total; assets
below the noise are reported as absent, not as small; and the cause of a step is not identified.

## 7. A proposal outline for NGED

**This outline is an option for the maintainer, not a commitment, and the effort figures are guesses
with a factor of two either way.** Each phase has a stop rule so that NGED can end the work with a
deliverable in hand.

| Phase | Deliverable | Effort (person-weeks) | Stop rule |
|---|---|---|---|
| 0. Known-answer test of the register-constrained inventory fit | The smallest test of the asset-inventory page, run on the six MW Primaries, written up as a pass or fail against its pre-registered predictions | 2 to 3 (about a day of one-core fits) | The decoy gets 1 MW or more at three or more of the six Primaries |
| 1. Inventory for the 16 Primaries, shown as inferred | One table per Primary: register rows, fitted season-year solar capacity with interval, any large engine's unit size and running-time distribution, discrepancies between register name and meter (the D8 and D14 kind), and the unregistered solar share against the three registers; one figure per asset | 6 to 10 | NGED cannot act on an inferred list, or the central asset register makes the list a cross-check |
| 2. Substation profile product | For each Primary and each of NGED's five representative days, a half-hourly profile of demand, solar, and each large asset from the fitted blocks, driven by a climatological weather year, in the layout of the published DFES profiles; a comparison against the DFES cluster profile on held-out months | 6 to 10 | The fitted profile does not beat the cluster profile on held-out months at most Primaries |
| 3. Scenario blocks | New connections from the register at their energisation dates; an assumed battery scheduler and engine equations labelled "assumed", with parameters checked where the asset is large; long-horizon scenario calibration against a block bootstrap | 8 to 16 | Scenario bands are no narrower than the block bootstrap at the same coverage |
| 4. Integration and what-ifs | Writing the profiles into NGED's load-set format for the Switch-Level Analyser's boundary nodes; a days-1-to-10 what-if that adds a dormant registered engine to the forecast, scored on known events | 6 to 12 | NGED's planners do not take the load set, or the what-if's simulated step misses the observed step by more than 30% on the known events |

**The smallest first deliverable is phase 0 plus the solar rows of phase 1.** The solar rows exist
in draft: experiment 673's per-Primary table with its register cross-check, and the probe's 2.4 to
3.9 MW at the six MW Primaries. Shown to NGED as "what we infer sits behind each substation and how
sure we are", with the 8-of-12 register shortfall and the D8 and D14 placement discrepancy, that
table is a deliverable NGED can react to in weeks, and their reaction answers [question
9's](#9-open-questions-for-the-maintainer) first two questions.

**What to ask NGED for, and what each answer changes.**

- The `MPAN to Substation Number` mapping's coverage of unregistered assets, and the connections
  team's own list for the 16 Primaries. If complete, phase 1 shrinks to the unregistered share.
- Running hours or metering of the engines registered under D14 and D5. If held, the engine block is
  fitted to records rather than inferred.
- Which representative day and which boundary nodes the Switch-Level Analyser reads for the trial
  area, and in what format. This decides whether phase 4 is a file export or a project.
- Whether NGED's planners want per-substation profiles at all, or whether the cluster profiles are
  judged adequate for primaries. A "no" removes phases 2 to 4.
- Which pipeline connections at the 16 Primaries NGED expect to energise in the next 2 years. This
  decides whether block 3 has anything to add.
- Whether NGED's reverse-power-flow and export-limit records can be shared, so the MVA fold and the
  summer-peak-generation day can be modelled with known signs.

**How it reuses the forecasting work.** Every block in phases 1 to 3 is a module that exists in
`nged-ml-research` (the demand GAM, the solar plant and capacity fit, the engine detector, the
battery estimator and templates, the injection harness, the event table), and the asset-inventory
page's missing piece, the joint fit, is the one new module. The scenario calibration reuses the
scoring code of the calibration theme, and the what-if reuses the lag-free forecast. The production
repository contributes the data contracts, the cleaned power table, the metadata table, and the
anonymisation rules; none of its serving path changes.

## 8. Counter-arguments and risks

**The strongest case against proposing this is that it spends the project's scarce people on a
planning product during the months the forecast has to ship, in a field NGED already have tooling
for, with a deliverable that cannot be validated against truth.** Each clause is a risk in its own
right.

- **Scope creep, and a simulator is never finished.** A block library invites one more block. The
  phase table's stop rules and the "fitted or assumed" label on every block are the defence, and the
  minimum set (demand, solar, new connections) is the scope to hold to.
- **Validation.** No ground truth exists below the Primary, the list is shown as inferred, and the
  only truth is injected. NGED's acceptance of an inferred list is untested, and the maintainer's
  answers record that NGED "cannot confirm the lists beyond the register" (asset-inventory page).
- **Competing with NGED's own tools.** PSS/E, the Switch-Level Analyser, the DFES profiles, Triton,
  PRIDE, and Fractal Flow all exist. A product framed as a simulator is a sixth tool; a product
  framed as two tables NGED's chain reads is an input. The framing decides whether NGED's planning
  team sees a colleague or a rival.
- **Trust.** The forecast's standing with NGED rests on a visible feature set and on scores. An
  inventory that names an engine where there is none, which free fits did four times, damages the
  forecast's standing too, because the same team delivers both.
- **The regulator may do it first.** Ofgem's asset-visibility decision and the Flexibility Market
  Asset Registration platform (due by the third quarter of 2027, aimed first at assets under 1 MW,
  [energy-forecasting
  review](https://openclimatefix.github.io/nged-substation-forecast/background/energy-forecasting-review/))
  both move towards registers that would supersede an inferred list for the assets they cover.
- **The forward-model forecast loses skill at short leads.** The discussion session judged that a
  forward-model forecast "would probably be less skilful than the current forecasts at short leads",
  and the explicit solar term lost 0.37, 0.25, and 0.27 points against the free neural network with
  ensemble weather ([experiment
  644](../../00644-pv-explicit-term-with-ens-forecast-weather/result.md)). Any suggestion that the
  simulator becomes the forecast should be withheld.
- **Training the forecaster on simulated series.** The maintainer calls this "a bit sketchy", and
  the evidence agrees: injected events raised error on ordinary rows as their share of training rows
  rose, although the 5% arm scored better than its reference on event-free rows ([experiment
  486](../../00486-lagged-network-injection-rate-and-schedule/result.md)).

**What the project would lose by spending the effort here instead of on the production port.** The
v1.0 release for January 2027 still needs the five delivery tables, the port of the neural networks
(the production repository "has no neural-network code and no `torch` dependency", [final
models](../../final-models.md)), the quantile output contract, the clamp and calibration steps, and
the handover workstreams
([roadmap](https://openclimatefix.github.io/nged-substation-forecast/roadmap/)). The research queue
holds paused lines with ranked options waiting for January: the load-shift detector's learnt scorer,
the per-Primary solar capacity trajectory, and the cloud-error test. Phases 0 to 1 of the proposal
cost about 8 to 13 person-weeks, which the advisor's guess puts at roughly one person's January to
March. If the port slips because of it, the forecast NGED were promised arrives late so that a
product NGED, so far as the repository records, did not ask for arrives early.

## 9. Open questions for the maintainer

- **Do NGED's planners want per-substation profiles, or are the DFES cluster profiles judged
  adequate for primaries?** A "no" removes phases 2 to 4 and leaves an inventory product; a "yes"
  makes the profile the lead deliverable and the inventory its explanation.
- **What does the `MPAN to Substation Number` file already tell NGED, and do the connections team
  hold a fuller list?** If NGED can list every registered asset per meter, the inventory's value is
  the unregistered solar share and the behaviour parameters only, and phase 1 shrinks by about half.
- **Is the proposal to be made before or after the January 2027 release?** Before, it competes with
  the port; after, phase 0 can run in January alongside the paused research lines, and the
  solar-only first deliverable could be shown in the first quarter.
- **Should the asset list be a Flexpectation delivery table or a separate agreement?** The five
  delivery tables were specified in a formal report and need NGED's agreement to change
  ([requirements](https://openclimatefix.github.io/nged-substation-forecast/background/requirements/)).
  A sixth table keeps one contract; a separate agreement keeps the forecast's scope clean.
- **Does the decoy rule apply to solar?** The asset-inventory page's open question. Every register
  name lists solar, so the decoy can only be tested for engines, batteries, and wind; if solar is
  what matters most, phase 0 can drop its engine and battery arms and reduce to the existing solar
  work, which cuts the test to days.
- **How much of the LLM's role is wanted in the first proposal?** If the matching and edit-proposal
  loop is in the proposal, phase 1 needs the comparison against the plain fit and the move log; if
  the LLM is kept to drafting the explanations, phase 1 is the convex fit alone and about a third
  shorter.
- **Which horizon is the proposal's headline: the DFES's 10 to 25 years, or days 1 to 10?** The
  maintainer said long horizons matter most. A long-horizon headline puts the product beside the
  DFES and needs NGED's planning team as the customer; a days-1-to-10 headline puts it beside the
  forecast and needs the control room, and the two customers would judge it by different evidence.
- **Is the project willing to ask NGED for the engine running hours, the export limits, and the
  reverse-flow records?** The asset-inventory page records no request to NGED for any of the three.
  Any of the three would turn an inferred parameter into a recorded one and would shorten the
  validation argument.

## What was searched

**Web searches run (standard mode), in order:** NGED DFES 2025 methodology; NGED Network Development
Plan and power-flow tools; behind-the-meter solar estimation from feeder net load; open-source
power-system simulators; GB DNO digital twins; the CREST demand model; NGED's Long Term Development
Statement; data assimilation and identifiability for aggregated loads; LLM agents calibrating
simulations; Ofgem asset visibility; Western Power Distribution load-flow tools; gas-engine dispatch
models; building-energy calibration criteria; UK Power Networks planning tools; NGED customer
behaviour assumptions; Elexon profile classes; NREL dsgrid. Two NGED PDFs were fetched and read in
full text (the DFES 2025 methodology and the NDP 2026 methodology). The searches found no GB
distribution-operator project inferring per-substation inventories from net telemetry, no
power-system LLM calibration study, and no identifiability analysis of behind-the-meter
disaggregation; each absence is a statement about one search.
