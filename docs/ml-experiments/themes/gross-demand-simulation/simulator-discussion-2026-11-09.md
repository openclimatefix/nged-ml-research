# The simulator discussion of 2026-11-09

**This page is a lightly edited transcript of a discussion session, not a reviewed result.** Gross
demand is demand before the embedded generation behind a substation is netted off. On 2026-11-09 the
maintainer asked a Discuss session (an interactive Claude Code session used for discussion, which
runs no experiments) whether a simulator of gross demand at substations would be useful, whether it
could be built, and whether the simulator could itself be used as a forecast. The session answered
from merged pages and stored outputs and ran no fits. The numbers in the record were not re-checked
when the discussion took place. The [table of checked numbers](#numbers-checked-against-merged-pages)
below says which ones were checked against merged pages on 2026-11-10.

## Take-home message

**The session judged that a simulator with exact labels would be most useful for evaluating
forecasts.** It judged labelling real data the cheaper first step. It also concluded that labelling
alone cannot give the exact truth needed to score a forecast under two sets of rules. Those two sets
of rules are the normal running arrangement (the electricity network as configured when no load has
been transferred between substations) and the prevailing arrangement (the configuration in force on
the day). The maintainer's starting point was that a forecast's score, and even the ranking of
forecasting methods, depends on which set of rules is used, and the telemetry is not always in the
normal running arrangement. A simulator that puts in every load transfer and every generator itself
would know the correct labels for both sets of rules.

**What was asked.**

- Would a perfect simulator of net demand at substations, with controls for every input, be useful?
- Is a good-enough simulator buildable, and with what methods?
- Could the simulator, with its parameters inferred from real substation data, be used as an
  explainable forecast that can look months ahead and add a generator it has never seen?
- How does heavy labelling of real data (load transfers, solar capacity, large generators, with
  hindsight and past weather) compare with building a simulator?

**What the session concluded.**

- **Evaluation is the strongest use.** Exact labels would let a forecast of the normal running
  arrangement and a forecast of the prevailing arrangement each be scored against its own truth.
- **Training data is the weakest use.** In experiment 486, training with more artificial load-shift
  events raised forecast error on ordinary rows.
- **A first simulator would be assembled from parts the repository already has:** the demand
  generalised additive model (GAM) with its calendar, temperature, heating, holiday, and clock-change
  terms; the physics solar and wind plant models; the injected-event generator; and the fitted
  per-substation properties of the substation cards. A generative model of the residual, a model of
  how often load transfers happen, and a fixed rule for when each battery charges and each engine
  runs would be new.
- **A bottom-up model of customers and appliances is not worth building first**, because the
  session found no appliance, tariff, or behaviour data per substation in the project's files.
- **A simulator fitted to the metered record is circular** when it is used to test methods that
  share its assumptions: the test then checks the simulator's own assumptions. The session proposed
  two defences: take the truth from independent sources (real metered farm output injected into a
  Primary), and judge the simulator by whether the existing forecasting methods keep their ranking
  on simulated series.
- **The simulator and a forward-model forecast would be the same code**, run with chosen parameters
  or with inferred parameters. Its forecast skill is bounded by how well the hidden components can
  be separated from net power, and it would probably be less skilful than the current forecasts at
  short leads.
- **Labelling first, simulator later.** Labelling has no realism gap and improves every score the
  project already computes. A peer session on graph approaches to load-transfer detection reported
  that its detectors find about two thirds of the megawatts in injected long load shifts, and name
  both substations for under a fifth of transfer ends. Those labels would therefore be biased
  towards large, long, one-sided events, and could not score forecasts under both sets of rules.

**What was left open.**

- The session did not establish which use the maintainer has in mind. Known-answer testing and
  scenario testing need different simulators.
- The session proposed three cheap first experiments: a realness classifier on a first simulator,
  rank transfer of existing forecasting methods, and known-answer recovery of an injected farm. None
  was started. Their thresholds were the session's first guesses.
- Which labels beyond load transfers a labeller should produce first. The session's view was
  reverse flow (power flowing from the Primary back up to the 33 kV network) at the eight MVA
  Primaries, which are metered in apparent power without a sign, then large-generator steps, then
  solar capacity.

**The maintainer then asked for an alternative to strict disaggregation to be assessed.** Strict
disaggregation splits every half-hour into the contribution of each asset. The alternative fits a
list of assets and a few parameters per asset, and is assessed on [An asset-inventory
simulator](asset-inventory-simulator.md).

## Numbers checked against merged pages

**Most numbers in the record match the merged pages; three need a correction.** "Checked" means the
number was found on the merged page named, on 2026-11-10. "Not re-checked" means no merged page was
found that states it.

| Claim in the record | Source | Status |
|---|---|---|
| Injected-event recall of about 6% (1 to 2 noise units) and 23% (2 to 4 noise units); 0 of 5 and 0 of 3 real events found | Stored output of [experiment 660](../../00660-shared-generator-of-injected-load-transfer-events/result.md) | Checked against the stored file. The result page has since merged: its classifier cannot tell injected from real event windows (area under the curve 0.55) |
| The capacity fit recovers 0.75 to 0.88 of an injected farm | [Experiment 129](../../00129-differentiable-physics-pv-capacity-known-answer-test/result.md), [experiment 157](../../00157-differentiable-physics-joint-pv-capacity-known-answer/result.md) | Checked, with a correction: experiment 129 reports a median real-farm recovery of 0.78 to 0.86. The 0.88 is experiment 157's quote of experiment 129, and 0.75 to 0.87 is experiment 157's refit on its own rows |
| About 0.17 of the recovery is lost to the demand terms and to the fixed solar shape | [Experiment 231](../../00231-pv-capacity-audit-where-recovery-goes/result.md) | Checked (0.173) |
| The plant shape, not the sunshine, explains the gain from a nearby farm's output | [Experiment 269](../../00269-pv-capacity-gain-decomposition/result.md) | Checked |
| No candidate solar shape passes the placebo test; a cloud-load term makes the intervals honest | [Experiment 474](../../00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals/result.md) | Checked against the page's interim take-home |
| Neighbour lag encoders trained on synthetic shifts raise CRPS by 0.21 points of capacity | [Experiment 567](../../00567-neighbour-lag-encoders-synthetic-load-shifts-in-pretraining-and-fine-tuning/result.md) | Checked |
| Experiments 486 and 567 show that injected events raise error on ordinary rows | [Experiment 486](../../00486-lagged-network-injection-rate-and-schedule/result.md), experiment 567 | Checked, with a correction: experiment 486 finds the harm as the share of injected training rows rises, and its 5% arm scores better than its reference on rows with no labelled event; experiment 567 reports that the harm on rows with no labelled event is not shown (+0.14, interval -0.04 to +0.27) |
| The sum over a donor and its recipients still steps by about half the donor's step; 15 events at the six Primaries | [Experiment 459](../../00459-load-shift-conservation-check-donor-and-recipient-sum/result.md) | Checked |
| About half the worst unlabelled days are sharp-edged blocks, and neighbours do not explain them | [Experiment 175](../../00175-dp-unlabelled-day-shapes/result.md) | Checked |
| At D8 the XGBoost model's lead comes from origin lags | None found | Not re-checked |
| The penalised segmentation finds 0.64 of injected megawatts and pairs 0.18 of transfer ends; it matched 8 of 77 logged shifts; 246 real events | [Experiment 653](../../00653-edge-state-event-table-penalised-segmentation/result.md) | Checked |
| The recurrent neural network finds 0.48 to 0.52 of injected megawatts; a classifier separates its injected events from real ones at area under the curve 0.91 | [Experiment 658](../../00658-edge-state-event-table-neural-network/result.md) | Checked |
| Of 27 mirrored sustained pairs in the telemetry, 1 matches a label | [Experiment 580](../../00580-verified-load-shift-event-set-and-label-audit/result.md) | Checked |
| A rewired graph scored the same as the real graph in the graph layer | [Experiment 651](../../00651-graph-layer-over-electrical-neighbours-in-the-demand-network-part-1/result.md) | Checked (2.452 for the rewired graph and for the real-graph arms) |
| The shoulder-hour dip tells the sign change up to export about 3 times the night import; D12 exports about 5 times | [Experiment 508](../../00508-mva-shoulder-hours-and-d8-d22-dips-solar-or-load-shift/result.md) | Checked |
| A metered farm's meter outage does not reappear in D10's power; a known solar step is found in size and to within a month, with invented steps at D10 and D11 | [Experiment 206](../../00206-dp-meter-regime-steps-e1-farm-outages-in-primary/result.md), [experiment 207](../../00207-dp-meter-regime-steps-e2-step-capacity-known-answer/result.md) | Checked |
| The gas-engine bands of experiment 264 are a component a simulator could reuse | [Experiment 264](../../00264-dp-gas-engine-asymmetric-bands-at-d8/result.md) | Checked, with a correction: the page's verdict is not supported for the engine model, because its engine mixture also fits a running engine at two substations that have none |
| Experiments 618, 624, 649, 656, 660, and 662 had no merged page | The merged pages on 2026-11-09 | Correct when said; all six pages have merged since |
| The cost estimates (CPU-hours and GPU-hours) and the thresholds of the proposed experiments | The session's own guesses | Not re-checked |

## How the record was made

**The record below was produced by a script and edited only where the script says.**
`scripts/transcript_to_record.py`, beside this page, reads the session's transcript and keeps, in
order, the maintainer's messages, the session's replies, and the messages exchanged with the peer
session. Tool calls and tool results are dropped. The three tool results that carry findings the
replies rely on are each replaced by one line naming the page or file they came from. The script
removes paths under the home directory, socket addresses, and session identifiers. It replaces the
older anonymous labels of NGED's metered generators with the current ones (Wind A to C, Solar A to
F, Battery A, and Gas A), and stops with an error if an older label or an email address survives.
Headings inside a reply are pushed down below the entry headings, a bold-only line gains a colon,
and lists gain the blank lines and indents the docs renderer needs; the text is then rewrapped to
100 characters with `scripts/lint/reflow_docs.py`. The first entry is the instruction that launched
the session. The maintainer's messages were dictated and are kept as spoken. The record uses
"switching event" where the session did; elsewhere the project now writes "load shift".

## The record

**The verbatim record is not on this page yet.** Running the transcript script on the session's
transcript needs a permission the worker session that wrote this page did not have, so the
maintainer decides whether the record is added. The take-home and the checked numbers above stand
without it.

<!-- record-start -->
<!-- record-end -->
