# The event-table line in plain words

**Reading only the power data, we find about two thirds of the megawatts of long, deep load shifts,
but we name both substations of a transfer for only about a third of its megawatts. Nothing we tried
reached the 0.60 share set for adopting a method, so the results describe single substations well
and do not yet say where a transfer went.** The line ran from experiment 653 to experiment 682,
which tested a reconstruction by edge flows after the graph models of experiments 672 and 676.
NGED's event labels are incomplete, so every count of real events on this page is a lower bound.

## What the line was trying to do

A load transfer moves part of one Primary substation's demand onto neighbouring Primaries for hours
or days. The line asked whether power readings alone can build a table of such events, with the
donor Primary, the recipient Primaries, the start, and the end. The tests inject fake transfers,
made by one shared generator ([experiment
660](../../00660-shared-generator-of-injected-load-transfer-events/result.md)), into the real power
history of 15 Primaries, because the real record holds almost no verified long transfers.

Five terms recur. **E1** is the event table of [experiment
653](../../00653-edge-state-event-table-penalised-segmentation/result.md), which cuts each Primary's
residual (the demand forecast error) into steps. **E1-J** is E1 plus the recipient search of
[experiment 670](../../00670-event-table-recipient-search-on-six-cell-grid/result.md), which looks
at each connected Primary for a step of the opposite sign shortly after a large downward step.
**Found** means an injected event has a decoded event at the right Primary, with the right sign and
start. **Paired** means the decoded donor and recipient are the right two Primaries. Every share is
weighted by megawatts. A noise unit is a Primary's usual difference between successive 6-hour means.
A *deep* shift is 8 noise units or more and a *shallow* one is 2 to 8. A *long* shift lasts over 7
days (or has no end), a *medium* one 1 to 7 days, and a *short* one 1 day or less.

## What works

- **Long and deep shifts are found.** E1 finds 0.67 of the injected megawatts of shifts longer than
  7 days and 8 noise units or more deep ([experiment
  665](../../00665-event-table-recall-by-length-and-depth/result.md)).
- **The search for the recipient triples the pairing.** E1-J pairs 0.29 of the long and deep
  transfer megawatts and 0.25 of the medium and deep ones, against 0.09 for E1 (experiment 670).
  Rearranging the connections at random removes the gain, so the gain comes from the real
  neighbours.
- **Searching from the rise as well as the fall helps a little more.** E1-J-sym, which also looks
  for the donor of each unpaired rise, pairs 0.38 of the long and deep megawatts against 0.29, and
  0.32 of the medium and deep megawatts against 0.25, but adds 442 decoded transfers for 63 more
  correct ends ([experiment
  680](../../00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/result.md)).
- **Choosing edge flows by a fixed cost per change does not help.** E1-F, which accepts a pair of
  steps when one flow explains both better than two separate steps, pairs 0.33 of the long and deep
  megawatts, below the symmetric search, and decodes 2990 transfers for 329 correct ends. A graph
  that connects any two Primaries pairs nearly as much as the real neighbours, so the edge-flow line
  closes at 15 Primaries ([experiment
  682](../../00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model/result.md)).
- **The real record supports the method only thinly.** The one verified transfer, between Primaries
  D5 and D16 in December 2025, is paired in the right direction, but the decoded starts are late.

![Paired and found shares per cell for E1, E1-J, and E1-J on a rewired
graph](../../00670-event-table-recipient-search-on-six-cell-grid/figure_recall_by_cell.svg)

## What does not work

- **Pairing stays below 0.40, whatever was added.** The graph neural network of [experiment
  672](../../00672-graph-model-that-finds-the-donor-of-a-load-transfer/result.md) paired 0.08. The
  donor-probe hybrids of [experiment
  678](../../00678-donor-posterior-hybrid-for-load-transfer-pairing/result.md) paired 0.36 at best,
  and the edge-state neural network of experiment 676 paired no more than the hybrids.
- **The donor is the bottleneck.** [Experiment
  675](../../00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe/result.md)
  found that, of the recipient megawatts that E1-J located but left unpaired, 0.87 belonged to
  transfers whose donor E1-J had not found. A pretrained encoder made the donor probe worse, and a
  logistic regression on the node inputs did as well as any neural encoder.
- **Short events are not paired.** Transfers of a day or less are paired for 0.08 (deep) and 0.03
  (shallow) of their megawatts, and ending the added donors after 6 hours instead of 2 days did not
  change that (experiment 676, post hoc).
- **False transfers are common.** Of the transfers E1-J decodes, 69% match no injected end
  (experiment 670).
- **Long and deep events cannot be checked on the real record.** None of the 77 shifts that NGED's
  logs record as lasting more than 1 day falls in the long and deep type.

## What the table is good for

- **Descriptive tables per Primary.** The table gives each Primary's steps, their sizes, and their
  durations, including the event-free stratum (time more than 14 days from any logged or verified
  event).
- **Training and testing data for other detectors.** The injected worlds have known starts, ends,
  and partners.
- **A first screen of candidate transfers for NGED to check.** The table's long and deep calls can
  go to NGED for confirmation, because the real record cannot confirm them.

## What a next step would need

- **More Primaries, or the adjacency of the whole fleet.** At 15 Primaries the recipient is often
  unmetered or below its noise, and 8 of the 15 Primaries report apparent power (MVA), which has no
  sign and so cannot show which way power flows ([experiment
  674](../../00674-mva-sign-labels-soft-and-hard-pv-prior-day-labeller/result.md) found the sign
  label from a day labeller unusable at those Primaries).
- **A more realistic injector.** The generator is checked against 49 real windows of power history
  and 2 verified transfers.
- **More verified events.** The 77 logged shifts longer than 1 day include none in the type where
  the methods work best.

## Experiments in the line

| Experiment | Question in one line | Verdict | Headline |
|---|---|---|---|
| [653](../../00653-edge-state-event-table-penalised-segmentation/result.md) | Can a penalised segmentation build the event table? | Not supported | Finds about two thirds of injected megawatts; pairs 0.18 |
| [660](../../00660-shared-generator-of-injected-load-transfer-events/result.md) | Can one generator of injected transfers look real enough? | Supported | A classifier cannot tell the fake events from real ones (0.55) |
| [665](../../00665-event-table-recall-by-length-and-depth/result.md) | How much does E1 find, by length and depth? | Partly supported | 0.67 of long and deep megawatts |
| [670](../../00670-event-table-recipient-search-on-six-cell-grid/result.md) | Does a recipient search let E1 pair transfers? | Not supported | 0.29 against 0.09, short of 0.60 |
| [672](../../00672-graph-model-that-finds-the-donor-of-a-load-transfer/result.md) | Does a graph neural network pair better than the search? | Not supported | 0.08 against 0.29 |
| [674](../../00674-mva-sign-labels-soft-and-hard-pv-prior-day-labeller/result.md) | Can a day labeller give the sign at apparent-power Primaries? | Inconclusive | Misses 54 of 59 real reverse-flow days |
| [675](../../00675-pretrained-graph-model-for-load-transfers-part-1-diagnostics-and-frozen-probe/result.md) | Why are recipients unpaired, and does pretraining help the donor? | Not supported | Donor missing for 0.87; pretraining worse |
| [678](../../00678-donor-posterior-hybrid-for-load-transfer-pairing/result.md) | Do a probe's donor alarms raise the pairing? | Not supported | 0.36 against 0.29 |
| [676](../../00676-pretrained-graph-model-for-load-transfers-part-2-edge-state-network/result.md) | Does an edge-state neural network pair more than the hybrids? | Not supported | Pairs 0.32 against 0.36 (best hybrid); a copy without links pairs as much |
| [680](../../00680-event-table-reconstruction-from-edge-flows-part-1-symmetric-search/result.md) | Does also searching from each unpaired rise for its donor pair more? | Partly supported | 0.38 against 0.29 (long and deep); 442 added transfers |
| [682](../../00682-event-table-reconstruction-from-edge-flows-part-2-change-cost-model/result.md) | Does choosing edge flows by a fixed cost per change pair more? | Not supported | 0.33 against 0.29 (long and deep); all-pairs graph 0.32 |
