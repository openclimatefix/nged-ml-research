# Switching events

**A switching event moves part of a substation's load to a neighbouring substation for a while, and
the experiments that detect, inject, or remove such events carry the label LSC (load shift screen).**
This page says what a switching event looks like in the time series, why the project wants to detect
the large ones, and what the detector should output. The detector experiments are listed on the
[LSC detectors theme page](../themes/lsc-detectors/index.md).

## What a switching event is

- **A switching event switches a circuit from one substation to another.** The recipient substation
  gains the circuit's load, and the donor substation loses it. The experiments ignore that a circuit
  carries its own daily shape, and treat the event as a transfer of a constant amount of power
  (unchecked).
- **A switching event is temporary.** The two substations return to their old levels when the
  circuit is switched back. Events last from hours to months (unchecked).
- **Switching events can overlap.** For example, 1 MW may be switched away from substation A from
  Monday to Friday, and a further 0.5 MW from Wednesday. The total switched power is then 1 MW, 1.5
  MW, 1 MW, and 0 (unchecked).
- **A circuit can go to several recipients, and some recipients are not in the dataset.** In those
  cases the detector sees no partner (exp 653 to 692; 15 Primaries, injected events; provisional).
  Of the 66 hand-filtered real transfers, 21 have no metered receiver (docs).

## How a clear event looks in the time series

The clearest real event seen so far shows all five features below. Most events show fewer of them.

- **A temporary level shift** that starts and ends on a single half-hour.
- **The shift moves day-time and night-time demand by (almost) the same amount.** A shift of
  demand-shaped size, or one that appears only by day, points to something other than a switch.
- **The shift is mirrored at the partner substation**, roughly equal and opposite, and the start
  and the end fall on the same half-hours at both substations.
- **Both substations return to their old levels afterwards.**
- **The sum of the two substations does not change**, so the bulk supply point (BSP) above them does
  not notice the event.

## What the detector should find

- **The project cares about deep and long events more than shallow and short ones.** Demand noise
  hides small events, and the experiments do not expect to see them. An event under 4 noise units
  is found essentially never (exp 732; D8, simulated events; provisional).
- **Not every level shift is a switching event.** Some substations have permanent level changes,
  for example a new connection or a meter change. The normal-running-arrangement (NRA) forecast
  must follow those shifts. The NRA forecast must ride through a switching event, which is a
  transfer of load to a neighbouring substation (unchecked).
- **The ideal output is, for each half-hour, a distribution over the power that switching events
  have added to or removed from the substation.** The distribution must allow overlapping events and
  must be able to say "unknown" (unchecked).

## Why the project wants the events

- **Use 1: interpreting lagged power features.** If a lagged power feature lies inside a switching
  event, the forecast must try to undo the event, to give an NRA forecast (unchecked).
- **Use 2: scoring the NRA forecast.** The evaluation removes the half-hours inside switching events
  from the time series (unchecked).
- **Use 3: reporting events to NGED.** NGED cannot easily extract the switch instructions from their
  operational systems, so NGED would like the project to detect switching events from the time
  series and report them (unchecked).
- **The measured gain of use 1 is small.** Perfect knowledge of the events improved the forecast by
  about 1.5 pinball points around events and by about 0.005 over a year (exp 208, 212, 226 to 332;
  mixed designs; provisional). A band head with no detector matched a mask near shifts (exp 297;
  provisional).

## What labels exist

- **NGED's switching log is incomplete and noisy.** It holds start, end, series, a power estimate,
  and a sign. The estimate and the sign are not reliable, and the log does not give the partner.
  Most logged events are far too small to see in the data (exp 580; 184 logged events; established).
- **The log shows 184 events, of which 23 last 7 days or more.** Only 1 of those 23 appears as a step
  in the data (exp 580; provisional). The hand-filtered set holds 66 real transfers, of which 35
  last over 1 day (docs).
- **No labels will exist for the scaled-up NGED area.** A detector for that area must work without
  labels (unchecked).

## What the experiments found

- **Rule detectors find about half of the real transfers over 1 day.** The rule RO-NC found 18 to
  21 of 35 (exp 146, 302; real, one area; provisional). Recall on injected events overstated recall
  on real events (0.770 against 0.357; docs).
- **A neural network trained on synthetic transfers alone lost to the rule.** It found 7 of 35
  against 18 (exp 189; provisional). The 732 detector found 7.7% of the simulated steps of 5 points
  or more (exp 732; D8; provisional).
- **A one-day window hides the cues of a long event.** The permanence cue needs more than 14 days
  (exp 714; 16 Primaries, injected bank; provisional), and a human sees the clear event only in a
  month of data.
- **The donor is the bottleneck.** Pairing donor and recipient never passed 0.40 of the moved power
  (exp 665 to 692; injected events; provisional).
- **Training on real data is awkward because real data holds an unknown number of unlabelled
  events.** An injected event lands on top of events that nobody has labelled (unchecked). A
  realistic simulator of substation demand with total control over events is a large piece of work
  that may be tackled later (unchecked).

## Ideas under discussion

- **A long-context neural network.** A convolutional network would read about 6 months of the
  target substation and its neighbours, and output the switched-power distribution for the central
  month. It might also read the lag-free demand estimate, to remove seasonal signals, although that
  hurts as much as it helps at some substations (unchecked).
- **An unsupervised or weakly supervised detector** that uses the cues that events are temporary
  and usually mirrored at a neighbour (unchecked).
