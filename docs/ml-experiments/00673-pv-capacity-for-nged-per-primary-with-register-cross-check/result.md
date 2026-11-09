---
id: "00673"
slug: pv-capacity-for-nged-per-primary-with-register-cross-check
title: >-
  What unmetered solar capacity does the probe read behind each of the 14 Primaries, and how does it
  compare with the registered solar capacity inside each Primary's service area?
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  simple. Stored: one small private parquet of table rows, the figure sources, and this page. Designs:
  one defensible construction (stored probe tables joined to register totals computed inside each
  service-area polygon). Callers: the script reads and edits none of the stored outputs of
  experiments 267, 474, 479, 649, 662, and 671. Serving path and degradation rule: not touched. No
  fit and no GPU; CPU only, one process.
hypothesis: >-
  The calibrated probe capacities of experiments 649 and 671 are consistent with the registered
  solar capacity that lies inside each Primary's service-area polygon: at every Primary that has a
  probe interval, the 90th percentile of the probe is at or above the register floor, once the
  capacity that NGED already meters is removed from the floor.
prediction: >-
  Written before any register total was computed. (1) At the six MW Primaries the probe's 90th
  percentile is at or above the Feed-in Tariff floor at 5 or 6 of 6 (experiment 649 found 5 of 5
  where a floor existed, using postcode-district shares). (2) With the Embedded Capacity Register
  and the Renewable Energy Planning Database added to the floor, the probe is above the combined
  floor at fewer Primaries than with the Feed-in Tariff alone, because those two registers list
  larger sites that NGED often meters. (3) At the five non-folding MVA Primaries the linear-probe
  interval lies below the register floor at 2 or more of 5, because experiment 671's recovery
  factor at the MVA Primaries is uncertain and the registers list capacity of all sizes. (4) The
  register total inside the polygon differs from experiment 267's postcode-district share by more
  than 25% at 3 or more Primaries.
decision_rule: >-
  A descriptive product, so there is no verdict threshold. Each Primary gets one flag from the
  probe's 80% interval (10th to 90th percentile) and the register floor, fixed before any total is
  computed: "probe above floor" if the 10th percentile is at or above the floor; "overlaps" if the
  floor lies inside the interval; "below floor" if the 90th percentile is below the floor; "not
  comparable" where no probe interval is defined (D12 linear; D8, D22, D5, and D10 for the MVA
  conversion) or the floor cannot be built. The verdict is read from prediction (1): supported at 5
  or more of 6, partly supported at 3 or 4, not supported at 2 or fewer. No threshold for use in
  production is set.
benchmark: >-
  The Feed-in Tariff floor of experiment 267 (1 MW size cut, alternating-current basis), and
  experiments 474 R4 and 479 for the MW Primaries.
row_set_id: >-
  Not a forecast score. The stored probe outputs of experiments 649 (April to September 2025,
  2026-06 cutoff), 662, and 671 (2026-06 cutoff). Development data only; nothing dated 2026-07-01 or
  later.
metric: >-
  Capacity in MW of alternating-current output at the 10th, 50th, and 90th percentiles per Primary;
  registered solar capacity inside the service-area polygon per register; ratio of the probe's
  median to the floor; one flag per Primary.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Planned. This page will give NGED a table of unmetered solar capacity per Primary beside the solar
  capacity that three public and NGED-held registers list inside each Primary's service area.
---

# Plan: unmetered solar capacity per Primary, with register cross-checks, for NGED

**This experiment fits nothing. It joins the stored probe tables to register totals and writes one
page for NGED.** The page has one section per group of Primaries and a register cross-check for all
14 Primaries. The maintainer approved the page on 2026-11-09.

## Inputs

- **MW Primaries (D5, D9, D10, D11, D14, D16):** experiment 649's calibrated table (its
  `capacity_table.parquet`), raw beside calibrated, with the energy-basis rows.
- **MVA Primaries:** experiment 671's recommendation table and experiment 662's capacity table
  (linear probe and fold model, raw and calibrated). The floor-arm and zero-reactive-power-arm
  recoveries come from experiment 662's `recoveries_2026-06.csv`.
- **Registers:** the Feed-in Tariff installation report (Ofgem, 31 March 2025), the Renewable Energy
  Planning Database (Q2 2026, cached by the solar BMU census), and NGED's Embedded Capacity Register
  (August 2026). Each is totalled inside the Primary's `area_wkt` polygon in
  `data/NGED/metadata.parquet`.

## Reactive-power correction

NGED have said to assume power factor 1 for the substation meters, and MVA Primaries never read zero
because some reactive power is always present. Experiments 656, 662, and 671 simulated apparent
power with a reactive part of 0.25 times the median real power (power factor 0.97, the "PF arm").
The page reports 662's floor arm and zero-reactive arm beside the PF arm wherever they exist, names
the reactive model behind every number, and says that the conversion loss at the five non-folding
MVA Primaries is likely smaller under power factor 1 than the PF-arm figure. Nothing is refitted.

## Method for the register totals

- Feed-in Tariff has postcode districts and no coordinates. A district's capacity is shared to a
  Primary by the share of the district's delivery points inside the polygon (experiment 45's
  `fit_district_polygon_shares`). The page says this is a share, not a point-in-polygon total.
- The Embedded Capacity Register and the Renewable Energy Planning Database carry coordinates, so
  each solar row is placed by point-in-polygon against `area_wkt`. Technology filters and size cuts
  are stated per register.
- Double-counting is checked by matching the Embedded Capacity Register and the Renewable Energy
  Planning Database by name and distance, and the Feed-in Tariff cannot be matched row by row (no
  coordinates), so it is reported separately and not summed with the others.
- Capacity that NGED already meters (the metered solar generators in the metadata inside the
  polygon) is flagged, and subtracted from the floor where the match is unambiguous.

## Figures

(a) Probe interval against the register floor per Primary, all 14, MW and MVA marked differently and
report-only Primaries greyed. (b) S(u) example curves for D19 and D12 from experiment 671's stored
fits.

## Reviews

Sonnet code review before the numbers are quoted; Sonnet review of the numbers, the prose, and the
figures. No Opus.
