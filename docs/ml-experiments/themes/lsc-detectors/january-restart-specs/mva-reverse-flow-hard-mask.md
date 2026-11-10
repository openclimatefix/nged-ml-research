# Spec: MVA reverse flow, sign label and the hard-mask arm

**This is a copy of the sections of a Sonnet research answer of 2026-11-09 that the January
restart needs, trimmed of conversation.** "Evidence" means a merged page of this repository, and
"inference" means the answering agent's own tabulation of the cleaned power table before
2026-07-01. The answer's survey of earlier experiments and its table of cases against the working
assumption are left out; the experiment numbers cited below lead to them. The zero-fit experiment
in the last section has not run. NGED's labels in the table below are the `Reverse Power Flow?`
column of a private file. They are a hint, not truth.

## Short answer

- **The working assumption is mostly right but not always.** The one real PV-driven case with a
  signed meter, D10, reverses only in April to August, 09:00 to 15:00 UTC (evidence for the
  direction, inference: 231 of 231 negative half-hours). At the eight MVA Primaries the best
  evidence is the same shape at D12.
  Counter-cases exist: one signed-MW series is negative at every hour of the day, and D8 has a gas
  engine (Gas A) behind it.
- **A hard high-PV mask is an established precision gate that costs recall.** Experiment 417's
  A+H arm cut false reverse flow from 12.0% to 3.6% and cut recall from 0.77 to 0.37.
- **The eight cannot be calibrated where it matters.** Experiment 508: nothing calibrates above an
  export of about 3 times the night import, and D12 is about 5.3 times.
- **Recommended first experiment:** a zero-fit, CPU-only test of a per-Primary masked day labeller
  (last section).

## What NGED says, and where it conflicts with our data

**Source.** The only record is `data/NGED/substation_hierarchy_2026-10-15/Monitor Direction.csv`
(private; README says received 2026-10-15 from NGED), column `Reverse Power Flow?`. There is no
confidence column and the `Comments` column is empty. I found no written record of NGED saying the
label is only a hint; that statement is in the brief.

| Series | Label |
|---|---|
| Marsh Lane (D8) | Yes |
| Stickney (D12) | Maybe |
| Langrick (D22) | Maybe |
| Warth Lane (D2), Sleaford Road Boston (D3), Ingoldmells (D4), Chapel St Leonards (D19), Alford (D21) | No |
| MW series and the 2 MVA wind farms | blank (MW series: directional monitoring "Yes") or "No" (wind farms) |

Also, `docs/background/network.md` (this repository) says 10 sites are metered in apparent power
(the eight plus two wind farms), and "one site, and possibly two more, have shown reverse flow on
sunny days"; it gives no source. Experiment 79 records that NGED cannot supply real and reactive
power at the eight.

**Conflicts with our data.**

- **D8 "Yes".** Experiment 370: L0 finds no reverse flow (share at most 0.006). Experiment 506:
  dips to 0.26 points on at least one sunny day. Experiment 508: 55 of 71 dip days flagged engine,
  not separable from solar or load transfer. NGED's "Yes" may refer to the engine, not to PV.
- **D12 "Maybe".** Our data are far stronger than "Maybe" (3 independent fits, ratio 5.3, 656 fold
  map flags D12 midday in all six months). The one method that fails to see it is the 10:00 to
  15:00 dip statistic (506), because the crossing is in the shoulders.
- **D22 "Maybe".** 13 dip days, and the cloudy-day test does not support solar (508: 0 cloudy dips
  of 122 days, Fisher p 0.12). Fits find about 0% to 3.7% reverse. Unresolved.
- **The five "No".** Consistent. D2, D3, D4 have lowest readings of 4.9 to 5.4 MVA (152); D19
  0.92 and D21 1.6 MVA, both far above zero given their PV estimates.

## Known-answer routes and calibration limits

**Routes that exist.**

- **Hide the sign at six MW Primaries (D5, D9, D10, D11, D14, D16)** by taking the magnitude
  (reactive floor 0, 2, 4 points, or power factor 0.97), injecting a metered farm (Solar A to C) at
  levels 0.5 to 8 times the median sunny-midday reading, leaving the farm out in rotation
  (79, 358, 408, 412, 417, 420, 506, 508).
- **Real reverse flow with a signed meter:** D10 (58 crossing days), D5 (5 days); 130 real
  half-hours in 417's count.
- **D12 itself has no truth.** Closure against the bulk supply point cannot resolve the sign (152:
  coefficient 1.02, variance cut 0.7%; 506: only 58% to 71% of the children are metered).
- **Metered solar farm nearby (inference, not run as a test).** `docs/background/network.md`
  shows D12's MVA peaks absent on 3 and 4 May when a nearby metered solar farm generated less. No
  experiment scored this as a label.
- **D8's engine:** 44 days of the engine's own recorded output (experiment 194).

**Calibration limits.**

| Limit | Number |
|---|---|
| 508 dip statistic: usable (area at least 0.8, lower end above 0.6) | export-to-import ratio below 1.5, midday 0.82; 1.5 to 3, shoulders 0.85 and 0.88 against midday 0.51; above 3, not estimable (99.4% and 97.4% of days cross zero) |
| 508 D12 | ratio 5.3, shoulder dips as common on cloudy days (sunny-to-cloudy 1.05, 1.01 to 1.10); not calibrated |
| 506 power factor 0.97 | area 0.838; threshold flags 34% of no-sign-change days; reactive power fills the dip |
| 508 selection | only 22% of April to September days qualify, because recorded farm output is missing in 20% to 63% of shoulder half-hours |
| 417 | gate false reverse 12.0%; zero-crossing labeller area 0.96 at level 2.0 but 0.58 at level 1.0 |
| 427 | labeller trained on high-demand Primaries fails at low-demand D5 (61.7% false reverse) |
| D12's export-to-import ratio | 5.27 in 508 (81.6 over 15.6 points in 506's May to July clear days); beyond the calibratable range of about 3 |

## Data a hard-masked label would need, and what is on disk

| Need | On disk | Caveat |
|---|---|---|
| Power readings, unflagged | `data/NGED/cleaned_power_time_series.delta` (33 series) | cut at 2026-07-01 |
| CAMS shape `u` and clear-sky index `k` per Primary | `data/ml-experiments/00412-*/work/cams/D*.npz` (16 Primaries, 121,248 half-hours each) | span unchecked; by my arithmetic it runs to about the end of July 2026, so cut at 2026-07-01 |
| Extended CAMS and ERA5 | `00620-*/inputs/cams_demand_points_extended.parquet`, `00607-*/work/era5_features.parquet` | extended to 2026-10-31 for the final test; read only before 2026-07-01 |
| Sun elevation per Primary | `00508-*/work/elevation_D*.npy` | |
| Capacity table for the eight (p10, p50, p90, AC and DC) | `00474-*/work/table_R4_pooled_table_era5_cams_mean_cloudterm.parquet`; `nged_table.md` | reactive power zero, placebo failed at every row (0.25 to 0.51), so circular at the eight |
| Capacity from other methods | 649 (six MW only), 656 and 662 (eight, provisional, one cutoff) | D2 differs: 6.9 vs 11.3 MW |
| Capacity from outside the telemetry | `00129-*/work/registers.parquet` (five MW Primaries only) | none for the eight; experiment 506 advises obtaining one |
| Effective capacity | `data/effective_capacity` | dated 2026-09-05 |
| Day-level dip tables and frozen thresholds | `00506-*/work/part1_*.parquet`, `00508-*/work/partA_*.parquet`, `partA_thresholds.json` | thresholds frozen on the six |
| Monitor Direction labels | `data/NGED/substation_hierarchy_2026-10-15/Monitor Direction.csv` | private; labels only |
| Clear-sky mask | `k` in the 412 arrays; 152 used the Haurwitz clear-sky value | no stored per-Primary mask table |
| Explicit PV estimate per Primary and half-hour | not on disk as a table | build as `C_p50 x u`; the neural PV module was not adopted (644 per 649) |

## Recommended first experiment

**First experiment (CPU only, zero fits): the per-Primary masked day labeller on known answers.**

- **Mask.** Per Primary, a half-hour is in the mask only if CAMS shape `u` is at least 0.1 (the
  sun-edge shoulders 508 found best; 0.3 would lose them) and the capacity gate holds: the
  Primary's 90th-percentile PV (474 R4) times `u` exceeds the dark-hour reading. The gate is
  per Primary because that series shows a global PV mask is wrong where non-PV export exists.
- **Label.** Inside the mask, flag a day by 508's shoulder minimum (thresholds frozen on the six in
  ratio bins) and by experiment 417's W-shape score, and label as reverse flow the half-hours
  between the two minima on flagged days. Both rules exist already, so no new method is built.
- **Known-answer test.** Injection of Solar A to C into D5, D9, D10, D11, D14, and D16 at levels
  1, 2, and 3 with leave-one-farm-out, plus the no-injection placebo at the five clean Primaries,
  plus real D10. Score day-level area under the curve, half-hour recall at a false-reverse rate of
  at most 5%, and the placebo rate, by export-to-import bin as in 508.
- **Then apply to the eight** and report a table of flagged days by Primary, sunny against cloudy,
  with no claim about truth. Check each Primary's mask coverage: share of its low (below 5 points)
  readings that the mask excludes.

**Prediction that could fail.** At export-to-import 1.5 to 3, the masked labeller reaches
half-hour recall of at least 0.6 at false-reverse of at most 5%, and its no-injection placebo rate
is at most 1% at each of D5, D9, D11, D14, and D16, with thresholds frozen on the other Primaries.
On the eight, flagged days exceed 10% of sunny April to August days at D12 only, are at most 5% at
D8 and D22, and are 0 at D2, D3, D4, D19, and D21. A prediction written with these numbers can be
wrong at either end, and would fail 417's bars if copied.

**Strongest reason it may fail.** At low-demand Primaries (D5 and D10 mean sunny-midday reading
12 to 16 points against 38 to 53 elsewhere) the dip is a normal low-demand day: 427's labeller
called 61.7% of D5's sunny middays reverse with nothing injected. At power factor 0.97 reactive
power fills the dip (506: 34% false-flag). Above export of 3 times night import the method cannot
be calibrated, and D12 is at 5.3 times. In short: the mask removes night and cloud false calls but
not the low-demand sunny-day false calls, and no known answer exists at D12's export level.

**Not CPU, but higher value.** Signed MW for a few weeks at Stickney from NGED (79's advice,
repeated in 79's lessons) would settle D12 without a fit. I found no record that it was asked.
