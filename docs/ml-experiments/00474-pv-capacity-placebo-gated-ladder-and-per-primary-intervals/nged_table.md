# Unmetered solar capacity at each Primary (NGED table)

Season-year July 2025 to June 2026, in megawatts. Capacities come from the
pooled solar-shape model driven by the mean of ERA5 and CAMS irradiance, with two
cloud terms; D12 uses the fit through the meter.

| Primary | Status | AC capacity, MW (10th / 50th / 90th) | DC capacity, MW (10th / 50th / 90th) | Confidence |
|---|---|---|---|---|
| D2 | MVA-only: fit to the apparent-power reading, reactive power assumed zero | 6.7 / 6.9 / 7.9 | 8.2 / 9.0 / 10.3 | coverage 93% / 93%, placebo 0.51 |
| D3 | MVA-only: fit to the apparent-power reading, reactive power assumed zero | 4.0 / 4.2 / 4.7 | 4.9 / 5.4 / 6.2 | coverage 93% / 93%, placebo 0.51 |
| D4 | MVA-only: fit to the apparent-power reading, reactive power assumed zero | 0.6 / 0.8 / 1.1 | 0.7 / 1.0 / 1.4 | coverage 93% / 93%, placebo 0.51 |
| D5 | export limit: the clip-free solar term does not represent it | 3.1 / 3.1 / 3.5 | 3.7 / 4.0 / 4.5 | coverage 93% / 93%, placebo 0.51 |
| D8 | MVA-only: fit to the apparent-power reading, reactive power assumed zero | 8.0 / 8.6 / 9.7 | 9.7 / 11.1 / 12.8 | coverage 93% / 93%, placebo 0.51 |
| D9 | pinned capacity too high in the fit through the meter | 2.1 / 2.2 / 2.4 | 2.6 / 2.8 / 3.1 | coverage 93% / 93%, placebo 0.51 |
| D10 | signed MW reading | 2.8 / 2.9 / 3.3 | 3.4 / 3.8 / 4.2 | coverage 93% / 93%, placebo 0.51 |
| D11 | signed MW reading | 3.1 / 3.2 / 3.6 | 3.8 / 4.1 / 4.7 | coverage 93% / 93%, placebo 0.51 |
| D12 | reverse flow; fit through the meter (physical fleet shape), bootstrap of about 30 resamples | 6.6 / 7.4 / 8.2 | 8.2 / 9.4 / 10.6 | recovery tests used the physical fleet shape; coverage 75% / 0%, placebo 0.25 |
| D14 | signed MW reading | 3.6 / 3.7 / 4.3 | 4.4 / 4.9 / 5.6 | coverage 93% / 93%, placebo 0.51 |
| D16 | pinned capacity too high in the fit through the meter | 2.9 / 3.1 / 3.4 | 3.6 / 4.0 / 4.5 | coverage 93% / 93%, placebo 0.51 |
| D18 | history only from 2025-03-01; report-only | 1.8 / 1.8 / 2.0 | 2.1 / 2.4 / 2.7 | coverage 93% / 93%, placebo 0.51 |
| D19 | MVA-only: fit to the apparent-power reading, reactive power assumed zero | 1.7 / 1.8 / 2.0 | 2.1 / 2.3 / 2.7 | coverage 93% / 93%, placebo 0.51 |
| D21 | MVA-only: fit to the apparent-power reading, reactive power assumed zero | 1.8 / 2.0 / 2.2 | 2.3 / 2.5 / 2.9 | coverage 93% / 93%, placebo 0.51 |
| D22 | MVA-only: fit to the apparent-power reading, reactive power assumed zero | 1.4 / 1.5 / 1.6 | 1.7 / 1.9 / 2.1 | coverage 93% / 93%, placebo 0.51 |

**Confidence column.** Coverage is the share of injected metered-farm capacities that the
80% interval contained in tests, first without and then with a cloud-following load.
Placebo is the 95th percentile of the share of a fake farm (a day-of-year profile with
another day's anomaly) that the fit found. The placebo test fails when it exceeds 0.20, so
it failed on every row. Status terms: MVA-only means the reading is apparent power with no
sign; the other Primaries have signed readings. D12 shows its season-year 2024 value
(July 2024 to June 2025), the latest the fit through the meter was run for.
