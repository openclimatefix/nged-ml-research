# Evaluation protocol for metered solar farms, version 1

**This page fixes how every experiment on the metered solar farms is scored.** It is provisional: it
copies the demand protocol's structure and changes only what a solar farm needs. An experiment
records `protocol: solar-v1` in its front matter. A change to any metric, fold, series list, slice,
interval, or tolerance makes a new version, and the superseded text moves to
`evaluation-protocol-archive/` as `solar-v1.md`.

| Version | Date | Change |
|---|---|---|
| `solar-v1` | 2026-10-28 | First version, provisional |

## What is scored

- **Series:** metered solar farms A to F, each on its own row. NGED's metered generators are never
  named, and a farm's time series is never published with its name or identifier.
- **Rows:** daylight half-hours, scored year 2025-09 to 2026-08, with the 6-month replicate of
  [Metered solar farms research default 1](research-default.md).
- **Folds:** unchanged from the research default's row set. Contiguous-month folds for solar are a
  later version.
- **Lead range:** 12 to 36 hours is the scored range today. NGED's horizon is 1 to 10 days, so every
  result states that it was scored at 12 to 36 hours only, until a version scores days 1 to 10.
- **Forecasts compared:** the forecast under test, `manual_heuristic`, `climatology`, and the
  XGBoost reference of experiment 243 with the floors of experiment 224.

## Slices

**Each metric is reported over the slices below, and every slice row shows its half-hour count.**
The demand-only slices (winter nights, cold nights, Economy 7, Christmas, bank holidays) do not
apply.

| Kind | Slices |
|---|---|
| Lead time | 12 to 24 hours and 24 to 36 hours |
| Condition | All daylight half-hours; by month; by sun elevation; clear-sky and cloudy half-hours |
| Series | Every farm on its own row |

**No MVA or signed-MW split applies.** The metered farms report megawatts.

## Metrics, reporting, and adoption

**The metrics, interval method, reporting rules, adoption conditions, and known-answer tests are
those of the frozen [`demand-v1`](evaluation-protocol-archive/demand-v1.md)**, with the changes this
page states. Linking the frozen copy means a later change to the demand protocol does not change
this protocol. The scored year 2025-09 to 2026-08 and its 6-month replicate are those of Metered
solar farms research default 1, and the held-out period for solar starts on 2026-09-01.

| Item | Value for solar |
|---|---|
| Headline metric | The quantile score over 11 levels, in points of capacity |
| Headline band | 12 to 36 hours (days 1 to 10 once a later version scores them) |
| Groups tested for harm | All farms pooled, and each farm on its own row; no MVA or signed-MW split |
| Tolerances | 0.02 points for the headline, 0.05 for the 0.01 and 0.99 tails and for each lead band, 2% relative for twCRPS |
| Minimum gains | 0, 0.02, and 0.03 points for the three tiers of `demand-v1` |

**A family with few series and few runs will rarely pass the gain test, and the rule does not lower
the bar for it.** A change that cannot reach the bar stays a candidate, and the maintainer may judge
it by a descriptive report instead.

## What this page does not cover

- A physical lock that stops research sessions reading the held-out actuals.
- Logging scores to MLflow, which a follow-on pull request adds.
