# Production reference and best guess: research plan

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 252 | Does the best-guess stack score within 0.05 points of BG2 at the eight Primaries with no stack forecast? | High (cards) | Nothing | Planned |
| 353 | How well does Primary demand research default 3's stack forecast D8 when the MVA meter's fold is modelled? | High (maintainer asked) | Nothing | Planned |
| 379 | What does each component of BG2 contribute at all 16 Primaries, out to 14 days, under `demand-v1`? | High (maintainer asked 2026-10-28) | Nothing | Planned |
| 360 | Which of BG2's five Primary members carries its lead over the GAM stack at D8? | Medium (maintainer agreed 2026-10-27) | Nothing | Planned |
| 378 | How does Primary demand research default 3, ported to all 16 Primaries, score against BG2, the manual heuristic, and climatology under demand-v1? | High (maintainer asked 2026-10-28) | 377's stored forecasts | Planned |
| 333 | Does a pre-trained time-series foundation model (Chronos-2), used zero-shot with or without weather covariates, come close to Primary demand research default 3? | Medium | Nothing | Planned |
| 380 | Which stages of the GAM stack could go for a small, measured loss, and is its gap to BG2 larger at the Primaries with less regular demand? | High (maintainer asked) | Nothing | Planned |
| 384 | What is special about the eight MVA-only Primaries that the GAM stack trails BG2 there? | High (maintainer asked 2026-10-28) | Nothing | Finished: no hypothesis shown; within the MVA-only group the gap follows the step rate |
| 396 | Which of about 35 changes to the weekly XGBoost demand recipe (holiday columns, winter weather columns, hyperparameters, ensemble-summary inputs, a split by lead) survive a few-fold screen? | High (maintainer asked 2026-10-30) | The manager's fit slot; five-variable ensemble extraction for tier B | Planned |
| 397 | Does a plain mixture density network match the simple XGBoost model for Primary demand, and does attention over the inputs help it? | High (maintainer asked 2026-10-30) | Slot from the manager | Finished: the plain network matched XGBoost, attention did not help (gate inert) |
| 419 | Do the solar summary, the sunny flag, and a different row-weight half-life, which helped the XGBoost model, also help the pooled mixture density network? | High (maintainer asked 2026-10-30) | Slot from the manager | Planned |
| 457 | Does giving 425's arm A the out-of-fold GAM stage-1 forecast, as an input or an additive base, lower CRPS at the eight MVA-only Primaries? | High (maintainer asked 2026-11-01) | Nothing | Finished: the GAM as an input lowers CRPS by 0.036 at the eight (equal to the minimum detectable effect); as an additive base it raises CRPS by 0.291 |
| 468 | How do the best wind, solar, and demand forecasters behave hour by hour and event by event, and where do they fail? | High (maintainer asked 2026-11-01) | Injected grid: experiments 457, 462, 463, 467 | Part A and real events finished; injected grid pending |
| 481 | How many fit seeds does the first-release neural forecaster need for Primary demand and wind before averaging more stops helping? | High (manager) | Nothing | Planned |
| Proposed | Boosted-tree sunshine term in the stack against Primary demand research default 3 | Medium | Nothing | Proposed |
| Unnumbered | Pre-registered hard switch on the member share of cold ensemble members (D3b) with the replicate-year cold nights pooled, against Primary demand research default 3 | Medium | Nothing | Proposed |

## Planned experiments

<!-- planned-table -->
