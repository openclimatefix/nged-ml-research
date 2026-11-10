# Final models: architectures and training regimes for the production port

!!! warning "Outline only"

    This file is the plan for the page. Each section lists the merged pages it will be written from.
    The prose follows in the same pull request.

**Planned location.** `docs/ml-experiments/final-models.md`, with a nav entry beside the Research default
pages. The page is for the maintainer who ports the models into `nged-substation-forecast`.

## Planned sections

1. **Overview and reading guide.** One table: forecast type, series, model, output, post-processing,
   headline score. Sources: experiments 666, 667, 668, 669.
2. **Series identifiers and units.** The `time_series_id` to D-label mapping (`00521-*/site_map.csv`),
   MW and MVA series. Sources: 667, 668, 669.
3. **Demand at the 16 Primaries.** Sources: 667 and 668 (Candidate, inputs, offset, three-step
   schedule), 681 and PR 353 (extra-lag prevailing head), 652 and 647 (survival hazard, mixture
   quantiles), 684 (truncated Gaussian), 683 and 688 (calibration and winter median error), 687
   (detector features, not adopted), 677 (pending slot).
4. **Demand at the BSPs and GSPs.** Sources: 669, 668, 633, 622.
5. **Wind.** Sources: 666, 668, 659.
6. **Solar.** Sources: 666, 668, 659, 684 (night gate), PR 342 (clamp).
7. **Post-processing shared by all types.** Clamps (PR 342, 684), walk-forward band calibration
   (683, README of the backtest folder), what is not recalibrated.
8. **Production-port notes and open questions.** Exact reproduction list, unvalidated parts,
   disagreements between pages and which page is current. Unmetered solar capacity (673, 679) as
   context only.

Each section holds the same five blocks: architecture, training regime, post-processing, scores,
port notes. Every choice links to the experiment that decided it.
