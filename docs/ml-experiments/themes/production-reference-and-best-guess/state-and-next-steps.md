# Production reference and best guess: state and next steps

**The BG2 best guess beats the stored production-style forecast, and a simple weekly-refitted XGBoost
model comes within 0.05 points of BG2 at the six MW Primaries.** Scores are the quantile score, an
estimate of the continuous ranked probability score ([CRPS](../../glossary.md#crps)), in points of
capacity. In a contrast, a negative value means the first forecast is better. A Primary is a
[primary substation](../../glossary.md#primary-substation). The six MW Primaries report signed
megawatts, and the eight MVA Primaries report apparent power in megavolt-amperes only
([glossary](../../glossary.md#mw-and-mva)). Where a result uses another set of Primaries, the
paragraph says so.

**The final models supersede the BG2 recipe as the design for the Primaries.** The [final models
page](../../final-models.md) documents the neural Candidate (P7+D50 with a prevailing head), which is
the Primaries design, together with the wind and solar networks. [BG2](../../glossary.md#bg2) is the
XGBoost reference of [Experiment
87](../../00087-bg2-best-guess-assembled-with-drop-one-ablations/result.md): a five-member blend
with a 14-day correction and an explicit solar offset at five Primaries. BG2 is part of the final
test. The [neural simplification series](neural-simplification-series.md) records which
simplifications the networks can take.

## What works

**The stored production-style forecast loses to the `manual_heuristic` at Primaries.** The
[`manual_heuristic`](../../glossary.md#manual_heuristic) forecasts each half-hour from the power at
the same weekday and time in earlier weeks. In [Experiment
13](../../00013-primary-demand-against-stored-baselines/result.md), the
[`xgboost_baseline`](../../glossary.md#xgboost_baseline) forecast (the production recipe, read from
a stored table of unconfirmed origin) had a mean absolute error of the median 1.69 points above the
`manual_heuristic` (99% interval +0.98 to +2.59; 16 Primaries). A gain over production alone
therefore overstates a new forecast, so every comparison needs the `manual_heuristic` beside it.

**The best guess beats the production-style forecast by 3.8 to 6.3 points.** [Experiment
33](../../00033-best-guess-architecture-vs-production/result.md) found 4.89 points for Primary
demand (98.33% interval -5.66 to -4.06), 3.81 for solar farms, and 6.27 for wind farms.

**BG2 beats the `manual_heuristic` for all four target classes.** In Experiment 87, the mean
absolute error of the median at Primaries is 7.43 for BG2 against 8.70, and the quantile score is
5.260 against 6.182.

**Most of the Primary gain comes from the [14-day
correction](../../glossary.md#14-day-correction).** In Experiment 87, the correction on a [frozen
XGBoost model](../../glossary.md#frozen-xgboost-model) lowers the Primary score by 1.38 of the 1.67
points that BG2 gains. At the four supply points the correction made the score 0.305 worse (0.060
to 0.594), so BG2 serves them without it.

**The three recent-power inputs are the part of BG2 that cannot be removed.** [Experiment
102](../../00102-simp1-simplifying-the-bg2-best-guess-before-production/result.md) found that
removing them made the Primary score 0.36 points worse once one substation's commissioning months
were cut, and the supply-point score 0.28 worse. The wind farm forecasts lost nothing measurable.
The experiment did not test a lag-free model with a level term.

**A plain weekly XGBoost model is nearly as good as the five-member blend.** [Experiment
379](../../00379-bg2-components-all-16-primaries-to-14-days/result.md) found it within 0.05 points
of BG2 at the six MW Primaries and within 0.02 at the eight MVA Primaries. Two of BG2's five
members (the monthly XGBoost and the pooled-by-type model) added nothing measurable.

**The weekly XGBoost model matches the GAM stack at most Primaries.** [Experiment
381](../../00381-simple-weekly-xgboost-demand-against-gam-stack/result.md) compares it with the
stack of generalised additive models. It was level with the stack at 9 of 14 Primaries and better
at 4 (D8, D12, D22, and D5). At the six MW Primaries it is 0.08 worse (-0.12 to +0.27), and worse at
D11 (+0.15) and D14 (+0.30), so one demand recipe does not suit both groups. Removing the Kalman
filter makes the stack 0.12 worse at the six MW Primaries
([Experiment 380](../../00380-gam-stack-ablation-and-jaggedness-at-16-primaries/result.md)).

**On the five development months, K1 beat BG2 by 0.32 points at 13 Primaries.** K1 is the neural
network of Experiment 620 with a band that widens while a load shift is detected. The contrast is on
rows with no labelled event ([Experiment
645](../../00645-k1-vs-stored-bg2-on-the-development-months/result.md); interval -0.42 to -0.22).

**Default 3 matches BG2 at the six MW Primaries.** [Experiment
378](../../00378-primary-demand-default-3-at-all-16-primaries-under-demand-v1/result.md) scored
research default 3 (the GAM-stack demand forecast) at -0.12 against BG2 at the six, and 0.16 worse
at the eight MVA Primaries without D8 and 0.63 worse with it.

## What does not work

**On July to October 2026, BG2 beat K1 by 0.71 points, and the lead is smaller than it looks.** In
[Experiment 633](../../00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network/result.md), BG2
scored 4.23 and K1 scored 4.94 (Experiment 620). The 4.23 is BG2 fitted once. The lead halves on
event-free rows to 0.38 (3.91 against 4.29). BG2 also reads power 3.5 hours old, against at least 24
hours for K1. The current reading is Experiment 645: on the development months, away from the
transfers, K1 was the better of the two, so BG2's lead in the window is not a standing advantage.

**A network fitted once did no better than the `manual_heuristic` on that window.** In [Experiment
620](../../00620-bakeoff-final-test-demand-networks-at-primaries/result.md), K1 scored 4.94 against
4.85 for the `manual_heuristic` (not separable), and K1 lost at four Primaries (D14, D15, D16, and
D18). Experiment 620 names load transfers at D14 to D18 and a single fit as candidate causes. The
weekly-refit arm was skipped because its estimated 3.7 hours or more exceeded the 3-hour cap.

**Removing days with load shifts reverses the K1 result.** [Experiment
637](../../00637-ara-detector-rescore-of-the-bakeoff-final-test/result.md) removed the days a frozen
detector flags (6.1% of rows, 119 Primary-days, 102 of them at D18). K1 minus the `manual_heuristic`
moved from +0.10 (-0.03 to +0.22) to -0.15 (-0.26 to -0.05), with CRPS 4.29 against 4.44. The
Candidate moved from +0.15 to -0.13, and K3 stayed worse (+0.20). D14 still loses, because the
detector flags only 5 of its 112 days.

**The GAM stack does not transfer unchanged to the eight MVA Primaries.** In [Experiment
252](../../00252-gam-stack-at-eight-more-primaries/result.md), a solar term at each monthly refit
gave point estimates at or below BG2 at five of the eight, while D8, D12, and D22 stayed worse.
[Experiment 360](../../00360-bg2-member-drop-one-at-d8/result.md) found that BG2's lead at D8 is
more likely to come from what its members learn than from blending (one substation, one year).

**The q05 to q95 band covers too few readings in the final-test networks.** It covered 0.73 to 0.77
of readings in Experiment 620. BG2's 95% quantile covers 93.2% at the six MW Primaries against a
nominal 95% ([Experiment 379](../../00379-bg2-components-all-16-primaries-to-14-days/result.md)).

## Open questions

- **It is unknown whether the 14-day correction and the weekly refit hold on a second year.** The
  recipe was chosen on one validation year (July 2025 to June 2026), so its gains are probably
  optimistic.
- **It is unknown why K1 lost ground after July 2026.** Experiment 633 names the lags (3.5 hours
  against 24 hours) and BG2 following load shifts: BG2's median follows 0.56 of the D18 transfer
  and K3's follows 0.33.
- **The best training source for solar and wind farms is unresolved.** Experiment 87 could not rank
  the ensemble-trained model against the ERA5-trained mixture at 6 solar and 3 wind farms. ENS is
  the European Centre for Medium-Range Weather Forecasts ensemble, and ERA5 is its reanalysis
  ([glossary](../../glossary.md#ens-era5-and-cams)).
- **The cause of the MVA-only skill gap is unresolved.** [Experiment
  56](../../00056-mva-only-primaries-skill-against-mw-primaries/result.md) found 10.5 percentage
  points of skill against the `manual_heuristic` (15.0% for the MW group, 4.5% for the MVA group; 95%
  interval 2.1 to 17.5). [Experiment
  384](../../00384-mva-only-primaries-what-is-special/result.md) tested 16 Primaries and showed no
  hypothesis; the only difference was a weaker price correlation (0.08 against 0.17).

## Next experiments

**Each option below comes from the [research plan](plan.md) or from a merged result's "What to try
next".**

1. **An option is to score the lag-enabled network against BG2 with equal 3.5-hour lags.** It comes
   from Experiment 633's advice and removes the lag asymmetry.
2. **An option is to refit K1 and the Candidate every 14 days through July to October 2026.** It
   comes from Experiment 620's advice, to test whether ageing weights explain the narrow bands.
   [Experiment
   667](../../00667-one-year-backtest-of-the-two-output-demand-network-at-primaries/result.md), with
   monthly refits over a year, is partial evidence.
3. **An option is to run the weekly-refit BG2 arm that Experiment 633 skipped.** The arm was skipped
   because of its estimated cost.
4. **An option is a boosted-tree sunshine term in the stack against research default 3.** The plan
   lists it as proposed with medium priority, beside a hard switch on the member share of cold
   ensemble members (D3b).

## Parked

**Wind and solar refit schedules wait, and so does load-shift event work.** Load-shift detection and
unmetered PV capacity are paused until January. The [neural simplification
series](neural-simplification-series.md) holds the wind and solar serving settings.
