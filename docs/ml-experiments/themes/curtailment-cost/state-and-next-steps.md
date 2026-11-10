# Curtailment cost: state and next steps

**No forecast has been shown to curtail less at the bulk supply points, and on the six MW Primaries
the experiments read for this page rank the stack ahead of BG2 and the weekly XGBoost model.**
Curtailment is the capping of generators beneath a substation when its export would pass a limit.
The scoring code plans the caps from a forecast of the substation's export, then charges the energy
lost to the caps plus a penalty for export that stays over the limit. The code runs, reproduces a
hand-worked answer, and has scored several forecasts, but the pilot holds too few events to rank
forecasts at the headline limit. An option for the next step is to score the January 2027 delivery
of about 2,500 time series with the same code.

**Terms.** NGED is National Grid Electricity Distribution. A bulk supply point (BSP) is a substation
that steps 132 kV down to 33 kV or 66 kV, and a Primary is a primary substation. MW is real power,
and MVA is apparent power. GBP is pounds sterling. The stack, BG2 (the second best-guess recipe),
climatology, and the manual heuristic are defined in the [glossary](../../glossary.md). A node is a
substation scored together with the generators beneath it. The NGED baseline is the manual
heuristic's substation forecast with every generator at full capacity. A scaled limit is a fraction
(0.9, 0.75, or 0.5) of a substation's maximum export before 2025-07-02. An arm is one pair of
forecasts, one of the substation's export and one of its generators' output. Avoidable cost is an
arm's cost minus the cost of a perfect forecast. All costs are calibration to history, not savings.

## What works

**The metric code runs and reproduces a hand-worked answer exactly.** All eight quantities of a
made-up half-hour (a 45.0 MW limit and two generators) agree with the hand calculation ([Experiment
84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)). The code plans caps
from the 0.95 quantile of the export forecast and shares the cut among generators in proportion to
their capacity. The cost is an energy proxy: the generator energy cut, plus 20 times the energy left
over the limit, at GBP 2,000 per MWh. Experiment 84 varied the price and the cost form only, and
[Experiment 268](../../00268-lsc-curtailment-metric-v2-member-wise-ens/result.md) changed how the
planned cut is built.

**The metric accepts ensemble members.** Member-wise inputs reproduce Experiment 255 to within 2.0%
per arm ([Experiment 268](../../00268-lsc-curtailment-metric-v2-member-wise-ens/result.md)). Of the
stored farm-forecast rows, 34% have crossing quantiles, which a set of members cannot reproduce.

**On the six MW Primaries, the stack costs less than BG2 and the weekly XGBoost model.** At 0.5
times the maximum, BG2 costs GBP 7.35 million more than the stack (95% interval 1.6 to 14.9 million).
That figure ranks the two forecasts and is not money saved ([Experiment
277](../../00277-lsc-curtailment-metric-on-existing-forecasts/result.md)). The stack is cheaper than
the weekly XGBoost model at 4 of 6 MW Primaries at 0.9 times and at all 6 at 0.5 times ([Experiment
381](../../00381-simple-weekly-xgboost-demand-against-gam-stack/result.md)). The cost is undefined
at the eight MVA-only Primaries.

**The three final demand networks breach the 0.9 limit at one Primary only.** On July to October
2026 at the six MW Primaries (0.9 limit, 0.95 planning quantile), only D9 breaches the limit, at
about GBP 520 a day for the networks against GBP 484 for the manual heuristic. At the 0.75 limit,
D9 costs GBP 4,400 to 6,400 a day against GBP 5,600 for the heuristic, and the Candidate is lowest
([Experiment 620](../../00620-bakeoff-final-test-demand-networks-at-primaries/result.md)).

**The G4 band does not raise cost on event-free rows.** On injected load shifts, the combination of
A1 and the G4 band lowers the cost in the 7 days after the shift by GBP 124 per Primary-day
(Bonferroni interval -268 to -19). The 0.9-limit saving rests on one Primary, D16, and on the real
December 2025 pair D16's cost is almost unchanged ([Experiment
601](../../00601-nra-curtailment-cost-and-tail-calibration-of-assembled-candidates/result.md)).

## What does not work

**The pilot cannot rank forecasts at the 0.9 limit.** Boston and Skegness, the two BSPs of the
pilot, together have 19 half-hours beyond 0.9 times their maximum in the scored year (2025-07-02 to
2026-06-30), and no difference between arms is significant at the 5% level at that limit. At 0.75
times, with 46 half-hours, the stack costs GBP 2,440,720 more than the best guess (95% interval
96,371 to 6,686,043). At 0.5 times, with 323 half-hours, the best guess is GBP 10,802,520 cheaper
than climatology (-20,880,964 to -3,342,553), and its gap to the NGED baseline includes zero. We
have no evidence that NGED operates at 0.5 times, so that limit is a rehearsal ([Experiment
84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)).

**At 0.9 times, the best guess has the lowest avoidable cost of the six arms, and most of that cost
is breach.** Its avoidable cost is GBP 571,823, and 81% of that is breach: the forecast ordered 13.6
MWh of cuts on event half-hours against 35.1 MWh needed (Experiment 84).

**Fitting a whole distribution to a forecast plans far more curtailment than the stored forecast
does.** On 17 half-hours at one node, Student-t and skew-normal fits planned 911 and 1,364 MWh of
curtailment against 57.5 MWh for the stored BG2 forecast, with 33.7 MWh needed. Their avoidable
costs were GBP 2.36 million and 3.12 million against 618,171. The other recalibrations were not
separable from the stored forecast ([Experiment
97](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md)).

**Load-shift events do not separate forecasts by curtailment cost.** Across 33 events at the
six MW Primaries, no difference in avoidable cost during events is established, because 2 to 11
events per Primary account for the cost, and one Primary accounts for most of it ([Experiment
478](../../00478-load-shift-unpadded-curtailment-cost-and-after-event-slices/result.md)). Across all
rows, that experiment found the stack far cheaper than the weekly XGBoost forecasts at two Primaries,
but the arms carry different widening (1.15 on the arms of Experiment 383), so the gap mixes
widening with skill.

**No curtailment cost exists for the BSPs and GSPs of the final models.** No manual heuristic
baseline exists for them ([Experiment
669](../../00669-one-year-backtest-of-the-lag-network-at-bulk-and-grid-supply-points/result.md)).

## Where the pages disagree

**Experiments 83 and 97 disagree on the lower tail because they score different forecasts and
periods.** In Experiment 83, on the older forecasts of Experiment 48 pooled over 20 demand series,
coverage of the 0.05 quantile was 0.066 to 0.085, so the lower tail was too narrow. In Experiment
97, on BG2 for 16 Primaries at 12 to 36 hours, coverage was 0.035, so the lower tail was too wide
([Experiment 83](../../00083-curtailment-cost-and-tail-calibration-worked-example/result.md);
[Experiment 97](../../00097-cal1-post-hoc-recalibration-of-bg2-quantiles-tails/result.md)).
Experiment 97 scores the later forecast.

## Open questions

- **Which forecast curtails least, once about 2,500 series are scored?** Experiment 84 names this
  as its next question, and the same function scores any number of nodes.
- **How does NGED choose its operating limit, and how does it allocate caps among generators?** The
  idea that NGED curtails early to keep headroom is the maintainer's unconfirmed understanding, and
  the questions for NGED are collected in [issue
  #1318](https://github.com/openclimatefix/nged-substation-forecast/issues/1318).
- **Are the NGED baseline and the signs of the nodes right?** The baseline comes from the same
  unconfirmed understanding. Boston and Skegness are inferred as import-positive, Bicker Fen's sign
  is unconfirmed, and Walpole's is unknown, so the metric excludes both.

## Next experiments

**The theme's [research plan](plan.md) has no proposal waiting, so the options below come from the
"next step" sections of merged pages.**

1. **An option is to score the January 2027 delivery of about 2,500 series with the unchanged code.**
   This option comes from Experiment 84. It depends on the delivery, and it is the only option
   that could rank forecasts at the headline limit.
2. **An option is to store the 0.01 and 0.99 quantiles in every forecast set.** This option comes
   from Experiment 83. The stored sets of Experiment 48 stop at the 0.95 quantile, and later sets
   store 11 levels.
3. **An option is to judge any tail recalibration over at least 2 years of scored runs.** This
   option comes from Experiment 97, whose 9 scored months gave intervals of plus or minus 0.02 on
   the coverage of the 0.95 quantile.

**A gap, not a plan option:** curtailment cost has not been scored for the BSPs and GSPs, the eight
MVA Primaries, wind and solar farms, or the year-backtest files of the final models.

## What is parked, and why

**Load-shift event detection and unmetered photovoltaic capacity are paused until January, so no
curtailment work that depends on them is under way.** A ranking of forecasts waits on the delivery
of the larger series set, and the questions about NGED's practice wait on NGED's answers.

**Before using any cost from this theme, check these reporting rules** ([Experiment
84](../../00084-substation-limit-curtailment-cost-worked-example/result.md)):

- Label every cost "calibration to history, not a saving", because the perfect forecast still costs
  GBP 87,460 at 0.9 times.
- Report the 13% of generator-half-hours that had no forecast, because those generators were not
  capped.
- Report the unscored half-hours, which hold 14% of the excess energy at 0.5 times.
- Publish generator figures only as sums over at least 4 generators, to keep NGED's metered
  generators anonymous.
