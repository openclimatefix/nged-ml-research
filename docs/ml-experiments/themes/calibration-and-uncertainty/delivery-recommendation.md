# Delivering uncertainty to NGED's power-flow studies

**Our current recommendation is to deliver 11 stored quantiles per series and half-hour plus a
dependence table, a direct forecast of each bulk supply point's own meter, and a small set of
unlabelled scenarios.** The quantiles let NGED's own Python code rebuild marginal forecasts and
coherent paths. The direct forecast supplies the tail of the flow through a bulk supply point (BSP),
which a sum of node scenarios did not reproduce in any experiment we ran. This page collects the
findings of experiments
[434](../../00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts/result.md),
[436](../../00436-joint-scenarios-across-demand-wind-solar-and-leads/result.md), and
[625](../../00625-coherent-members-from-quantiles-k-sweep-and-path-checks/result.md) to
[631](../../00631-bsp-group-sum-with-metered-generators-against-direct-bsp-forecast/result.md) in
one place. It states the recommendation, the five questions NGED has to answer before the design is
fixed, and the corrections that later pages should carry. Every result rests on 10 calendar months
for the Primaries, 3 wind farms, and 6 solar farms, and on two BSPs for the BSP results. National
Grid Electricity Distribution (NGED) has not yet confirmed that any part of this design suits its
workflow.

## The recommendation

**Store 11 quantiles per series and half-hour, and let the client rebuild paths from them.** In
[experiment 628](../../00628-quantile-form-client-side-scenario-rehydration/result.md), 1,000 client
draws from the 11 stored quantiles (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95,
0.99) scored within 0.3% of the pipeline's 51 values on CRPS. The client extends the quantile
function past q01 and q99 with a Gaussian tail and clamps wind to 0 to 100% of capacity. Solar is
not clipped at 100%, because 2.3% of solar actuals exceed it. The client code is about 70 lines, and
one run's 11 quantiles for 25 series take 772 kB.

**Add a dependence table, chosen by family.** The table is 51 past observed paths for a Schaake
shuffle (rank-ordering independent draws to follow past paths) for the wind and solar farms.

- For the 3 wind farms and 6 solar farms, the Schaake shuffle scored 34.4% below independent draws
  on the variogram score (interval -42.5% to -26.1%, [experiment
  625](../../00625-coherent-members-from-quantiles-k-sweep-and-path-checks/result.md)).
- For the 16 Primaries, no coupling we tried scored measurably better than independent draws. The
  best, a shuffle of anomaly paths, scored -0.3% (-1.6% to +1.2%). We therefore recommend no
  coupling for the Primaries, and the ensemble ranks are worse there (+6.8%).
- A fitted Gaussian copula on its own scored 6.6% worse than the pipeline's version on the solar
  farms' variogram score. The pipeline's version also uses the weather ensemble's member ranks,
  which the quantiles do not carry. We did not test that explanation by ablation.

**About 20 paths are enough for the marginal forecast and cost 2% to 3% of the coupling score on the
farms.** In experiment 625, 20 paths cost 2.0% (wind farms) to 3.1% (solar farms) of the 51-path
variogram score and 1.0% for the Primaries, and 10 paths cost 5.2% to 11.1% on the farms. The CRPS
of 20 paths was within 0.5% of the 51-value CRPS in all three families. By the registered tests, the
farms needed K = 51 and the Primaries K = 20, so "about 20" is a rounding of mixed results, not a
registered pass.

**Deliver a direct forecast of each BSP's own meter beside any node scenarios.** Summing the 16
Primaries to each of two BSPs put 31% to 79% of actual export flows beyond the sum's q05 (nominal
5%), in every coupling ([experiment
629](../../00629-bsp-group-sum-of-nodes-against-direct-bsp-forecast/result.md)). Adding the 9
metered wind and solar farms and a time-of-day remainder fixed most of the export side (Boston
10.6%, Skegness 3.0%, for the Schaake shuffle) but pushed the import side to 14.3% and 32.6% beyond
the q95 ([experiment
631](../../00631-bsp-group-sum-with-metered-generators-against-direct-bsp-forecast/result.md)). The
direct forecast of each meter gave 4.9% and 7.2% beyond the q05 and 6.7% and 9.9% beyond the q95.
Coupled sums did beat independent draws on CRPS by 2.6% to 7.7% once the farms were in, so coupling
still matters for the node scenarios.

**If NGED will run only a few scenarios by hand, give it at least 10, chosen without labels.**
[Experiment 627](../../00627-scenario-subset-selection-from-51-coherent-paths/result.md) found that
cutting the 51 paths into K groups on one summary and drawing one path per group beat a random draw
on tail coverage in all 10 months. At K = 5 no rule held the 5th and 95th percentiles within 10% of
the range of the 51 paths. For a BSP's export limit, an observed daily peak export beat the largest
of 5 chosen paths on 12% of days (1.6% for all 51 paths) and the largest of 10 chosen paths on 6.3%
of days. A label such as "the low-wind scenario" does not carry from day to day or lead to lead:
under the Schaake shuffle a path's wind rank between days 1 to 3 and days 4 to 10 has a rank
correlation of 0.19.

**The bake-off exports store 11 quantile levels, and NGED's delivery tables use 13.** The 13
delivery levels (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98, 0.99) share
7 levels with the 11 stored levels. Delivering all 13 means re-predicting the stored forecasts at
the extra levels, which experiment 628 did not measure.

## Five questions for NGED

These are questions, not commitments. The answers decide which parts of the design are worth
building.

1. **How many scenarios does NGED run by hand?** A hand-run study in a graphical tool allows a
   handful of runs per element per forecast, which sets how few paths the delivery must offer.
2. **How many power-flow runs can NGED afford, and are they scripted?** A scripted study allows
   thousands of runs, in which case NGED can draw its own scenarios from the stored quantiles.
3. **Which BSPs, GSPs, or circuits matter?** Can NGED give the weight each node carries in the flow
   through each constrained element, or link its network model to our series identifiers?
4. **Does a direct forecast of the flow through a BSP, GSP, or circuit already exist on NGED's
   side?** Our direct forecasts of two BSP meters are the only ones we have scored.
5. **Should the delivery tables hold 11 or 13 quantile levels?** The stored forecasts hold 11, and a
   move to 13 needs the forecasts re-predicted.

## What we tried

- **[Experiment
  434](../../00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts/result.md)
  (`not supported`).** Re-ordering the stored demand forecast's 51 values by member temperature or
  by past observed paths did not beat independent draws across the 16 Primaries. Temperature ranks
  made the Primaries' paths 6.7% less coherent. The Schaake shuffle scored 3.5% lower across leads
  at days 1 to 10.
- **[Experiment 436](../../00436-joint-scenarios-across-demand-wind-solar-and-leads/result.md)
  (`partly supported`; joint block void, see the erratum on its page).** Across the farms, a Schaake
  shuffle scored 28% (wind) and 40% (solar) below independent draws, and the copula-corrected
  ensemble ranks 29% and 43% below. Across the Primaries nothing beat independent draws by more than
  0.7%. Tying scenario k to ensemble member k brought no gain.
- **[Experiment 625](../../00625-coherent-members-from-quantiles-k-sweep-and-path-checks/result.md)
  (`partly supported`).** The K sweep and the anomaly-path shuffle above. It also re-scored
  experiment 436's joint block and found no difference between couplings (within +0.8%).
- **[Experiment 626](../../00626-widen-the-tails-of-the-51-scenario-member-values/result.md)
  (`partly supported`).** A separate upper and lower factor brought demand's q05 and q95 exceedance
  to 4% to 6% in every lead band, and did not fix wind or solar. Demand's q90 was still exceeded for
  11% to 16% of actuals (nominal 10%), which points to a median that sits too low rather than tails
  that are too narrow.
- **[Experiment 627](../../00627-scenario-subset-selection-from-51-coherent-paths/result.md)
  (`partly supported`).** The subset results above.
- **[Experiment 628](../../00628-quantile-form-client-side-scenario-rehydration/result.md) (`partly
  supported`).** The client-side rebuild above. The tail rule changed CRPS by under 0.02%, and the
  stored upper tails were too narrow for any of the three tail rules to repair.
- **[Experiment 629](../../00629-bsp-group-sum-of-nodes-against-direct-bsp-forecast/result.md) (`not
  supported`).** Sums of the Primaries missed the BSP export tail. The twCRPS test could not rank
  anything, because neither BSP's meter reached the scaled export limit in the development months.
- **[Experiment
  630](../../00630-roster-upper-demand-tail-calibration-event-days-and-bias-correction/result.md)
  (`partly supported`).** The stored neural-network forecasts exceeded their q90 on 13% to 17% of
  event-free readings. A correction fitted on the first three months removed about a quarter to a
  half of the excess on the last two, because the bias changes with the season.
- **[Experiment
  631](../../00631-bsp-group-sum-with-metered-generators-against-direct-bsp-forecast/result.md)
  (`not supported`).** The farm-inclusive sums above. The cause of the import-side miss (the Primary
  forecasts' bias, the farms, or the remainder) was not separated.

## Corrections to carry into other pages

- **Do not repeat that the 51 member tails are "7.6% to 8.8% too narrow" at q95.** Experiment 436
  took the q95 with numpy's default quantile definition. With the Weibull definition (the k-th
  smallest of 51 values sits at level k/52), the q95 exceedance is 5.2% for demand, 6.5% for wind,
  and 5.7% for solar ([experiment
  626](../../00626-widen-the-tails-of-the-51-scenario-member-values/result.md)). The stored upper
  quantiles of the demand forecasts are still too low, as experiments 626, 628, and 630 show.
- **Do not repeat experiment 436's joint-block verdict.** Its scoring code mixed nodes and members,
  and experiment 625 replaced the joint-block scores. The within-family results of experiment 436
  are unaffected.
- **Do not write that demand has no mixture.** The bake-off's demand networks are mixture-density
  networks with 3 components, but the export stores only the mean of 5 seeds' quantiles, which is
  not any single mixture. The wind and solar mixture parameters were also never stored. Experiment
  628 concluded that mixture parameters were not needed to rebuild marginal forecasts or a
  Schaake-shuffle coupling from the 11 quantiles. Recovering the parameters would need a rebuild of
  the inference of experiments 612 and 613, which has not been done.
- **Do not describe BSP sums as validated.** Experiments 629 and 631 each rest on two BSPs and on a
  scaled export limit that was never reached, so the threshold-weighted score could not rank the
  sums.

## Open question

**How can coherent node scenarios be made to agree with the direct BSP and GSP forecast?** NGED's
power-flow study needs node injections, and the direct forecast supplies the flow through the
element. Experiment 631 shows that the two disagree in the tails when the node scenarios are summed.
A design that conditions the node scenarios on a quantile of the direct forecast was proposed in an
advisor review and has no experiment yet. We do not know whether NGED wants this, which is question
4 above.

## Recommended changes to the production docs

These are recommendations for a later pull request to the production repository, not commitments.
Each quotes the passage as it stands on `origin/main` of the production repository on 2026-11-10.
Where a passage depends on the state of the production code, which this research repository cannot
see, the entry says to check the code before editing.

### `docs/roadmap/delivery-tables.md`

- **Table 1's three-representation framing leaves out the dependence table, the direct BSP forecast,
  and the scenario subsets.** The page says: "The Milestone 1 report describes **three** ways of
  expressing uncertainty. We may deliver one or several of these: ... 3. **Ensemble of percentile
  forecasts** — per-member *and* per-percentile." Add a short section after Representation 3, headed
  "Coherent scenarios, direct flow forecasts, and scenario subsets", with this text: "A power-flow
  study needs one value per node and half-hour in each scenario, and the scenarios must move
  together as the nodes do in reality. We therefore also offer, as a recommendation pending NGED's
  answers: (1) the 11 stored quantiles per series and half-hour together with a dependence table,
  which is 51 past observed paths for a Schaake shuffle for the wind and solar farms, and no
  coupling for the 16 Primaries; (2) a direct forecast of the meter of each bulk supply point,
  because summing node scenarios did not reproduce the tail of the flow through a bulk supply point
  in any experiment we ran; (3) on request, 10 or more scenarios chosen without labels by
  stratifying the 51 paths on one summary." Link the three claims to experiments 625, 629, 631, and
  627.
- **Representation 1 describes scenarios as the NWP members.** The page says: "`ensemble_member` |
  `int8` | ID of the NWP ensemble member driving this forecast. ECMWF ENS has 51 members." and the
  Representation 3 table repeats the member identifier. Experiment 436 found that tying scenario k
  to ensemble member k scored the same as a Schaake shuffle across Primaries and wind farms and no
  better across solar farms, and that ensemble copula coupling overstated the correlation between
  Primaries (0.93 against 0.29 realised). Add a note: "A delivered scenario need not be one weather
  member's forecast. The scenarios built from the stored quantiles are numbered 1 to 51 with no link
  to any weather member, and no scenario carries a label that holds from one lead to the next."
  Check the code before editing the field description, because the code may still deliver one
  forecast per member.
- **Representation 2 lists 13 percentile levels, and the stored forecasts hold 11.** The page says:
  "`p1, p2, p5, p10, p20, p35, p50, p65, p80, p90, p95, p98, p99`". The stored forecasts hold 0.01,
  0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, and 0.99, which share 7 of the 13 levels.
  Add under the table: "The research forecasts store 11 levels, not these 13. The two sets share p1,
  p5, p10, p50, p90, p95, and p99. Delivering all 13 needs the forecasts re-predicted at the extra
  levels, and we do not yet know whether NGED needs them." Keep the 13-level table, and state the
  open question as a bullet, not as a decision.
- **Representation 2's note does not say how far to trust the tails.** The page says: "NGED is far
  more interested in the **tails** than the shoulders." Add: "On the 2025-09 to 2026-06 development
  months the stored q99 of the Primaries was exceeded by 1.0% to 1.6% of event-free actuals for the
  XGBoost forecasts and 1.9% to 3.1% for the neural networks, and the stored q90 by 13% to 17%, so
  the upper tail of demand is too low. Extending the quantile function past q01 and q99 with a tail
  rule does not repair this (experiments 628 and 630)."

### `docs/techniques/probabilistic-forecasting.md`

- **The page's closing section says the full pipeline is planned, not what we found about
  coherence.** The section "How this maps onto the project" ends: "**The full pipeline** ... a
  quantile-objective XGBoost emits Representation 3; the pooling recipe above derives Representation
  2 for delivery." Add a section before it, headed "Making scenarios coherent across sites and
  leads", with this content: "A quantile forecast describes each site and half-hour separately. A
  power-flow study needs scenarios whose sites and leads move together. In the experiments we ran, a
  Schaake shuffle of 51 past observed paths scored 34% below independent draws on the variogram
  score across 3 wind farms and 6 solar farms. No coupling scored measurably better than independent
  draws across 16 Primaries, and ranking the draws by the weather ensemble's members scored 6.8%
  worse there. About 20 paths lose 2% to 3% of the 51-path score on the farms. Summing node
  scenarios to a bulk supply point did not reproduce the tail of the flow through that point, and a
  direct forecast of the meter did." Link experiments 436, 625, 629, and 631.
- **The tail caveat quotes a literature finding but not our own.** The page says: "The mixture's own
  p1 and p99 may therefore need the same parametric-tail treatment before they can be trusted." Add
  after it: "In our own data, a Gaussian tail beyond q01 and q99 gave the best coverage past the
  stored knots (0.2% quantile: 0.20% to 0.45% of actuals, against 0.53% to 1.05% for clamping), and
  the tail rule changed CRPS by under 0.02% (experiment 628). A tail rule cannot repair stored
  quantiles whose upper tail is already too low."
- **The page does not say that the research forecasts are quantile forecasts stored per site and
  half-hour.** Its section "Turning 51 quantile sets into one: the pooling recipe" describes pooling
  51 members times K quantile levels. Check the code and experiments 425, 429, and 430 before
  editing: if the research default stores one set of 11 quantiles per site and half-hour (as
  experiment 628 reports), add a sentence "The research forecasts store one set of 11 quantiles per
  site and half-hour, from which the 51 scenario values are interpolated" and say which pipeline
  each recipe applies to.
- **The claim about Vincentization is untested here.** The page states as settled that
  level-by-level averaging discards the between-member spread. None of the experiments on this page
  tested it, so leave the passage as it stands and do not cite our results for it.
