# Delivering uncertainty to NGED's power-flow studies

**National Grid Electricity Distribution (NGED) needs three deliverables for its power-flow studies:
11 stored quantiles per series and half-hour with a dependence table, a direct forecast of the meter
of each bulk supply point (BSP), and a small set of unlabelled scenarios.** The quantiles let NGED's
own Python code rebuild marginal forecasts and coherent paths. A BSP is the substation that feeds a
group of primary substations (Primaries). The direct forecast supplies the tail of the flow through
a BSP, which a sum of node scenarios did not reproduce in any experiment we ran. This page collects
the findings of experiments
[434](../../00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts/result.md),
[436](../../00436-joint-scenarios-across-demand-wind-solar-and-leads/result.md), and
[625](../../00625-coherent-members-from-quantiles-k-sweep-and-path-checks/result.md) to
[631](../../00631-bsp-group-sum-with-metered-generators-against-direct-bsp-forecast/result.md) in
one place. The page states the recommendation, five questions for NGED, and the corrections that
later pages should carry. The results on scenario coupling rest on 10 calendar months of Tuesday
runs for 16 Primaries, 3 wind farms, and 6 solar farms. The BSP results and the demand-tail results
of experiments 629 to 631 rest on 5 development months, and the BSP results on 2 BSPs. NGED has not
yet confirmed that any part of this design suits its workflow.

Terms: K is the number of paths in a scenario set. q05 and q95 are the forecasts at the 5% and 95%
levels. CRPS (continuous ranked probability score) scores one series' marginal forecast. The
variogram score scores how well a scenario set's dependence between series and leads matches
reality.

## The recommendation

**Store 11 quantiles per series and half-hour, and let the client rebuild paths from them.** In
[experiment 628](../../00628-quantile-form-client-side-scenario-rehydration/result.md), 1,000 client
draws from the 11 stored quantiles (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95,
0.99) scored within 0.3% of the pipeline's 51 values on CRPS. The client extends the quantile
function past q01 and q99 with a Gaussian tail and clamps wind to 0 to 100% of capacity. Solar is
not clipped at 100%, because 2.3% of solar actuals exceed it. The client code is about 70 lines, and
one run's 11 quantiles for 25 series take 772 kB.

**A Schaake shuffle helps the wind and solar farms and does nothing for the Primaries, so the
dependence table is chosen by family.** For the wind and solar farms, the table is 51 past observed
paths, which a Schaake shuffle uses to rank-order independent draws. For the Primaries, the table is
empty.

- For the 3 wind farms and 6 solar farms, the Schaake shuffle scored 34.4% below independent draws
  on the variogram score (98.33% interval -42.5% to -26.1%, [experiment
  625](../../00625-coherent-members-from-quantiles-k-sweep-and-path-checks/result.md)).
- For the 16 Primaries, no coupling we tried scored measurably better than independent draws. The
  fitted copula scored 0.7% lower (-1.5% to +0.2%) and a shuffle of anomaly paths 0.3% lower (-1.6%
  to +1.2%), and both intervals include zero. Coupling by the weather ensemble's member ranks scored
  6.8% higher, which is worse. We therefore recommend no coupling for the Primaries.
- A fitted Gaussian copula on its own scored 6.6% worse than the pipeline's version on the solar
  farms' variogram score. The likely reason is that the pipeline's version also uses the weather
  ensemble's member ranks, which the quantiles do not carry. We did not test this reason by
  ablation.

**About 20 paths score 2% to 3% worse than 51 paths on the farms' variogram score, and 20 paths
passed the registered test only for the Primaries.** In experiment 625, 20 paths scored 2.0% (wind
farms) to 3.1% (solar farms) worse than 51 paths on the variogram score and 1.0% worse for the
Primaries, and 10 paths scored 5.2% to 11.1% worse on the farms. The CRPS of 20 paths was within
0.5% of the 51-value CRPS in all three families. By the registered tests, the farms needed K = 51
and the Primaries K = 20, so "about 20" is a rounding of mixed results.

**Deliver a direct forecast of each BSP's own meter beside any node scenarios.** Summing the
Primaries beneath each of 2 BSPs (9 at Boston, 7 at Skegness) put 33.9% to 79.0% of actual export
flows below the sum's q05 (nominal 5%) for the K1 demand network, in every coupling ([experiment
629](../../00629-bsp-group-sum-of-nodes-against-direct-bsp-forecast/result.md)). Adding the 9
metered wind and solar farms and a time-of-day remainder reduced the export-side miss to 10.6% at
Boston and 3.0% at Skegness for the Schaake shuffle, but the import side then had 14.3% and 32.6% of
actual flows above the q95 ([experiment
631](../../00631-bsp-group-sum-with-metered-generators-against-direct-bsp-forecast/result.md)). The
direct forecast of each meter gave 4.9% and 7.2% below the q05 and 6.7% and 9.9% above the q95.
Coupled sums did beat independent draws on CRPS by 2.6% to 7.7% once the farms were in. Coupling
therefore still matters for the node scenarios.

**If NGED runs scenarios by hand, offer a choice of at least 10 paths, chosen without labels; a
hand-picked 5 did not cover the tails.** [Experiment
627](../../00627-scenario-subset-selection-from-51-coherent-paths/result.md) found that ranking the
51 paths on one summary (for example the peak Primary demand), cutting them into K groups, and
drawing one path per group beat a random draw on tail coverage in all 10 months. At K = 5 no rule
kept the 5th and 95th percentiles of the chosen paths within 10% of the range of the 51 paths. For a
BSP's export limit, an observed daily peak export exceeded the largest of 5 chosen paths on 12% of
days (1.6% for all 51 paths) and the largest of 10 chosen paths on 6.3% of days. A label such as
"the low-wind scenario" does not carry from day to day or lead to lead: under the Schaake shuffle a
path's wind rank between days 1 to 3 and days 4 to 10 has a rank correlation of 0.19.

**The stored forecasts hold 11 quantile levels, but the production docs' delivery tables list 13.**
The 13 delivery levels (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 0.98,
0.99) share 7 levels with the 11 stored levels. Delivering all 13 means re-predicting the stored
forecasts at the extra levels. Experiment 628 did not measure the effect of that re-prediction.

## Five questions for NGED

The five items below are questions, not commitments. The answers decide which parts of the design
are worth building.

1. **How many scenarios does NGED run by hand?** In the advisor's illustration (an assumption, not
   NGED's statement), a hand-run study in a graphical tool allows about 3 to 10 runs per element per
   forecast, which sets how few paths the delivery must offer.
2. **How many power-flow runs can NGED afford, and are they scripted?** A scripted study could allow
   thousands of runs, in which case NGED can draw its own scenarios from the stored quantiles.
   Related questions: which decision the study feeds (outage planning, flexibility procurement, or a
   curtailment notice), at what lead time, and what scalar each node takes (MW or MVA, which power
   factor, and power available or power after an active-network-management cap).
3. **Which BSPs, grid supply points (GSPs), or circuits matter?** Can NGED give the weight each node
   carries in the flow through each constrained element, or link its electricity-network model to
   our series identifiers? Which nodes in the study do we not forecast, and what does NGED assume
   for them?
4. **Does a direct forecast of the flow through a BSP, GSP, or circuit already exist on NGED's
   side?** We scored direct forecasts of the meters of 2 BSPs. The 2 GSPs have direct forecasts too,
   which we did not use in a sum.
5. **Should the delivery tables hold 11 or 13 quantile levels?** The stored forecasts hold 11, and a
   move to 13 needs the forecasts re-predicted.

## What we tried

- **[Experiment
  434](../../00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts/result.md)
  (`not supported`).** Re-ordering the stored demand forecast's 51 values by member temperature or
  by past observed paths did not beat independent draws across the 16 Primaries. Temperature ranks
  scored 6.7% worse than independent draws on the variogram score across the Primaries. The Schaake
  shuffle scored 3.5% lower (better) than independent draws on the variogram score across leads at
  days 1 to 10.
- **[Experiment 436](../../00436-joint-scenarios-across-demand-wind-solar-and-leads/result.md)
  (`partly supported`; joint block void, see the erratum on its page).** Across the farms, a Schaake
  shuffle scored 28% (wind) and 40% (solar) below independent draws, and the copula-corrected
  ensemble ranks 29% and 43% below. Across the Primaries nothing beat independent draws by more than
  0.7%. Tying scenario k to ensemble member k scored the same as a Schaake shuffle across Primaries
  and wind farms and no better across solar farms.
- **[Experiment 625](../../00625-coherent-members-from-quantiles-k-sweep-and-path-checks/result.md)
  (`partly supported`).** Experiment 625 ran the K sweep and the anomaly-path shuffle described
  above. It also re-scored experiment 436's joint block and found no difference between couplings
  (within +0.8%).
- **[Experiment 626](../../00626-widen-the-tails-of-the-51-scenario-member-values/result.md)
  (`partly supported`).** A separate upper and lower factor brought demand's q05 and q95 exceedance
  to 4% to 6% in every lead band, and did not fix wind or solar. Demand's q90 was still exceeded for
  11% to 16% of actuals (nominal 10%), which points to a median that sits too low rather than tails
  that are too narrow.
- **[Experiment 627](../../00627-scenario-subset-selection-from-51-coherent-paths/result.md)
  (`partly supported`).** Experiment 627 gave the subset results described above.
- **[Experiment 628](../../00628-quantile-form-client-side-scenario-rehydration/result.md) (`partly
  supported`).** Experiment 628 tested the client-side rebuild described above. The tail rule
  (clamp, linear, or Gaussian) changed CRPS by under 0.02%, and the stored upper tails were too
  narrow for any of the three rules to repair.
- **[Experiment 629](../../00629-bsp-group-sum-of-nodes-against-direct-bsp-forecast/result.md) (`not
  supported`).** Sums of the Primaries missed the BSP export tail. The threshold-weighted CRPS test
  could not rank the coupling methods, because neither BSP's meter reached the scaled export limit
  in the development months.
- **[Experiment
  630](../../00630-roster-upper-demand-tail-calibration-event-days-and-bias-correction/result.md)
  (`partly supported`).** The stored neural-network forecasts exceeded their q90 on 13% to 17% of
  event-free readings (5 development months). A correction fitted on the first three months removed
  about a quarter to a half of the excess on the last two, because the bias changes with the season.
- **[Experiment
  631](../../00631-bsp-group-sum-with-metered-generators-against-direct-bsp-forecast/result.md)
  (`not supported`).** Experiment 631 scored the farm-inclusive sums described above. The cause of
  the import-side miss (the Primary forecasts' bias, the farms, or the remainder) was not separated.

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
- **Do not write that demand has no mixture.** The bake-off's demand neural networks are
  mixture-density neural networks with 3 components, but the export stores only the mean of 5 seeds'
  quantiles, which is not any single mixture. The wind and solar mixture parameters were also never
  stored. Experiment 628 concluded that mixture parameters were not needed to rebuild marginal
  forecasts or a Schaake-shuffle coupling from the 11 quantiles. Recovering the parameters would
  need a rebuild of the inference of experiments 612 and 613, which has not been done.
- **Do not describe BSP sums as validated.** Experiments 629 and 631 each rest on 2 BSPs and on a
  scaled export limit that was never reached, so the threshold-weighted CRPS could not rank the
  sums.

## Advisor review of the delivery design (2026-11-07)

This section summarises an advisor review that read the background docs and experiments 430, 434,
and 436 and proposed the delivery design above. The review is the only place the full reasoning for
the design is written down. Statements about NGED's workflow that no source states are marked
(guess) and need NGED's confirmation.

**The binding quantity is the flow through one constrained element, and that flow is a weighted sum
over many nodes.** For about half of NGED's generators the limit sits at a BSP or GSP. For most of
the rest it sits on a circuit and binds only after the next fault, found by power flow plus N-1
contingency analysis (one outage simulated at a time). In an almost radial part of the electricity
network every weight is 1 for the nodes below the element. In a meshed 33 kV part the weights are
power transfer distribution factors (the share of an injection at a node that flows through the
element). The flow is then close to linear in the node injections: F is about the sum of w_i times
P_i.

**The review corrected four parts of the working understanding.**

- Reverse flow is not itself a breach. It is a breach only where it exceeds the reverse-direction
  rating of a transformer or circuit, or pushes voltage or protection out of limits (guess about
  which binds most often in the trial area).
- Real-time curtailment is done by active network management (ANM), not by a study days ahead. A
  study 1 to 10 days ahead is more plausibly for outage planning, flexibility procurement, or
  warning generators of likely curtailment (guess).
- "A power-flow model takes only scalars" is true of one run and is not the constraint. A
  time-series power flow takes one scalar per node per half-hour. The real limit is how many runs
  the analyst can afford, which depends on whether the runs are scripted.
- The power flow needs nodes we do not forecast: unmetered generators, other Primaries, and the 10
  sites metered in MVA (megavolt-amperes, which carry no flow direction). Our scenarios cover only
  part of the sum.

**For curtailment, the tail of the sum decides the outcome, so dependence between nodes matters more
than any single site's tail.** The review's worked example is two neighbouring solar farms under one
circuit with a 15 MW export limit, each forecast roughly Normal(6 MW, sd 2 MW) at noon (a
simplification, because solar output is bounded by clear sky and capacity):

| Assumed correlation | sd of the sum | P(sum > 15 MW) | q95 of the sum |
|---|---|---|---|
| 0 (independent draws) | 2.83 MW | 14.5% | 16.7 MW |
| 0.62 (realised in experiment 436) | 3.60 MW | 20.3% | 17.9 MW |
| 0.95 (ensemble copula coupling) | 3.95 MW | 22.4% | 18.5 MW |

- Independent draws understate the chance of a breach by about a quarter and put the q95 1.2 MW too
  low. An overstated correlation puts the q95 0.6 MW too high, so generators are curtailed more than
  needed. The curtailment metric prices a breach at 20 times the curtailed energy, so overstating
  dependence is the safer error, but not a free one.
- Experiment 436's finding that nothing beat independent draws across the Primaries does not mean
  the Primaries are independent. Their realised correlation was 0.29, so the sd of a sum of 16 equal
  Primaries is 2.3 times the independent value. Experiment 436's tail table showed 13.9% of actual
  group net flows above the q95 of independent-draw sums, against 6.6% for the Schaake shuffle and
  2.6% for ensemble copula coupling, while the variogram score barely moved. The group-sum
  exceedance rate is therefore the right instrument for this decision, and the variogram score is
  the wrong one.
- Demand and solar errors are negatively correlated (-0.36 realised), so in the export direction a
  sunnier-than-forecast day also lowers demand through embedded solar and the two errors add. No
  experiment has tested cross-family dependence.
- A Gaussian copula has no tail dependence. A Schaake shuffle copies real past days and keeps tail
  dependence empirically, limited by the size of its date pool, which favours the shuffle for the
  tail.

**No single delivery option wins.** The review's comparison:

| Option | Analyst's work | Gets the tail right? | Can we validate it? |
|---|---|---|---|
| Many draws (marginals plus copula), analyst picks | Moderate: sampling code, then choosing what
"extreme" means | Only as far as the marginal tails and the copula's tail dependence allow; 1,000
draws carry about +/-0.3 points of Monte Carlo error at q99 | Indirectly, through group sums |
| (a) Pre-chosen stress scenarios per element | Least: feed K vectors straight into the power flow |
Yes at the chosen level, if the element flow's quantile is right | The element quantile, yes; the
split across nodes, no |
| (b) Distribution of the element flow itself | Low, but the result cannot be fed into a power flow
| Best, where the element is metered | Yes, against the element's own meter |
| (c) Importance sampling or conditional simulation | High: weighted samples confuse
non-statisticians | Yes, efficiently | As for (a) |
| (d) Quantiles of group sums as extra series | Low | Yes for the named groups | Yes |

- Option (b) has an advantage: the Boston and Skegness BSPs are metered and already among the
  series, so a direct forecast is trained and scored on the true flow with no copula. A sum of node
  scenarios cannot match it, because unmetered load, losses, and the direction-blind MVA Primaries
  sit inside the BSP meter but outside any sum of our nodes. Experiments 629 and 631 confirmed this.
- Options (a) and (c) reduce to one recipe. For element E, day d, and level tau (for example 0.5,
  0.9, 0.95, and 0.99 of that day's peak export), deliver the coherent vector of node values most
  likely given that the flow equals its tau quantile. Under a Gaussian copula and a linear flow this
  vector has a closed form, mu + Sigma w (q - mu_F) / sigma_F squared. With Schaake paths it is the
  drawn path whose element flow lies nearest the quantile. The label "Skegness export, day 3, 95th
  percentile" names the target quantity and level, is recomputed in every run, and never refers to a
  member's identity, so nothing carries over from day to day.
- A deterministic bounding case is worth adding beside the statistics: maximum generation with
  minimum demand. For solar, maximum generation is clear-sky output capped at capacity and needs no
  tail statistics. The added value is the probability that a day approaches that ceiling (guess that
  NGED's analyst already runs this case).

**The review recommended validating the element flow against its own meter on four quantities.**

- The exceedance rate at q90, q95, and q99 (and q01 to q10 on the export side), by lead band, with
  summer middays separately.
- Threshold-weighted CRPS at the scaled limit (0.9 times the maximum export observed before the
  scored period). The score is proper and fits curtailment, because it scores every half-hour and
  weights only the region above the line.
- The Brier score of the flow exceeding the scaled limit, with a reliability diagram.
- The tier 1 curtailment energy proxy of `studies.node_curtailment`, with the realised unmet
  fraction.

**The independent sample is the weather day, not the half-hour, so q99 cannot be validated.**
Export-side tails live on roughly 150 to 180 summer half days a year per BSP (estimate). A q95 test
then rests on about 8 to 9 exceedance days per BSP per year (binomial error about +/-1.6 points on a
5% rate), and a q99 test on about 2. The firm limit had been reached once in the pilot. What cannot
be validated with our data: q99 and beyond; tail dependence in the far tail; any element that is not
metered, including every post-fault circuit; combined rare events such as an outage on a clear-sky,
low-demand day; and the split of a stress scenario across nodes, as distinct from the sum. The
review advised stating these limits to NGED's analyst rather than covering them with a score.

**The review's recommended design.** For each element NGED names, starting with the export side of
the Boston and Skegness BSPs:

- Deliver the element flow's quantiles from a direct forecast of the metered flow.
- Deliver 3 to 5 conditional stress vectors per element per day, at tau in {0.5, 0.9, 0.95, 0.99} of
  the daily peak export, generated from Schaake paths and rescaled to hit the element quantile.
- Keep per-node marginals (13 quantiles, with widened tails) and the Schaake template dates on S3,
  so a scripted analyst can draw more scenarios.

The fallback is marginals plus a Schaake or copula table with client-side sampling, a 20-line client
recipe, and published group-sum quantiles for the two BSPs so the analyst can check their own sums.
The review's cheapest next experiment was a CPU-only re-score of experiment 436's stored scenario
tables as BSP-group sums against the BSP meters, which became experiments 629 and 631. The design
with conditional stress vectors (the second bullet) has not been built or tested, and is the open
question below.

## Open question

**No experiment yet tests how to make coherent node scenarios agree with the direct forecast of the
flow through a BSP or GSP.** NGED's power-flow study needs node injections, and the direct forecast
supplies the flow through the element. Experiment 631 shows that the two disagree in the tails when
the node scenarios are summed. One candidate design conditions the node scenarios on a quantile of
the direct forecast. No experiment tests it yet, and we do not know whether NGED wants it. Questions
3 and 4 above bear on it.

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
  together as the nodes do in reality. We propose, pending NGED's answers: (1) the 11 stored
  quantiles per series and half-hour together with a dependence table: for the wind and solar farms,
  51 past observed paths to drive a Schaake shuffle; for the 16 Primaries, none; (2) a direct
  forecast of the meter of each bulk supply point, because summing node scenarios did not reproduce
  the tail of the flow through a bulk supply point in any experiment we ran; (3) on request, 10 or
  more scenarios chosen without labels by stratifying the 51 paths on one summary." Link claim (1)
  to experiment 625, claim (2) to experiments 629 and 631, and claim (3) to experiment 627.
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
  more interested in the **tails** than the shoulders." Add: "On 44 Tuesday runs of 2025-09 to
  2026-06, the stored q99 of the Primaries was exceeded by 1.0% to 1.6% of event-free actuals for
  the XGBoost forecasts and 1.9% to 3.1% for the neural networks (experiment 628). On 5 development
  months, the neural networks' stored q90 was exceeded by 13% to 17% of event-free actuals
  (experiment 630). The upper tail of demand is therefore too low. Extending the quantile function
  past q01 and q99 with a tail rule does not repair this (experiments 628 and 630)."

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
  direct forecast of the meter came closer to the nominal rate (4.9% to 9.9% of actuals beyond each
  tail, against 5%)." Link experiments 436, 625, 629, and 631.
- **The tail caveat quotes a literature finding but not our own.** The page says: "The mixture's own
  p1 and p99 may therefore need the same parametric-tail treatment before they can be trusted." Add
  after it: "In our own data, a Gaussian tail beyond q01 and q99 gave the best lower-tail coverage
  of the three tail rules we tested (0.20% to 0.45% of actuals below the 0.2% quantile, against
  0.53% to 1.05% for clamping), but on the upper side 0.66% to 1.19% of actuals lay above the 99.8%
  quantile, against 0.2% nominal, and the tail rule changed CRPS by under 0.02% (experiment 628).
  None of the three tail rules repaired stored quantiles whose upper tail was already too low."
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
