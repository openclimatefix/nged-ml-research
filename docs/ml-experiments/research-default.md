# The research default and the rule for adopting a change

**Each target family has one fixed research default forecast, one fixed rule for adding a change to
it, and a [log of versions](research-default-log.md).** An experiment of kind `change` asks whether
the research default plus change x beats the research default. A win produces the next version of
the research default. The rule exists so that the experiments build one forecast in order, and so
that a gain is always measured against the current forecast and never against a stale or partial
one. How a change is scored, and when it is adopted, is fixed by the family's evaluation protocol:
[Primary demand](evaluation-protocol-demand.md), [metered solar
farms](evaluation-protocol-solar.md), or [wind farms](evaluation-protocol-wind.md).

**Adopting a change into the research default does not put the change into production.** The
`release-model` skill governs production changes. This page says "research default" and "adopt"
because [production
deployment](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/)
already uses "champion" and "promote" for the deployed XGBoost model. Every version has a name made
of the family, the words
"research default", and a number N that counts up within the family from 1, for example "Primary
demand research default 1" for the GAM stack A+H+S. "C1" and "C2" already name the planned contrasts
on every result page, so a bare "default 3" or "v3" is never used. Every `result.md` and
pull-request body cites the version by name, for example "default plus x versus Primary demand
research default 1". The two arms of a `change` experiment are called "default" and "default plus
x".

**The research default is not what runs in production.** Production serves a much weaker XGBoost
model, stored as `xgboost_baseline`. The [demand protocol](evaluation-protocol-demand.md#the-three-fixed-benchmarks)
gives its score beside the score of the frozen reference BG2.

## Families

**Each family has its own research default, series, retraining cadence, and version line.** The
family definition below fixes these values once, and an experiment's plan never sets or changes
them. The row set, lead range, headline metric, protected groups, and benchmarks of each family are
in its evaluation protocol.

| Family | Series | First version of the research default | Retraining cadence |
|---|---|---|---|
| Primary demand | 16 Primaries | Primary demand research default 1: the GAM stack A+H+S of experiments [220](00220-dp-bg-stage4-assembly-with-drop-one/result.md) and [251](00251-dp-final-best-guess-assembly-clip-and-coast-against-bg2/result.md) | Monthly for the GAM stack; weekly for its XGBoost benchmark BG2 |
| Metered solar farms | Farms A to F | Metered solar farms research default 1: the blend assembled by experiment 300 | A single fit or monthly, as the family's first version states |
| Wind farms | W1 to W3 and the public wind farms of experiment 288 | Wind farms research default 1: the ERA5-trained XGBoost with the offset table of experiment [303](00303-wind-adapter-in-production-setting/result.md); the current version is Wind farms research default 2, the frozen physics plant plus a mixture-density head of experiment [347](00347-wind-physics-hybrid-plant-then-mdn/result.md) | A single fit of the physics plant, with the head trained once on the hindcast (experiment 347); experiment 346 tested refit cadence for the offset table of Wind farms research default 1 |
| Bulk and grid supply points | To be named with the first default | Bulk and grid supply points research default 1: to be named when the first default is chosen. No adoption until then | Weekly |

## Reusing the research default's code and forecasts

**A `change` experiment starts from the stored forecasts of the research default and, when the
change acts inside a fitted stage, from the default's code.** The change is one of two kinds, and
the experiment's page says which.

- **Tier 1 changes act on stored forecasts.** A clip, a band-widening rule, a blend, a
  recalibration, or a quantile interpolation needs only the research default's stored forecasts. The
  D8 as-of widening, the lead-band interpolation, the D5 clip, and the blend of experiment 287 are
  all of this kind.
- **Tier 2 changes act inside a fitted stage.** A change to the convex model, the head, the solar
  summary, or the features needs the research default's code. The experiment imports that code
  directly from the folder of the experiment that holds it, with that folder on `PYTHONPATH`, as the
  `ml-experiment` skill describes. The code moves into `src/studies` in a reviewed pull request
  only when several experiment folders need it or when production might one day need it.

**The research default's definition holds four items.** These are the code (a script in an
experiment folder and its commit, a module and commit in `src/studies`, or "stored forecasts
only"), a configuration file naming every setting and the seed policy, the stored forecasts for
every scored, replicate, and holdout period, and the scores. The seed policy matters: averaging
three seeds of the correction network scored 0.123 points below arm D, against 0.104 for the mean
single seed ([experiments 201 and 220](00220-dp-bg-stage4-assembly-with-drop-one/result.md)), so
"more seeds" would otherwise adopt as a change.

**Each version's stored forecasts are written once and never overwritten.** Several worktrees share
one `data/` folder, and a re-run by one session would replace the evidence of another.
Experiments in flight read Primary demand research default N while Primary demand research
default N+1 is being built. Each version is written once with `experiment_store.write_predictions`
under the adopting
experiment's id and a fixed name. Where a refit is needed to reproduce the default, the reproduction
uses the default's cadence exactly, because refit cadence is itself a contrast (experiment 278).

## What an experiment must contain to be eligible

**An experiment is eligible for adoption only when its front matter says `kind: change` and its
plan, committed before any score is read, states five items.** Diagnostics, audits, data checks, and
detection or decision studies are `kind: other` and never adopt. The five items are the following.

- The family and the research default version the experiment starts from.
- The arms: the default, and the default plus x. The default's arm reads the stored forecasts of
  that version, with a reproduction gate when the experiment refits the default, as experiment 278
  did.
- The tier of the change (tier 1 or tier 2) and its complexity tier, as the evaluation protocol of
  the family defines them.
- The pre-registered default-plus-x arm or arms.
- The controls that apply: a known-answer test, a placebo, a leakage check.

## When a change is adopted

**A change is adopted only when the conditions of the family's evaluation protocol hold.** The
conditions, the candidate status for a change that fails one of them, the rules for several arms and
for blends, the holdout, and the interval conventions are all in the protocol. The reference text is
the section ["When a change is adopted"](evaluation-protocol-demand.md#when-a-change-is-adopted) of
the demand protocol.

## Experiments planned before the rule

**Experiments planned before the rule are grandfathered, and their winners reach the research
default only through a rebase run.** Their front matter has no `kind`, and `experiment_tables.py
--check` accepts its absence for an experiment whose id is below the cut-off id of 320. A proposal
with no
plan commit yet is re-planned as `kind: change` where it tests a change.

## The rebase rule for stale results

**A result measured against an older version stays valid as an estimate of x on that version, but it
answers a stale question.** The estimate (Primary demand research default 3 plus x, minus Primary
demand research default 3)
is still correct for Primary demand research default 3. The estimate does not apply to x on Primary
demand research default 4. Adoption one
change at a time is path-dependent: a change that failed against one version may pass against a
later one, and two changes that act on the same error can both win alone while only the first to
merge adopts. The two gains of experiment 220 added to a stacking ratio of 0.99, while the drift
term improved the bare GAM-K by 0.40 points and worsened arm D by 0.03.

1. A result that passes against a version that is no longer current becomes a candidate with the
   status "candidate (stale)".
2. A short rebase run scores (current default plus x, minus current default) against the stored
   forecasts of the current version, under the same adoption conditions and with no new plan.
3. A rebase is needed unless every adoption since the experiment's version belongs to a different
   family.
4. Where two candidates change the same component (two widening rules, for example), the rebase
   scores them head to head.
5. Adoptions go in merge order. The second of two overlapping changes gets a rebase run against the
   new version.
6. Changes rejected against the last few versions stay on a retest list that the periodic ablation
   revisits.

**After every fifth adoption in a family, an ablation experiment removes each adopted component in
turn.** The ablation is an experiment with its own id and a planned `result.md`. The
[demand protocol](evaluation-protocol-demand.md#when-a-change-is-adopted) states when a component is
dropped.

## Who adopts, and what the log holds

**The experiment's own pull request carries the adoption.** Its front matter holds `promoted_to`
(the name of the new version, for example "Primary demand research default 2") and the scores
against the three benchmarks, and the same pull request adds a row to the
[log](research-default-log.md). The reviewing session checks the adoption conditions before merging and
assigns the version number. Versions are serial within a family.

**Each log row holds eight items.** These are the change, the adopting experiment, the scored-year
and replicate gains with intervals, the absolute headline score, the three benchmarks' scores on the
family's rows, the stored-forecast name, the commit of the code, and the status "scored-year gain is
an upper bound". A result against an old version keeps its verdict with the status "candidate
(stale)" until a rebase runs. A component that a later ablation removes sets `production_status:
superseded` on its experiment.

**Adoption needs the maintainer's approval in four cases.** These are a new data source, a cost
above £30 or a recurring cost in production, a change that would need a serving-path
change to port, and a change whose code must enter `src/studies`. Other adoptions run
autonomously under the merge rules in the `ml-experiment` skill.

**A model release is a separate step.** The Primary-demand research default is a GAM stack, and a
model release is a numbered, trained XGBoost model, so most demand adoptions are not release
candidates until the GAM stack has a production design. The `release-model` skill starts from the
log of versions since the last release.

## Retraining cadence

**The cadence is part of the research default, so each family's definition states one, and a change
of cadence is a `change` experiment.** The evidence differs by family.

| Family | Evidence on cadence | Cadence |
|---|---|---|
| Primary demand, XGBoost benchmark BG2 | Weekly beats monthly by 0.123 (experiment 87) to 0.13 (experiment 48) points | Weekly, with a 12-month-window monthly refit as a research proxy ([experiment 85](00085-cheaper-retraining-schedules-rank-arms-like-weekly-demand/result.md)) |
| Primary demand, GAM stack | Weekly minus monthly +0.001, quarterly minus monthly +0.010 (experiment 278) | Monthly |
| Bulk and grid supply points | Weekly beats monthly by 0.160 points (0.066 to 0.247, experiment 87) | Weekly |
| Metered solar and wind farms | Monthly minus frozen -0.044 for solar and +0.007 for wind (experiment 48), inside a ±0.05 band | A single fit, or monthly, as the family's first version states |

**Quarterly refits are not used for the XGBoost-based families.** Experiment 85 found that quarterly
refits scored 0.38 points above weekly refits on the same 12-month window, and advised against using
them to compare arms that differ in cadence or in how fast they adapt. For the GAM stack, quarterly
refits save about 19 seconds per arm-year and add a loss of 0.010 points (99% interval -0.014 to
+0.032), concentrated in winter, so they are not shown adequate. Their effect on the power of a
default-plus-x contrast has not been measured.
