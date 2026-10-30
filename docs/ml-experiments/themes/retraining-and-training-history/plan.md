# Retraining and training history: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [P16](#p16-a-longer-ens-history) | Does the ENS backfill (2021-03 to 2024-03) give an extra validation fold, or improve the single-stage or stage 2 models? | 5 | The maintainer, who runs the fetch on a cloud machine | Deferred; tracked in [issue #959 (Extend ECMWF ENS training history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959), and the 51-member fetch cannot run autonomously because it needs a cloud machine in `us-west-2` |
| [E418](#e418-pretrain-on-era5-and-cams) | Does pretraining the pooled mixture density network on ERA5 and CAMS before fine-tuning on ENS lower the Primary demand score? | 1 | Nothing | Planned, [experiment 418](../../00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/result.md) |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### E418: pretrain on ERA5 and CAMS

**Pretraining the pooled neural network on the longer ERA5 and CAMS history could lift the whole
network, because the ENS archive covers only 2024-04 onwards.** The plan, the arms, and the decision
rule are in [experiment 418](../../00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/result.md).

### P16: a longer ENS history

**The ENS backfill from Source Cooperative (2021-03 to 2024-03) could give an earlier validation
fold, and the full fetch cannot run autonomously.** [Issue #959 (Extend ECMWF ENS training
history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959) describes the
fetch. The archive is in `us-west-2`. Fetching the control member alone is about 530 GB and 4 to 5
hours, and fetching all 51 members is about 30 TB, which needs a cloud machine in that region and
the maintainer's approval of the cost. The issue also ties autonomous use of the fetch to the
scorer-protection work in issue 958. Every experiment since 00003 scores the same year on the same
16 series, so choices tuned there overfit it, and an earlier fold is the remedy. For demand, a
longer training history has hurt so far, so the backfill is valuable first as validation data.
