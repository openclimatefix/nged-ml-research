# Hierarchy and pooling: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [E421](#e421-history-fingerprint) | Can the pooled mixture density network infer which Primary it forecasts, and forecast a Primary it never saw, from the last 7 days of power alone? | 1 | Nothing | Planned, [experiment 421](../../00421-mdn-history-fingerprint-for-pooled-primary-demand/result.md) |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### E421: history fingerprint

**The last 7 days of a Primary's power might tell a pooled neural network which Primary it is
forecasting, so that the Primary embedding becomes optional.** The plan, the arms, and the decision
rule are in [experiment 421](../../00421-mdn-history-fingerprint-for-pooled-primary-demand/result.md).
