# Metered solar farms: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [300](../../00300-metered-farm-best-guess-assembly/result.md) | How good is the assembled best-guess day-ahead forecast for the six metered farms, and what does a new farm with no history get? | 1 | Nothing | Finished (partly supported): The assembled solar-farm forecast (a physical power model, two correction stages, and a blend with XGBoost) scores 3.3% better than plant-fed XGBoost (XGBoost on the plant forecast, its ensemble spread, the unit mean, and the availability ratio, with no weather features of its own) on the quantile score, which averages the pinball loss over 11 quantile levels, when each farm trains on its own history. See [experiment 300](../../00300-metered-farm-best-guess-assembly/result.md). |
| [430](../../00430-unified-forecaster-solar-and-pretraining-ablations/result.md) | Which of ensemble members passed one by one, a PyTorch plant, ERA5 and CAMS pretraining, and point masses at zero and capacity does a pooled mixture-density network for the six metered solar farms need, and which routes give coherent ensemble member paths? | High (maintainer, 2026-10-31) | Nothing | Finished: none of the four beats the plain network; the A6 wind arms are deferred |
| [433](../../00433-solar-idle-noise-data-look/result.md) | Do the six metered solar farms show the wind farms' idle-noise texture when they are not generating? | 1 | Nothing | Finished (supported): The metered solar farms read exact zeros at night, unlike the wind farms' noisy floor, so solar forecasts need no idle-noise detector and no cleaned-zero scoring. See [experiment 433](../../00433-solar-idle-noise-data-look/result.md). |
| [618](../../00618-solar-effective-capacity-sunny-rows-and-known-answer-test/result.md) | Can a solar farm's capacity be tracked from its very sunny half-hours alone, and did Solar A, B, and D produce more in 2021 to 2023? | 1 | Nothing | Finished (partly supported): Very sunny half-hours are enough to find a sudden change of 8% or more in a solar farm's capacity at Solar A, B, C, and F, with at most 0.7 false changes a year; Solar D has too few of them and needs a change of 12%. See [experiment 618](../../00618-solar-effective-capacity-sunny-rows-and-known-answer-test/result.md). |
| [636](../../00636-metered-solar-monthly-slope-capacity-known-answer-test/result.md) | Can a monthly regression of power on irradiance find known capacity steps at the six metered solar farms, including a loss of direct-current capacity hidden under inverter clipping? | Done | Nothing | Finished (refuted): A monthly fit of a solar farm's output against irradiance can spot a 10% loss of capacity within 3 months at Solar C and Solar F, and a loss of 6.3% or more at Solar C, but not at Solar A, B, or D, whose own history is too noisy. |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### Better irradiance for the metered farms

**[Experiment 218](../../00218-metered-solar-farm-plant-fit/result.md) found that the plant's shape
is not what limits the fit: giving each farm its own plant parameters raised the fraction of output
explained by only 0.003 over one shape shared by all farms.** The next question is whether a
solar-irradiance product other than the Copernicus Atmosphere Monitoring Service (CAMS), or a blend
of products, raises the fraction of the six farms' output explained by the plant model. The
experiment fails if the best blend raises that fraction by less than 0.02 over CAMS alone.

### XGBoost with the plant's mean and the members' spread

**[Experiment 224](../../00224-metered-farm-dayahead-two-number/result.md) found that an XGBoost
model scores lower pinball loss at the 0.01 to 0.95 quantiles and the plant at the 0.99 quantile.**
The proposal is to give an XGBoost model two extra input columns, the plant's mean response and the
standard deviation of the 51 plant responses, and to refit it weekly. The experiment fails if the
quantile score improves by less than 2% over plain XGBoost, or if the 0.99 pinball loss stays above
the plant's.
