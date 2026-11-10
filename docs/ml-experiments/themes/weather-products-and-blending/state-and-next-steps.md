# Weather products and blending: state and next steps

**Training on ERA5 weather lowered the error for solar and wind farms without power lags, and a few
other weather changes helped too, but no second weather product has been shown to help in a way a
live service could use.** ERA5 is the European Centre for Medium-Range Weather Forecasts (ECMWF)
reanalysis of past weather, ENS is the ECMWF ensemble forecast, and CAMS is the Copernicus
Atmosphere Monitoring Service. The final models in [Final models](../../final-models.md) differ by
forecast type in how they use these products. The Primary demand model pre-trains on ERA5 and CAMS,
fine-tunes on ENS, and takes two CAMS irradiance columns as inputs. The wind model takes ENS weather
with no ERA5 or CAMS pre-training. The solar model takes ENS weather with no pre-training.

Every figure comes from a merged result page of the theme, linked at its first use. The [theme
summary](index.md) holds the full evidence, and the [research plan](plan.md) holds the open
proposals. Terms are defined in the [glossary](../../glossary.md).

## Take-home message

**ERA5 training lowered the mean absolute error for wind and solar farms.** Training the
weather-to-power XGBoost model on ERA5 lowered the mean absolute error at 3 wind farms by 1.70 points
of capacity (95% interval -2.33 to -1.08) and at 6 solar farms by 0.57 (-0.83 to -0.32), up to day 6
and with no power lags ([Experiment 1](../../00001-era5-trained-stage1a-vs-ens-trained/result.md)).
The contrasts were revised after two reviews, so they are not pre-registered.

**Other weather changes gave clear gains in single experiments, and none has been confirmed for
production.** UKV wind (the Met Office's UK model) lowered the wind-farm quantile score by 0.198
(-0.316 to -0.069), an upper limit. Mapping the ENS shortwave onto CAMS lowered the Primary demand
score by 0.067 (-0.094 to -0.039). Weather as accurate as ERA5 lowered it by 0.10 (-0.16 to -0.04).
ERA5 training is the only one of these gains that no later experiment downgraded.

**Several other changes failed or stayed unresolved.** For demand, ERA5 training scored 1.37 points
worse ([Experiment 3](../../00003-era5-trained-stage1a-vs-ens-trained-demand/result.md)). A second
global ensemble, NOAA's GEFS, did not improve the sunshine anomaly, which is the solar output of a
generic fleet minus its month-by-half-hour mean ([Experiment
50](../../00050-second-weather-forecast-for-solar-anomaly/result.md)).

**Every option below is optional.** The final models take one weather source per forecast type, and
the research plan defers all three weather-product proposals.

## What works

**ERA5 training lowered the solar and wind farm error up to day 6, and an ENS-trained model was level
with it once power lags were added.** The ERA5 result was measured without power lags. The
comparison with power lags uses the ERA5-trained best guess of Experiment 33 as the comparator, at 24
to 54 hours only, and the quantile score. The ENS-trained model scored 0.218 points lower for wind
(-0.472 to +0.069) and 0.107 lower for solar (-0.327 to +0.053). Both differences are unresolved
([Experiment 54](../../00054-product-trained-xgboost-wind-farms/result.md), [Experiment
57](../../00057-product-trained-xgboost-solar-farms/result.md)).

**UKV wind lowered the wind-farm quantile score by about 0.2 points at 24 to 54 hours, and that
number is an upper limit.** An ENS-trained XGBoost model given UKV wind columns scored -0.198 (-0.316
to -0.069) at 3 wind farms. A shuffled-UKV placebo scored +0.013 (-0.012 to +0.040). One farm gained
clearly (-0.413) and one did not (+0.018), a post hoc split. The data come from the Centre for
Environmental Data Analysis (CEDA) archive, which differs statistically from the live feed
([Experiment 54](../../00054-product-trained-xgboost-wind-farms/result.md)).

**A UKV and ENS blend lowered the squared error of the sunshine anomaly at 5 Primaries by 7.6% over
0 to 54 hours in cross-validation, and the validation year cannot confirm it.** The validation
correlation gain was +0.011 (-0.006 to +0.027). The test was post hoc, its intervals resample only 6
or 7 months of forecast origin, and the experiment's verdict is a small gain. Replacing the 06 UTC
UKV run with the 00 UTC run still leaves the error 3.6% to 7.7% lower in every lead band
([Experiment 53](../../00053-ukv-vs-ens-solar-anomaly/result.md)).

**Mapping the ENS shortwave onto CAMS lowered the Primary demand score by a further 0.067.** The
97.5% interval was -0.094 to -0.039. The verdict stays inconclusive because the monthly mismatch to
CAMS reached 18.6%, and the mapping widened the member spread by 27% ([Experiment
14](../../00014-cams-trained-stage1-with-stage0-ens-correction-demand/result.md)). Experiments 1
and 14 are the two marked `candidate` for production. Neither is ready to port as it stands.

## What does not work

**ERA5 training did not carry over to demand series.** On 22 series that are not wind or solar farms,
the ERA5-trained XGBoost model scored 1.37 points worse (+0.68 to +2.17). Starting the ERA5 history
in 2023-09 instead of 2019-09 lowered the error by 1.32, a post hoc read that cannot separate drift
in the series from bad early data ([Experiment
3](../../00003-era5-trained-stage1a-vs-ens-trained-demand/result.md)).

**Weather as accurate as ERA5 gave a small gain for Primary demand.** The quantile score fell by 0.10
(99.38% interval -0.16 to -0.04) at 16 Primaries, and inside 2 days there was no gain ([Experiment
22](../../00022-era5-weather-ceiling-primary-demand/result.md)). Training on CAMS sunshine gave -0.035
against a planned 0.2 ([Experiment
9](../../00009-cams-trained-stage1a-vs-era5-trained-demand/result.md)).

**GEFS did not help the solar term of Primary demand.** The cross-validated squared error of the
anomaly was 1.109 times ENS's for GEFS, worse in all 9 held-out months, and 1.002 times for a fitted
blend, against the 0.99 needed. Experiment 50 tested 5 Primaries, not solar farms ([Experiment
50](../../00050-second-weather-forecast-for-solar-anomaly/result.md)).

**UKV shortwave, cloud, and temperature at 6 solar farms, and ICON-EU and IFS winds, were
inconclusive.** ICON-EU is the German weather service's European model, and IFS is ECMWF's
deterministic forecast. The solar score changed by -0.035 (-0.119 to +0.037), and the wind blend by
-0.055 (-0.200 to +0.072) at days 1 and 2 ([Experiment
57](../../00057-product-trained-xgboost-solar-farms/result.md), [Experiment
52](../../00052-more-nwp-products-for-wind-farms/result.md)). The products' runs in Experiment 52
were half a day older than ENS.

## Open questions

**Whether a live UKV history gives the wind-farm gain.** The CEDA archive has no 100 m wind and lacks
four hourly steps, and UKV was upgraded on 2026-01-21 ([Experiment
54](../../00054-product-trained-xgboost-wind-farms/result.md)).

**Whether the gain of better weather holds for a Primary demand model trained on ERA5.** Experiment
22 trained on the ENS control member, and ERA5 differs from ENS in bias (mean shortwave 132.6
against 139.1 W/m2).

**Whether the products' delivery times change the gain of any blend.** Experiment 53 assumed a
3-hour UKV delivery time without measuring it, and Experiment 52 fed runs half a day older than ENS.

**Whether a source of a different kind helps solar farms.** Experiment 50 tested GEFS at 5 Primaries
only, so GEFS is untested for solar farms.

## Options for next experiments

**An option is to test a shared stage 2 calibrator with a learned blend for a second product already
on disk (P15).** The question is whether CEDA UKV or Open-Meteo data can enter without retraining,
for solar and wind only. The arms are ENS alone, ENS plus one product with the shared calibrator and
the learned blend, a per-product-columns control, a fixed equal-weight mixture, and a learning curve
at 1, 2, 3, 6, and 12 months of the second product's history. The dependency is a decision on the
shape of the production model. The plan page notes that 9 generator series cannot reach a safe-entry
interval of plus or minus 0.15 points at 1 or 2 months of history.

**An option is to train a stage 1 XGBoost model on ENS early leads, with ceiling inputs (P13).** The
question is how much skill remains with ERA5, CAMS, or short-lead fields as inputs. The plan page
defers it, because Experiment 1 found no gain beyond day 6 and 9 generator series cannot resolve
gains below about 0.3 points of capacity. It has no dependency.

**An option is to use ensemble statistics as inputs (P11).** The question is whether the member mean,
spread, and percentiles of temperature and irradiance beat control-member rows. The plan page defers
it, because an ERA5-trained stage 1 model cannot take spread as an input. It has no dependency.

## What is parked, and why

**All three proposals are deferred in the research plan, at priority 5.** The reasons are the
resolution limit of 9 generator series, the two-stage chain scoring level with the single-stage
XGBoost baseline N1 for demand in Experiment 18, and the priority of documenting the final models
and porting them. A second weather product stays unproven for production, and Experiments 50, 52, 53,
54, and 57 are marked `not-for-production`.
