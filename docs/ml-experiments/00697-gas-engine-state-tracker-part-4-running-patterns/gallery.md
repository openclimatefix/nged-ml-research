# Experiment 697: exploratory figures

**These figures are exploratory, and none of them changes the verdict of [experiment
697](result.md).** Each shows the running share of D8's gas engine as the tracker labels it, in the
net era (2023-09-01 to 2025-02-28) and the solar era (2025-03-24 to 2026-02-28), and the null control
where a caption names one. The terms (tracker, running share, null control, cell, spell) are defined
on [the result page](result.md). The figures were drawn after the tables were seen, so any reading of
them is post hoc. The night share by month (G7), which is the share of half-hours from 00:00 to 05:00
that the tracker flags, and D8's dated rasters stay in the private data folder. Bins that hold fewer
than 20 days are not drawn as shares, and G6 shows their counts.

**G1: in the net era D8's labels show 4 or more running half-hours on almost every weekday, and on
fewer weekend days.** The left panel is the share of days with 4 or more running half-hours (a
running day), by weekday. The right panel is the share of running days after a running day and after
an idle day, for Tuesday to Friday. The null control's truth has no day-to-day link, so its flat line
is what a patternless engine gives. The null control is the one built on D8's reading before its
engine.

![Running days by weekday](g1_running_days.svg)

**G2: starts cluster around 06:00 to 08:00 and 15:00 to 17:00 local time, and stops around 19:00 to
22:00.** Each point is the number of on-spells that start, or end, at one local half-hour, for the
summer half and the winter half of each era.

![Start and stop clock times](g2_start_stop_times.svg)

**G3: the running share rises with the half-hour's price rank within its day, and is highest in the
red band.** The left panel ranks the half-hours of each local day by system price. The right panel
compares the red band (Monday to Friday, 16:00 to 19:00 local) with the 6 dearest half-hours of the
day that fall outside the band, and with other weekday half-hours. Bank-holiday evenings are not
drawn, because they hold fewer than 20 days.

![Price rank and the red band](g3_price_rank_and_red_band.svg)

**G4: the running share rises with the spark spread and not with the gas price.** The spark spread is
the system price minus the gas price divided by an assumed efficiency of 0.40, in pounds per MWh. The
left panel uses the National Energy System Operator's (NESO) held N2EX hourly day-ahead price, which
starts in 2025-09, so it holds only the part of the solar era from 2025-09. The Elexon APX within-day
price index, which starts in 2023-06, is in G9.

![Spark spread and gas price](g4_spark_spread_and_gas.svg)

**G5: D8's share minus the null control's share falls as the wind speed and the temperature
rise.** The bars are thirds of each driver within a cell, for the net era only, and the control is the
null control on the host series. D8's heating load changes its reading, so D8's raw weather link is
not read.

![Weather](g5_weather.svg)

**G6: bank-holiday weekdays, Christmas week, and clock-change days each hold fewer than 20 days, so
no running share is drawn for them.** The bars are the numbers of weekday evenings in each group and
era, and the dashed line is the floor of 20 days. This departs from the earlier version of the
figure, which drew shares from groups of 2 to 18 days.

![Days in each holiday and clock-change group](g6_holidays_and_clock_changes.svg)

**G8: the running share and the tracker's fitted block size by month and local half-hour.** Left
panels: the running share on weekdays. Right panels: the mean fitted block size in MW when flagged,
where a block is the tracker's fitted size of one unit of output. White cells hold fewer than 15
days.

![Season](g8_season.svg)

**G9: the running share is higher in the top third of the within-day price and of national demand,
and lower in the top third of embedded wind and of the forecast imbalance.** Each row is the running
share in the top third of a signal minus the bottom third, within a cell, with a 95% interval, for
D8's net and solar eras and for the null control on the host series. The Elexon within-day price
index is not the day-ahead auction.

![Other signals by third](g9_other_signals_by_third.svg)

**G10: the running share inside and outside a Short Term Operating Reserve (STOR) window.** The
share is on weekday evenings. Whether the engine holds a reserve agreement is unknown, so we read a
match with a window as a coincidence of timing and not as a cause. System warning days (5 in the net
era and 0 in the solar era) and gov.uk bank holidays (11 and 8) each hold fewer than 20 days and are
not drawn.

![Reserve windows](g10_stor_windows_warnings_and_holidays.svg)
