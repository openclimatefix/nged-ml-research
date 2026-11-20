# Calendar keys left in the labelling prompt

This file records which calendar keys the pinned labelling prompt still carries and why the plan
accepts them. Every number is interim.

## Fixed before the year probe

- **Weekday columns removed.** The prompt shows no weekday or weekend split and no per-day flags.
- **Weekly means only.** Table 1 holds one mean per 7-day block, and the daily rows are 7-day
  trailing means.
- **One climatology per prompt.** The climatology comes from one month, so the prompt shows no step
  on the first day of a month.
- **Local clock hours.** The hourly profile uses Europe/London clock hours.

## Accepted, and tested by the year probe

- **Weather values remain.** The daily weather values can still hint at the season.
- **Easter and Christmas dips remain.** A demand dip in the series can still hint at the year.
- **Part-week blocks remain.** A part-week block has an uneven mix of weekdays.
- **Seasonal drift remains.** The series drifts against one climatology.

## Year probe result

- **The probe passed.** Sonnet named the exact year for 2 of 10 prompts (the rule passes at 3 or
  fewer) and a year within one of the true year for 3 of 10.
- **The 10 answers are not independent.** The 10 prompts come from 5 candidates in 4 distinct
  years, so the probe has less power than 10 independent draws.
