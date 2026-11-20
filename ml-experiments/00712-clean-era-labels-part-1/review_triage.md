# Review triage for the experiment 712 plan

This file records what each review of the plan found and what the plan did with each finding.
Every number is interim.

## Design review (Opus)

- **Accepted, all 19 findings.** The reviewer gave file and line evidence for each, and each was
  checked before the design was revised.
- **Two blockers.** The all-rows veto K4 could not be passed by an arm with no effect, and the
  prompt carried the calendar date. K4 now compares against the stored placebo's interval. The
  prompt now holds week offsets only, and a year-guessing probe gates the pinned prompt.
- **Cost and counts corrected.** The injected bank holds 182 flagged shifts, not up to 246. The
  real record holds about 92 flagged shifts, not 86.

## Code review of the trial script (Opus)

- **Accepted, both blockers.** The script now refuses a rerun without `--resume`, and a billing
  probe gates every spend.
- **Accepted, the should-fix findings.** The fixes cover the alarm-day leak in the "also alarmed"
  line, the effective-capacity scale, timeout logging, the shared stop flag, the retained usage
  fields, and the working-directory isolation check.
- **Deferred, the calendar keys.** The weekday in the features, the first-of-month climatology
  step, and the holiday dips stay open until the prompt is pinned. The module docstring lists them.

## Numbers review (Sonnet)

- **Accepted.** US$0.0353 is the mean per check call and US$0.0338 the mean over all 22 calls. The
  21 checks are 20 scheduled checks plus 1 retry. The stage 1 and stage 2 call counts now say which
  arms they cover. The plan-quota sentence now says that the account-wide seven-day utilisation
  stayed at 0.32 for all 22 calls.
- **Rejected, the scaling paragraph and the design's section 6.4 disagree.** The page matches the
  trial and the script, so the page was left as it is.
- **Rejected, two omissions.** The US-dollar scaling cost and the K5 and K6b windows are omissions
  and not errors.

## Prose review (Sonnet)

- **Accepted.** The opening paragraph now defines the terms it uses, the bolded leads state
  conclusions, and the front matter wording was tightened.
- **Rejected, a four-row gate table.** The restructuring is optional and the gate content did not
  change.
- **Rejected, deleting the threshold paragraph from the body.** The front matter and the body keep
  their separate roles.
- **Rejected, a table inside the front matter.** The front matter takes a string.

## Billing

- **The calls bill to the maintainer's shared plan quota.** The trial's init event showed
  `apiKeySource` = none. The US-dollar figures are API-equivalent prices and not charges.
