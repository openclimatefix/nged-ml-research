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

## Code review of the labelling, detector, and scoring code (Opus)

- **Accepted, both blockers.** The labelling script gained a `--primary` flag, and a no-call
  `--agreement` mode now compares the labels with the known answers.
- **Accepted, the end-day bug.** The episode table now writes the detector's own end day for an
  `era` or `unknown` candidate, so the training mask no longer holds out the wrong rows.
- **Accepted, the safety and measurement items.** Any unexpected error now stops the other worker,
  each log entry has start and finish times, a resume re-checks the billing route and the prompt
  version, and the account-wide utilisation is summarised before and after a run.
- **Accepted, the calendar keys in the prompt.** The prompt now uses 7-day trailing means, one
  climatology, and local clock hours, with no weekday split. `calendar_keys.md` lists what remains.
- **Accepted, two scorer faults.** A gate that cannot be evaluated now blocks the verdict, and the
  self-test prints the five real-row interval cells that differ from the stored file.
- **Not applied.** The check that the `era_id` values are causal was left as a minor item.

## Review of the D8 pilot section (Sonnet, numbers and prose)

- **Accepted.** The 85 label calls are 65 distinct checks plus 20 retries, the extrapolation is
  stated on both bases, the recovery table closes at 19 candidates, and the terms the section uses
  are defined.
- **Rejected, "in 4 years" for the year probe.** The probe result file holds no count of years, so
  the claim could not be verified and was dropped.
- **Rejected, renaming the section heading.** Renaming it would change the anchor slug.
- **Changed, the stop rule.** The page quotes the plan's pilot rule (fewer than 40% of eras labelled
  `era` within 7 days) and says that rule was written for the injected bank and does not bind D8.

## Code review of the attempt 2 scripts (Opus)

- **Accepted, three blockers.** Attempt 2 could have overwritten attempt 1's signal files, the go
  tokens were not truly single-use, and nothing kept attempt 2 on D8 under about US$4. Attempt 2
  now writes to its own files, claims a token by an atomic rename before the first call, and
  refuses any run that is not D8 with arm `llm` and the stated limits.
- **Accepted, the should-fix items.** The probe score now checks the prompt stamp and records the
  hash the calls were made with. The summary reports structure-only counts of the reply format. The
  leftover attempt 1 go files were renamed to `.used`, and attempt 1 now refuses to spend.
- **Accepted, the test fixes.** The attempt 2 probe test no longer reads the real data folder, the
  always-true assertion is replaced, and new tests cover the non-D10 refusal and the three blocker
  cases.
- **Not applied.** One minor item, the module-level constants reassigned through `globals()`, would
  need a refactor to a frozen dataclass.

## Review of the attempt 2 section (Sonnet, numbers and prose)

- **Accepted.** The claim that the format fix removed the invalid replies was an overclaim and is
  removed. The page now says that attempt 1 and attempt 2 are separate calls, that attempt 1's
  strict parser accepted 55 of 65 checks, and that attempt 2 still has 30 of 67 replies over 40
  words. The scaling time is about 24 minutes, not attempt 1's 35 minutes. The terms the section
  uses are defined.
- **Rejected, adding the strict first-reply split (34, 1, and 30).** The page reports the two counts
  that matter for the comparison, and the split is in the call-log summary.
