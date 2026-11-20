"""List the (arm, fold, seed) fits of experiment 711, count them, and estimate the wall time.

Every job is one fit, run by one invocation of `e00711_fit.py` with a fixed seed. The generator
reads no data and runs no fit.

**The count follows the plan's job table** (946 fits: 423 Primary demand on 12 recent folds, 84 in
the 7-fold pairs, 180 in the replicate, and 259 for the solar farms). **The wall-time estimate
follows plan section 10.** A Primary demand fit took 62 s of GPU time in experiment 707's logs, and
up to twice that with data loading. The solar fit time rests on experiment 666's log, with the
window 3 times longer, so the estimate is 125 s per fit and up to twice that. Every figure is an
estimate until the first measuring fit replaces it.

```bash
# environment as in e00711_common.py
uv run --no-sync python ml-experiments/00711-*/e00711_jobs.py                      # counts and time
uv run --no-sync python ml-experiments/00711-*/e00711_jobs.py --write jobs.tsv  # one job per line
uv run --no-sync python ml-experiments/00711-*/e00711_jobs.py --pending            # skip finished
```
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00711_arms as arms

STREAMS: Final[int] = 2
SECONDS_PER_FIT: Final[dict[arms.TargetType, tuple[float, float]]] = {
    "primary": (62.0, 124.0),
    "solar": (125.0, 250.0),
}
"""The low and high estimates of a stream's seconds per fit, by target."""
PLAN_TOTALS: Final[dict[str, int]] = {
    "primary_recent12": 423,
    "primary_pairs": 84,
    "primary_replicate": 180,
    "solar": 259,
}
"""The plan's job counts, which `check_counts` compares with the table's."""


@dataclass(frozen=True)
class Job:
    """One fit: an arm on a fold with a fitting seed."""

    arm: str
    month: str
    seed: int

    @property
    def set_name(self) -> str:
        """Return the name of the job's prediction set."""
        return arms.set_name(arm=self.arm, month=self.month, seed=self.seed)

    def command_arguments(self) -> str:
        """Return the arguments that `e00711_fit.py` takes for this job."""
        return f"--arm {self.arm} --fold {self.month} --seed {self.seed}"


def plan_jobs(*, names: list[str] | None = None) -> list[Job]:
    """Return every job of the named arms (default: every arm that needs fits).

    Jobs are ordered by fold, then arm, then seed, so one stream reads one fold's rows while it
    works through the arms of that fold.

    Args:
        names: Arm names, or `None` for all of the fitted arms.

    Returns:
        One job per fold and seed of each arm.

    Raises:
        ValueError: If a named arm is stored (it has no fit).
    """
    selected = arms.fitted_arms() if names is None else [arms.ARMS[n] for n in names]
    if any(a.stored for a in selected):
        msg = "a stored arm has no fit"
        raise ValueError(msg)
    jobs = [Job(arm=a.name, month=m, seed=s) for a in selected for m in a.months for s in a.seeds]
    return sorted(jobs, key=lambda j: (j.month, j.arm, j.seed))


PAIR_ARMS: Final[tuple[str, ...]] = ("D0g", "D1g", "K1", "K1p")
"""The arms that the plan counts as 7-fold pairs; `D3` is counted with the 12 recent folds."""


def counts_by_group(*, jobs: list[Job]) -> dict[str, int]:
    """Return the number of jobs in each of the plan's four groups."""
    out = dict.fromkeys(PLAN_TOTALS, 0)
    for job in jobs:
        arm = arms.ARMS[job.arm]
        if arm.target == "solar":
            out["solar"] += 1
        elif arm.folds == "replicate12":
            out["primary_replicate"] += 1
        elif job.arm in PAIR_ARMS:
            out["primary_pairs"] += 1
        else:
            out["primary_recent12"] += 1
    return out


def check_counts(*, jobs: list[Job]) -> None:
    """Raise if the table's job counts differ from the plan's.

    Raises:
        ValueError: If a group's count differs from `PLAN_TOTALS`.
    """
    found = counts_by_group(jobs=jobs)
    if found != PLAN_TOTALS:
        msg = f"job counts {found} differ from the plan's {PLAN_TOTALS}"
        raise ValueError(msg)


def stream_hours(*, jobs: list[Job]) -> tuple[float, float]:
    """Return the low and high estimates of the total stream-hours of the jobs."""
    low = sum(SECONDS_PER_FIT[arms.ARMS[j.arm].target][0] for j in jobs)
    high = sum(SECONDS_PER_FIT[arms.ARMS[j.arm].target][1] for j in jobs)
    return low / 3600.0, high / 3600.0


def wall_hours(*, jobs: list[Job], streams: int = STREAMS) -> tuple[float, float]:
    """Return the low and high estimates of wall-clock hours on `streams` concurrent streams."""
    low, high = stream_hours(jobs=jobs)
    return low / streams, high / streams


def pending(*, jobs: list[Job], finished: set[str]) -> list[Job]:
    """Return the jobs whose prediction set is not in `finished` (a set of set names)."""
    return [j for j in jobs if j.set_name not in finished]


def summary_lines(*, jobs: list[Job]) -> list[str]:
    """Return the report: the count per group and arm, and the two-stream wall-clock estimate."""
    lines = [f"{len(jobs)} jobs (plan: {sum(PLAN_TOTALS.values())}): {counts_by_group(jobs=jobs)}"]
    for arm in arms.fitted_arms():
        n = sum(1 for j in jobs if j.arm == arm.name)
        lines.append(
            f"  {arm.name:4s} {n:4d} fits  {len(arm.months)} folds x {len(arm.seeds)} seeds"
        )
    low, high = stream_hours(jobs=jobs)
    wlow, whigh = wall_hours(jobs=jobs)
    lines.append(
        f"estimate (not measured): {low:.1f} to {high:.1f} stream-hours; on {STREAMS} streams "
        f"{wlow:.1f} to {whigh:.1f} hours of wall time"
    )
    return lines


def write_job_list(*, jobs: list[Job], path: Path) -> None:
    """Write one tab-separated line per job: arm, fold, seed, prediction-set name."""
    path.write_text("".join(f"{j.arm}\t{j.month}\t{j.seed}\t{j.set_name}\n" for j in jobs))


def main() -> None:
    """Print the count and the estimate, and optionally write the job list."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--write", type=Path, default=None, help="write one job per line")
    parser.add_argument("--pending", action="store_true", help="drop jobs whose set exists")
    args = parser.parse_args()
    arms.check_table()
    all_jobs = plan_jobs()
    check_counts(jobs=all_jobs)
    jobs = plan_jobs(names=args.arms)
    if args.pending:
        from e00711_common import WORK_DIR

        folder = WORK_DIR / "predictions"
        done = {p.name for p in folder.iterdir()} if folder.exists() else set()
        jobs = pending(jobs=jobs, finished=done)
    print("\n".join(summary_lines(jobs=jobs)))
    if args.write is not None:
        write_job_list(jobs=jobs, path=args.write)
        print(f"wrote {len(jobs)} jobs to {args.write}")


if __name__ == "__main__":
    main()
