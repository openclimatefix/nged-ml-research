"""Experiment 696: job-hours per arm and the resource log's peaks, from the queue's job logs.

Reads `queue_logs/job_*.out` (each ends with `ALLDONE <seconds>s`) and `queue_logs/resource_log.txt`
under this experiment's data folder, and prints the job-hours per arm, the total, the span of the
resource log, and its peak GPU memory.
"""

import collections
import re

from studies.sources import ML_EXPERIMENTS_DATA_DIR


def main() -> None:
    """Print the job-hours per arm and the resource log's span and peak GPU memory."""
    root = next(ML_EXPERIMENTS_DATA_DIR.glob("00696-*")) / "queue_logs"
    seconds: dict[str, float] = collections.defaultdict(float)
    for path in sorted(root.glob("job_*.out")):
        found = re.search(r"ALLDONE (\d+)s", path.read_text())
        if found:
            seconds[path.name.split("_")[1]] += int(found.group(1))
    for arm in sorted(seconds):
        print(f"{arm}: {seconds[arm] / 3600:.2f} job-hours")
    print(f"total: {sum(seconds.values()) / 3600:.1f} job-hours")
    lines = (root / "resource_log.txt").read_text().splitlines()
    rows = [line.split() for line in lines if line.strip()]
    gpu = [int(row[1].split(",")[0]) for row in rows if row[1].split(",")[0].isdigit()]
    print(f"resource log: {rows[0][0]} to {rows[-1][0]} UTC, peak GPU memory {max(gpu)} MiB")


if __name__ == "__main__":
    main()
