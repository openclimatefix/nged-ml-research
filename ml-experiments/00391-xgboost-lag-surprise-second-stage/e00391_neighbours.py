"""Build the neighbour list for the detector's veto from NGED's own interconnection table.

Reads NGED's `Primary Substation Interconnections.csv` (pairs of substation numbers) and
`Monitor Direction.csv` (which maps each substation number to a time series), and maps the metered
Primaries to their D-codes with experiment 298's `SERIES_TO_SITE` (copied below). Prints the
neighbours of each of the 16 Primaries, the Primaries with none listed, and the differences from
experiment 298's static list. Writes the list to `work/neighbours_nged.json`. NGED gives D15 and
D18 one shared substation number, so the two series share their interconnections here.

```bash
PYTHONPATH=ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations:\
ml-experiments/00383-* DATA_PATH_INTERNAL=... \
    uv run --no-sync python ml-experiments/00383-*/e00391_neighbours.py
```
"""

import csv
import json
from typing import Final

import e00391_stage1 as fit
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SERIES_TO_SITE: Final[dict[int, str]] = {
    3: "D3", 7: "D9", 8: "D11", 9: "D19", 10: "D10", 11: "D4", 12: "D15", 13: "D18",
    14: "D12", 15: "D22", 16: "D14", 17: "D5", 18: "D16", 24: "D21", 25: "D2", 26: "D8",
}  # fmt: skip
"""Experiment 298's `SERIES_TO_SITE`, copied because importing its module needs PyTorch."""
ADJACENCY_SERIES: Final[dict[int, tuple[int, ...]]] = {
    3: (15, 16, 26), 7: (24,), 8: (10, 14, 24), 9: (11, 24), 10: (8, 12, 13, 14, 24, 25),
    11: (9, 25), 12: (10, 13, 14, 16, 25), 13: (10, 12, 14, 16, 25), 14: (8, 10, 12, 13, 15, 16),
    15: (3, 14, 17, 18), 16: (3, 12, 13, 14, 26), 17: (15, 18, 26), 18: (15, 17),
    24: (7, 8, 9, 10, 25), 25: (10, 11, 12, 13, 24), 26: (3, 16, 17),
}  # fmt: skip
"""Experiment 298's `ADJACENCY_SERIES` (experiment 43's list), copied for the same reason."""
OLD_NEIGHBOURS: Final[dict[str, tuple[str, ...]]] = {
    SERIES_TO_SITE[a]: tuple(SERIES_TO_SITE[b] for b in bs) for a, bs in ADJACENCY_SERIES.items()
}
NGED_DIR: Final = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "substation_hierarchy_2026-10-15"


def nged_neighbours() -> tuple[dict[str, tuple[str, ...]], dict[str, int]]:
    """Return each metered Primary's metered neighbours and its count of unmetered neighbours."""
    with (NGED_DIR / "Monitor Direction.csv").open() as handle:
        monitors = list(csv.DictReader(handle))
    sites_of_number: dict[str, set[str]] = {}
    for r in monitors:
        if int(r["TimeSeriesID"]) in SERIES_TO_SITE:
            sites_of_number.setdefault(r["SubstationNumber"], set()).add(
                SERIES_TO_SITE[int(r["TimeSeriesID"])]
            )
    with (NGED_DIR / "Primary Substation Interconnections.csv").open() as handle:
        pairs = [(r["PrimarySubstation1"], r["PrimarySubstation2"]) for r in csv.DictReader(handle)]
    all_sites = {site for sites in sites_of_number.values() for site in sites}
    metered: dict[str, set[str]] = {site: set() for site in all_sites}
    unmetered: dict[str, int] = dict.fromkeys(metered, 0)
    for first, second in pairs:
        for here, there in ((first, second), (second, first)):
            for site in sites_of_number.get(here, ()):
                if there in sites_of_number:
                    metered[site] |= sites_of_number[there]
                else:
                    unmetered[site] += 1
    order = sorted(metered, key=lambda s: int(s[1:]))
    return {s: tuple(sorted(metered[s], key=lambda x: int(x[1:]))) for s in order}, unmetered


def main() -> None:
    """Print the list, the gaps, and the comparison with experiment 298's list."""
    new, unmetered = nged_neighbours()
    print("site: NGED metered neighbours | unmetered interconnections | old list")
    for site, neighbours in new.items():
        print(f"{site}: {list(neighbours)} | {unmetered[site]} | {list(OLD_NEIGHBOURS[site])}")
    print("Primaries with no listed metered neighbour:", [s for s, n in new.items() if not n])
    print(
        "Primaries with no listed interconnection at all:",
        [s for s, n in new.items() if not n and not unmetered[s]],
    )
    only_new = {s: sorted(set(new[s]) - set(OLD_NEIGHBOURS[s])) for s in new}
    only_old = {s: sorted(set(OLD_NEIGHBOURS[s]) - set(new[s])) for s in new}
    print("in NGED list only:", {s: v for s, v in only_new.items() if v})
    print("in the old list only:", {s: v for s, v in only_old.items() if v})
    fit.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (fit.WORK_DIR / "neighbours_nged.json").write_text(json.dumps(new, indent=1))


if __name__ == "__main__":
    main()
