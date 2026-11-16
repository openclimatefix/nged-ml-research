# ruff: noqa: E501, I001
"""Experiment 701, GAM side: the monthly T-all refits at the cuts 2026-07-01, 2026-08-01, 2026-09-01, 2026-10-01.

    E00377_SOURCE=control PYTHONPATH=<folders of 377, 700, 701> DATA_PATH_INTERNAL=<sandbox> \
        uv run --no-sync python ml-experiments/00701-*/e00701_gam_refit.py --sites D5 D9
    ... --check-cut 2026-06-01     # refit at a stored cut, compare with the stored fit, write nothing

For each site and cut this is experiment 377's `fit_tall` (T-all, difference-penalty weight 1.0,
experiment 114's design) on the 24 months before the cut, with the cache of `e00701_gam_cache.py`,
the pinned PV capacity of experiment 45's `frozen` vintage rule at the cut (zero at the eight MVA-only
Primaries), and no row that ends after the cut (`dp_gam` raises on lookahead). The new fits are
appended to the sandbox copy of experiment 377's `tall_fits/monthly/<site>.pkl`, which holds the
stored fits to the cut 2026-06-01 (the file is rewritten whole, atomically). Re-running replaces the
fits of the same cuts. The stored file is never touched.

Experiment 45's capacity vintages end at the origin Monday 2026-08-03 (weeks to 2026-07-27). At the
cuts 2026-09-01 and 2026-10-01 the signed-MW Primaries therefore hold the capacity of that vintage,
the median of its last 13 weeks up to the cut, as the stored rule does after a vintage's last week.
The script prints the source, origin, and held capacity of each fit.
"""

import argparse
import json
import pickle
import time
from datetime import UTC, datetime

import e00701_gam_extend377  # noqa: F401  (dates and paths first)
import e00701_gam_extend as ext
import e00377_common as common  # ty: ignore[unresolved-import]
import e00377_tall as tall  # ty: ignore[unresolved-import]
import numpy as np
import torch  # ty: ignore[unresolved-import]
import itertools


def main() -> None:
    """Refit each site at each cut and append to its monthly fit list."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", required=True)
    parser.add_argument(
        "--check-cut", help="refit at this stored cut (YYYY-MM-DD), compare, and write nothing"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    cache0 = common.load_cache()
    cuts = (
        [datetime.fromisoformat(args.check_cut).replace(tzinfo=UTC)]
        if args.check_cut
        else ext.MONTHS
    )
    report = {}
    for site in args.sites:
        cache = common.variant_cache(cache=cache0, site=site)
        vintages = common.vintages_for(cache=cache)
        fits = tall.load_fits(cadence="monthly", site=site)
        new = []
        for cut in cuts:
            started = time.monotonic()
            fit = tall.fit_tall(
                cache=cache,
                vintages=vintages,
                site=site,
                cut=cut,
                penalty_weight=common.T_ALL_LAMBDA,
            )
            seconds = time.monotonic() - started
            info: dict[str, float | str] = {
                "seconds": seconds,
                "n_theta": int(np.asarray(fit.fit.theta).size),
            }
            if fit.pinned is not None:
                info |= {
                    "capacity_source": fit.pinned.source,
                    "vintage_origin": str(fit.pinned.origin),
                    "held_points": float(fit.pinned.held_points),
                }
            if args.check_cut:
                stored = next(f for f in fits if f.cut == cut)
                info["max_abs_theta_diff_vs_stored"] = float(
                    np.max(np.abs(np.asarray(fit.fit.theta) - np.asarray(stored.fit.theta)))
                )
            report[f"{site}/{cut:%Y-%m-%d}"] = info
            print(site, f"{cut:%Y-%m-%d}", info, flush=True)
            new.append(fit)
        if args.check_cut:
            continue
        kept = [f for f in fits if f.cut not in {n.cut for n in new}]
        merged = sorted([*kept, *new], key=lambda f: f.cut)
        assert all(b.cut > a.cut for a, b in itertools.pairwise(merged)), "cuts must ascend"
        target = tall.fit_path(cadence="monthly", site=site)
        assert ext.SANDBOX in target.parents, f"{target} is outside the sandbox"
        temporary = target.with_name(f"{target.name}.tmp")
        temporary.write_bytes(pickle.dumps(merged))
        temporary.rename(target)
    ext.TIMING.mkdir(parents=True, exist_ok=True)
    (
        ext.TIMING
        / f"gam_refit_{common.SOURCE}_{'check_' if args.check_cut else ''}{'_'.join(args.sites)}.json"
    ).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
