"""Experiment 732: a read-only diagnostic of the half-hours that fail the solar-target check.

Run `e00732_cli.py solar_diag`. The diagnostic rebuilds the batches of the tensor stage, finds every
sun-up half-hour of a clear solar-world day whose solar target is not above zero, and classifies it.
It prints counts only. The segment and day of each failing half-hour go to `PRIVATE/solar_diag.log`.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import e00729_banks as banks729
import e00730_banks as banks730
import e00732_arms as arms
import e00732_core as core
import e00732_overlays as ov
import e00732_stages as stages
import numpy as np

MID = arms.MID
T = arms.T
ROW = arms.ROW
DAWN_COSZ: float = 0.2
"""A failing half-hour with a cosine of the zenith angle below this counts as dawn or dusk."""


def classify(
    *,
    batch: arms.Batch,
    overlays: Sequence[ov.TransferOverlay],
    originals: Sequence,
) -> tuple[Counter, Counter, list[str]]:
    """Return the class counts, the base counts, and the private log lines of one batch.

    Args:
        batch: The batch of the tensor check.
        overlays: The transfer overlays, in the order of the prepared segments.
        originals: The stored segments, in the same order.

    Returns:
        The counts of failing half-hours by class, the counts of all checked half-hours by source,
        and one private line per failing half-hour.
    """
    fail: Counter = Counter()
    seen: Counter = Counter()
    lines: list[str] = []
    for k, (si, day) in enumerate(batch.picks):
        seg = overlays[si].seg
        if seg.world not in ov.SOLAR_WORLDS:
            continue
        scored = slice(MID, MID + 48)
        up = (batch.x[k, ROW["cosz"], scored] > core.SUN_UP_COSZ) & (batch.t[k, T["valid"]] > 0.5)
        csi = batch.x[k, ROW["csi"], scored]
        if not up.any() or csi[up].mean() < core.CLEAR_DAY_CSI:
            continue
        tensor = batch.t[k, T["solar"]]
        overlay = seg.solar[day]
        stored = originals[si].solar[day]
        cosz = batch.x[k, ROW["cosz"], scored]
        fallback = bool(seg.s1_fallback[day])
        day_all_zero = not (overlay > 0).any()
        for h in np.flatnonzero(up):
            seen[seg.source] += 1
            if tensor[h] > 0.0:
                continue
            tags = [
                "masked by the unreliable rule"
                if ov.unreliable_solar(seg=seg)[day, h]
                else "not masked",
                "day has any positive output"
                if (overlay > 0).any()
                else "no positive output that day",
                f"source {seg.source}",
                "fallback day (no farm power or sky)"
                if fallback
                else "day left its drawn source: no",
                "whole day zero" if day_all_zero else "zero inside a day with output",
                "dawn or dusk (cosz below 0.2)" if cosz[h] < DAWN_COSZ else "cosz at least 0.2",
                "CAMS index missing" if not np.isfinite(csi[h]) else "CAMS index present",
                "zero in the overlay arrays" if overlay[h] <= 0.0 else "zero only in the tensor",
                "zero in the stored segment"
                if stored[h] <= 0.0
                else "positive in the stored segment",
            ]
            lo, hi = int(h), int(h)
            while lo > 0 and overlay[lo - 1] <= 0.0 and cosz[lo - 1] > core.SUN_UP_COSZ:
                lo -= 1
            while hi < 47 and overlay[hi + 1] <= 0.0 and cosz[hi + 1] > core.SUN_UP_COSZ:
                hi += 1
            run = hi - lo + 1
            tags.append(
                "zero run of 1 half-hour" if run == 1
                else "zero run of 2 to 8 half-hours" if run <= 8
                else "zero run of more than 8 half-hours"
            )  # fmt: skip
            for t in tags:
                fail[t] += 1
            fail["total"] += 1
            lines.append(f"segment {si} day {day} half-hour {int(h)}: " + "; ".join(tags))
    return fail, seen, lines


def format_counts(*, name: str, fail: Mapping[str, int], seen: Mapping[str, int]) -> list[str]:
    """Return the printed lines of one bank: counts only."""
    out = [f"{name}: {fail.get('total', 0)} failing of {sum(seen.values())} checked half-hours"]
    out += [f"  {k}: {v}" for k, v in sorted(fail.items()) if k != "total"]
    out += [f"  checked, {k}: {v}" for k, v in sorted(seen.items())]
    return out


def run(
    *,
    root: Path,
    banks: Mapping[str, tuple],
    log: Callable[[str], None],
) -> dict[str, dict[str, int]]:
    """Classify the failing half-hours of each bank, print counts, and write the private log.

    Args:
        root: The data folder of this experiment.
        banks: For each bank name, `(batch, overlays, originals)`.
        log: Receives the counts.

    Returns:
        The class counts of each bank.
    """
    private = root / "PRIVATE"
    private.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[str, int]] = {}
    private_lines: list[str] = []
    for name, (batch, overlays, originals) in banks.items():
        fail, seen, lines = classify(batch=batch, overlays=overlays, originals=originals)
        result[name] = dict(fail)
        private_lines += [f"{name} {x}" for x in lines]
        for line in format_counts(name=name, fail=fail, seen=seen):
            log(line)
    (private / "solar_diag.log").write_text("\n".join(private_lines) + "\n")
    return result


def report_mask(
    *,
    where: str,
    segments: Sequence,
    batch: arms.Batch,
    prepared: Sequence,
    log: Callable[[str], None],
) -> None:
    """Print the unreliable share of a bank and the solar-target check.

    The check runs with and without the mask. The output holds counts and shares only.
    """
    bad, total = ov.unreliable_counts(segments=segments)
    checks = arms.check_solar_target(batch=batch, prepared=prepared)
    masked, unmasked = (
        checks["solar_target_positive"],
        checks["info_solar_target_positive_unmasked"],
    )
    log(
        f"{where}: unreliable {bad} of {total} S1 sun-up half-hours of solar worlds "
        f"({bad / max(total, 1):.4f}); solar target positive on the rest {masked.value:.4f} "
        f"(n {masked.n}), unmasked {unmasked.value:.4f} (n {unmasked.n})"
    )


def stage_solar_diag(
    *,
    root: Path,
    root730: Path,
    env: stages.Env,
    log: Callable[[str], None],
    seed: int = 0,
) -> dict[str, dict[str, int]]:
    """Rebuild the picks of the tensor stage for U* and classify the failing solar half-hours.

    Args:
        root: The data folder of this experiment.
        root730: Experiment 730's data folder (read only).
        env: The stages' environment.
        log: Receives the counts.
        seed: The seed of the training batch.

    Returns:
        The class counts of the train and validation banks.
    """
    spec = core.ARMS["U*"]
    banks: dict[str, tuple] = {}
    for where, kind in (("train", "train"), ("validation", "validation")):
        path = banks730.bank_path(root=root730, kind=kind, seed=0 if kind == "train" else None)
        bank = banks729.load_bank(path=path)
        extra = {"seed": 0} if kind == "train" else {}
        overlays = ov.overlay_segments(
            segments=bank.segments, kind=kind, neighbours=env.neighbours, **extra
        )
        prepared = arms.prepare(
            overlays=overlays, holidays=env.holidays, stats=env.stats, medians=env.medians
        )
        picks = (
            stages.mirror_edge_picks(prepared=prepared, limit=80)
            + stages.solar_world_picks(prepared=prepared, limit=80)
            + stages.bsp_present_solar_picks(prepared=prepared, limit=80)
        )
        if where == "train":
            index = ov.build_pair_index(segments=bank.segments)
            batch = arms.make_batch(
                prepared=prepared,
                picks=picks,
                spec=spec,
                rng=np.random.default_rng(seed),
                index=index,
            )
        else:
            batch = arms.make_eval_batch(prepared=prepared, picks=picks, spec=spec)
        banks[where] = (batch, overlays, bank.segments)
        report_mask(where=where, segments=bank.segments, batch=batch, prepared=prepared, log=log)
    return run(root=root, banks=banks, log=log)
