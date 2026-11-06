"""Experiment 599: absorbed shares of E0, E1, E2w, and E3, with experiment 587's share definitions.

With `s` the sign of the event's depth at the Primary and `D` its absolute depth,

    total = s * (q[E] - q[N]) / D        refit = s * (q[E_on_N] - q[N]) / D

for a forecast quantile `q`, averaged over seeds and half-hours, by phase (as experiment 595).
`score_case` of experiment 587 is reused unchanged: it reads `common.world_dir`, `common.ARMS`, and
`common.LAG_FREE`, so this module hands it a copy of 587's common module with those replaced.
E0 is read from experiment 595's A1 folder (`e00599_common.world_dir`).
"""

import e00587_common as c587  # ty: ignore[unresolved-import]
import e00587_score as s587  # ty: ignore[unresolved-import]
import e00599_common as common  # ty: ignore[unresolved-import]
import polars as pl

s587.common = type("Common", (), {k: getattr(c587, k) for k in dir(c587) if not k.startswith("__")})
s587.common.world_dir = common.world_dir
s587.common.ARMS = dict.fromkeys(common.ALL_ARMS, ("", common.TAG))
s587.common.LAG_FREE = frozenset()


def main() -> None:
    """Score every arm on every case and write `scores/event_shares.parquet`."""
    cases = ["verified", *[f"inj{i}" for i in c587.designs()["spec_id"].to_list()]]
    frames, dropped = [], 0
    for arm in common.ALL_ARMS:
        for case in cases:
            table, n = s587.score_case(arm=arm, case=case)
            frames.append(table)
            dropped += n
    out = common.OUT / "scores"
    out.mkdir(parents=True, exist_ok=True)
    shares = pl.concat(frames, how="diagonal_relaxed")
    shares.write_parquet(out / "event_shares.parquet")
    print(
        shares.height,
        "rows;",
        dropped,
        "cells dropped for fewer than",
        s587.MIN_HALF_HOURS,
        "half-hours",
    )


if __name__ == "__main__":
    main()


def noise_floor() -> pl.DataFrame:
    """Return the q50 difference of the `E0chk` smoke refit on `obs` from 595's stored A1."""
    rows = []
    for f in sorted((common.OUT / "work/E0chk/obs/work/fits" / common.TAG).glob("*.parquet")):
        theirs = common.world_dir(arm="E0", world="obs") / "work/fits" / common.TAG / f.name
        j = pl.read_parquet(f).join(pl.read_parquet(theirs), on=s587.KEYS[2:], suffix="_s")
        d = (j["q50"] - j["q50_s"]).abs()
        rows.append(
            {"file": f.name, "rows": j.height, "max": float(d.max()), "mean": float(d.mean())}
        )  # ty: ignore[invalid-argument-type]
    return pl.DataFrame(rows)
