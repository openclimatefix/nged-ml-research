# ruff: noqa: E501
"""Scale experiment 430's stored forecasts by the trailing efficiency and aggregate the losses.

Each arm is a network, a source of expected power, a method, a multiplier kind (raw or
normalised), and a delivery lag. The multiplier of a forecast row is the method's value on the
last allowed day, per farm and refit. Losses are aggregated per month, farm, and band, so the
contrasts need no row-level storage.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00430-unified-forecaster-solar-and-pretraining-ablations:ml-experiments/00464-solar-efficiency-history-stage-0-post-hoc-scaling \
        uv run --no-sync python ml-experiments/00464-*/e00464_score.py
"""

from dataclasses import dataclass

import e00430_score as s430  # ty: ignore[unresolved-import]
import e00464_common as common
import polars as pl

QCOLS: list[str] = s430.QCOLS
METRICS: list[str] = [
    "score",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
    "cover01",
    "cover05",
    "cover95",
    "cover99",
    "median_minus_mean",
    "ae_mean",
    "ae_median",
]


@dataclass(frozen=True)
class Arm:
    """One scaled forecast: network, expected-power source, method, multiplier kind, and lag."""

    network: str
    source: str
    method: str
    normalised: bool
    lag_days: int

    @property
    def name(self) -> str:
        """Return the arm's label, used as its file name."""
        kind = "norm" if self.normalised else "raw"
        return f"{self.network}|{self.source}|{self.method}|{kind}|lag{self.lag_days}"


def arm_list() -> list[Arm]:
    """Return every arm: primary, replication, source (a), raw, and 6-day-lag arms."""
    arms: list[Arm] = []
    for method in common.METHODS:
        for source in common.PRIMARY_SOURCES:
            arms.append(Arm("a2_member", source, method, True, common.DELIVERY_DAYS))
            arms.append(Arm("a2_member", source, method, True, 6))
            arms.append(Arm("g0_onepass", source, method, True, common.DELIVERY_DAYS))
        arms.append(Arm("a2_member", "a", method, True, common.DELIVERY_DAYS))
        arms.append(Arm("a2_member", "c", method, False, common.DELIVERY_DAYS))
    return arms


def add_refit(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `refit` (the latest refit date at or before the row's origin) and `init_date`."""
    refit = pl.lit(None, dtype=pl.Datetime("us", "UTC"))
    for r in common.REFITS:
        refit = pl.when(pl.col("init_time") >= r).then(pl.lit(r)).otherwise(refit)
    return frame.with_columns(refit=refit, init_date=pl.col("init_time").dt.date())


def multipliers(*, base: pl.DataFrame, arm: Arm) -> pl.DataFrame:
    """Return the base frame's keys with the arm's multiplier and a `missing` flag."""
    est = (
        pl.read_parquet(common.RATIO_DIR / f"estimates_{arm.source}.parquet")
        .select("farm", "refit", "date", "reference", pl.col(arm.method).alias("value"))
        .with_columns(pl.col("refit").dt.convert_time_zone("UTC"))
    )
    keys = (
        base.select("time_series_id", "refit", "init_date")
        .unique()
        .with_columns(date=pl.col("init_date") - pl.duration(days=arm.lag_days))
        .join(
            est,
            left_on=["time_series_id", "refit", "date"],
            right_on=["farm", "refit", "date"],
            how="left",
        )
    )
    value = pl.col("value")
    if arm.normalised:
        value = value / pl.col("reference")
    lo, hi = common.MULTIPLIER_RANGE
    return keys.with_columns(
        missing=value.is_null() | value.is_nan(),
        multiplier=value.fill_nan(None).fill_null(1.0).clip(lo, hi),
    ).select("time_series_id", "refit", "init_date", "multiplier", "missing")


def scale(*, base: pl.DataFrame, arm: Arm, caps: pl.DataFrame) -> pl.DataFrame:
    """Return the base forecasts with every quantile multiplied and capped; add `missing`, `clipped`."""
    assert base["refit"].null_count() == 0
    mult = multipliers(base=base, arm=arm)
    joined = (
        base.join(mult, on=["time_series_id", "refit", "init_date"], how="left")
        .join(caps, on=["time_series_id", "refit"], how="left")
        .with_columns(pl.col("multiplier").fill_null(1.0), pl.col("missing").fill_null(value=True))
    )
    scaled = joined.with_columns(
        [pl.min_horizontal(pl.col(q) * pl.col("multiplier"), pl.col("cap")).alias(q) for q in QCOLS]
    )
    return scaled.with_columns(
        clipped=(joined["q99"] * joined["multiplier"] > joined["cap"]),
        arm=pl.lit(arm.name),
    )


def aggregate(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per month, farm, and band: row count, mean of each loss, missing and clipped shares."""
    rows = s430.losses(frame=frame.drop("missing", "clipped")).join(
        frame.select(*s430.KEYS, "missing", "clipped"), on=s430.KEYS
    )
    return rows.group_by("month", "time_series_id", "band").agg(
        n=pl.len(),
        missing=pl.col("missing").mean(),
        clipped=pl.col("clipped").mean(),
        **{m: pl.col(m).mean() for m in METRICS},
    )


def main() -> None:
    """Score the unscaled references and every arm; write one aggregate table per arm."""
    out_dir = common.SCORE_DIR / "agg"
    out_dir.mkdir(parents=True, exist_ok=True)
    bases = {}
    for network in common.NETWORKS:
        base = add_refit(frame=s430.load_mine(arm=network))
        base = base.with_columns(refit=pl.col("refit").dt.convert_time_zone("UTC"))
        caps = base.group_by("time_series_id", "refit").agg(cap=pl.col("q99").max())
        bases[network] = (base, caps)
        ref = base.with_columns(
            missing=pl.lit(value=False),
            clipped=pl.lit(value=False),
            arm=pl.lit(f"{network}|unscaled"),
        )
        aggregate(frame=ref).write_parquet(out_dir / f"{network}__unscaled.parquet")
        print(network, "unscaled rows", base.height, flush=True)
    for arm in arm_list():
        path = out_dir / (arm.name.replace("|", "__") + ".parquet")
        if path.exists():
            continue
        base, caps = bases[arm.network]
        aggregate(frame=scale(base=base, arm=arm, caps=caps)).write_parquet(path)
        print("done", arm.name, flush=True)


if __name__ == "__main__":
    main()
