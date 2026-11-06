# ruff: noqa: E501
"""Experiment 595 figures: total and refit share by phase, and the medians around five events.

Reuses experiment 587's figure code (`e00587_figures`) with its module-level names replaced: the five
arms (M0, D5, P5, C1, A1) and their colours, this experiment's docs folder, and a copy of 587's
common module whose `world_dir` and `ARMS` cover the arms. Run `e00595_score.py` first.
"""

from pathlib import Path
from types import SimpleNamespace

import altair as alt
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00587_common as c587  # ty: ignore[unresolved-import]
import e00587_figures as f  # ty: ignore[unresolved-import]
import e00595_common as common  # ty: ignore[unresolved-import]
import e00595_summary as summ  # ty: ignore[unresolved-import]
import polars as pl
from plotting import ocf_theme as theme

f.common = SimpleNamespace(**{k: getattr(c587, k) for k in dir(c587) if not k.startswith("__")})
f.common.world_dir = common.world_dir
f.common.ARMS = {a: ("", common.tag(arm=a)) for a in common.ALL_ARMS}
f.summ = summ
f.COLOUR = {
    "M0": theme.DATA_BLUE,
    "D5": theme.BRAND_ORANGE,
    "P5": theme.DATA_PURPLE,
    "C1": theme.DATA_GREEN,
    "A1": theme.DATA_SKY,
}
f.SCALE = alt.Scale(domain=list(f.COLOUR), range=list(f.COLOUR.values()))
f.ARM_COLOUR = alt.Color("arm:N", scale=f.SCALE, legend=alt.Legend(orient="top", title=None))


def docs_dir() -> Path:
    """Return the experiment's docs folder."""
    return next((Path(__file__).resolve().parents[2] / "docs/ml-experiments").glob("00595-*"))


f.docs_dir = docs_dir


def main() -> None:
    """Draw the share figure, the verified pair, and four injected events."""
    shares = pl.read_parquet(common.OUT / "scores" / "event_shares.parquet")
    f.share_figure(shares)
    obs = c510.clean_stack()
    verified = [("verified", e) for e in c587.case_events(case="verified")]
    f.examples_figure(
        name="figure_2_verified_pair.svg",
        title="Verified D5 and D16 transfer: daily mean of each arm's median forecast",
        panels=verified,
        obs=obs,
    )
    designs = c587.designs()
    picks = designs.filter(pl.col("start_day") == 3).filter(
        ((pl.col("depth_share_of_smaller_capacity") == 0.5) & (pl.col("length_days") == 15))
        | ((pl.col("depth_share_of_smaller_capacity") == 0.3) & (pl.col("length_days") == 7))
        | ((pl.col("depth_share_of_smaller_capacity") == 0.15) & (pl.col("length_days") == 30))
        | ((pl.col("depth_share_of_smaller_capacity") == 0.3) & (pl.col("length_days") == 15))
    )
    inj = [
        (f"inj{r['spec_id']}", c587.case_events(case=f"inj{r['spec_id']}")[0])
        for r in picks.iter_rows(named=True)
    ]
    f.examples_figure(
        name="figure_3_injected_examples.svg",
        title="Four injected load-shifts: the arms' median forecasts against the observed and the no-event power",
        panels=inj,
        obs=obs,
    )


if __name__ == "__main__":
    main()
