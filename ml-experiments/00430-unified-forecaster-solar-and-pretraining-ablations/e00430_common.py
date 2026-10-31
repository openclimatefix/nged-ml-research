# ruff: noqa: D102, E501
"""Shared constants and arm definitions for experiment 430 (solar and pretraining ablations).

Farms appear only as labels A to F. Every table written here lives under `data/ml-experiments/`.
The data layout is experiment 310's (one table per farm and forecast day, rows of one 00 UTC
ensemble run), extended by `e00430_members.py` with the 51 members' shortwave and temperature.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00430-unified-forecaster-solar-and-pretraining-ablations"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
MEMBER_DIR: Final[Path] = DATA_DIR / "member_arrays"
PRED_DIR: Final[Path] = DATA_DIR / "predictions"
E310_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00310-metered-farm-long-lead-scoring"
E345_PRED_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00345-pooled-solar-wind-mixture-density-network"
    / "solar_predictions"
)

N_MEMBERS: Final[int] = 51
SEEDS_GATE: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
"""Experiment 345's solar-only pooled network used five seeds; the gate reproduces all five."""
SEEDS_ABLATION: Final[tuple[int, ...]] = (0, 1, 2)
"""Screening seeds of every ablation arm (the design note: three when screening)."""
SEEDS_SECOND: Final[tuple[int, ...]] = (3, 4, 5)
"""A second seed set for contrasts smaller than the minimum detectable effect."""
PRETRAIN_END: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""ERA5 and CAMS rows before this time pretrain; the ensemble store starts here."""
FIXED_STATS_CUTOFF: Final[datetime] = datetime(2025, 8, 31, tzinfo=UTC)
"""Standardiser rows end here for the arms that need one standardiser across all four refits."""
FINE_TUNE_LEARNING_RATE: Final[float] = 1e-3
"""Half the from-scratch rate of experiment 345, since the weights start from a pretrained network."""
EPOCHS: Final[int] = 20
BATCH: Final[int] = 2048
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-4
VALIDATION_DAYS: Final[int] = 61
PLANT_TILT_DEG: Final[float] = 20.0
"""The farms fit 15 to 20 degrees in experiment 282; the plant arms start here."""
PLANT_AZIMUTH_DEG: Final[float] = 180.0
MEMBER_CHANNEL_SCALE: Final[tuple[float, ...]] = (500.0, 500.0, 200.0, 10.0)
"""Divisors of the member channels (shortwave, beam, diffuse, temperature) to give inputs unit size."""

SHARED: Final[tuple[str, ...]] = (
    "temperature",
    "wind10",
    "wind100",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
)
SOLAR_ONLY: Final[tuple[str, ...]] = (
    "ghi_mean",
    "ghi_std",
    "dni_mean",
    "dhi_mean",
    "elevation_deg",
    "azimuth_sin",
    "azimuth_cos",
    "relative_humidity",
    "longwave",
    "precipitation",
    "pressure_surface",
    "pressure_msl",
    "z500",
    "csi",
    "lead_hours",
)
GATE_COLUMNS: Final[tuple[str, ...]] = (*SHARED, *SOLAR_ONLY, "temp_missing")
"""Experiment 345's `solar_pooled` column list, in its order."""
AVAILABLE_IN_ERA5: Final[frozenset[str]] = frozenset(
    {
        "temperature",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "ghi_mean",
        "dni_mean",
        "dhi_mean",
        "elevation_deg",
        "azimuth_sin",
        "azimuth_cos",
        "csi",
        "lead_hours",
        "clear_sky",
        "temp_missing",
    }
)
"""Columns the ERA5 and CAMS pretraining rows hold; the other columns are 0 after standardisation."""


@dataclass(frozen=True)
class Config:
    """One network design: every switch of the shared skeleton that the solar ablations vary."""

    member_route: bool = False
    member_wise: bool = False
    """Train on one randomly drawn member per row and pass each member alone at inference."""
    drop_spread: bool = False
    clear_sky_input: bool = False
    plant: str = "none"
    """`none`, `frozen` (scale fitted, tilt fixed), or `e2e` (tilt, azimuth, scale, clip learned)."""
    atoms: bool = False
    pretrain: str = "none"
    """`none`, `finetune` (ERA5 and CAMS, then ensemble), or `only` (ERA5 and CAMS, no fine-tune)."""

    @property
    def needs_members(self) -> bool:
        return self.member_route or self.member_wise or self.plant != "none"

    @property
    def writes_paths(self) -> bool:
        return self.member_wise or self.member_route or self.pretrain == "none" and not (
            self.clear_sky_input or self.plant != "none" or self.atoms
        )

    @property
    def fixed_stats(self) -> bool:
        return self.pretrain != "none"

    @property
    def columns(self) -> tuple[str, ...]:
        cols = [c for c in GATE_COLUMNS if not (self.drop_spread and c == "ghi_std")]
        if self.clear_sky_input:
            cols.append("clear_sky")
        return tuple(cols)


ARMS: Final[dict[str, Config]] = {
    # Gate and A2: member handling.
    "g0_onepass": Config(),
    "a2_mean_only": Config(drop_spread=True),
    "a2_member": Config(member_route=True),
    "a2_memberwise": Config(member_wise=True),
    # A6: pretraining. Written for one-pass; `pre_*` arms with the member route are added by name
    # `m_pre_*` after the A2 decision.
    "a6_ens_only": Config(pretrain="ens_only_fixed"),
    "a6_finetune": Config(pretrain="finetune"),
    "a6_pretrain_only": Config(pretrain="only"),
    "a6m_ens_only": Config(member_route=True, pretrain="ens_only_fixed"),
    "a6m_finetune": Config(member_route=True, pretrain="finetune"),
    "a6m_pretrain_only": Config(member_route=True, pretrain="only"),
    # A5: physics layer, on the A2 and A6 winners (set when the stage runs).
    "a5_plain": Config(clear_sky_input=True),
    "a5_frozen": Config(clear_sky_input=True, plant="frozen"),
    "a5_e2e": Config(clear_sky_input=True, plant="e2e"),
    "a5m_plain": Config(clear_sky_input=True, member_route=True),
    "a5m_frozen": Config(clear_sky_input=True, plant="frozen", member_route=True),
    "a5m_e2e": Config(clear_sky_input=True, plant="e2e", member_route=True),
    # A7: bounded head.
    "a7_clipped": Config(),
    "a7_atoms": Config(atoms=True),
}
