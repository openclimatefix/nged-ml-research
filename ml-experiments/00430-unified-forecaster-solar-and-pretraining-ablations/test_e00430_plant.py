# ruff: noqa: D103, E402
"""Check experiment 430's PyTorch plant against the NumPy and pvlib originals it copies."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path[:0] = [
    str(Path(__file__).parent),
    str(Path(__file__).parents[1] / "00310-metered-farm-long-lead-scoring"),
]

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_members as m310  # ty: ignore[unresolved-import]
import e00430_plant as plant
import numpy as np
import pvlib
import torch  # ty: ignore[unresolved-import]


def _inputs(n: int = 4000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    zenith = rng.uniform(20.0, 88.0, n)
    return {
        "ghi": rng.uniform(0.0, 900.0, n) * np.cos(np.radians(zenith)) ** 0.5,
        "temperature_c": rng.uniform(-5.0, 30.0, n),
        "zenith": zenith,
        "azimuth": rng.uniform(60.0, 300.0, n),
        "doy": rng.integers(1, 366, n).astype(float),
    }


def test_erbs_matches_pvlib() -> None:
    d = _inputs()
    split = pvlib.irradiance.erbs(ghi=d["ghi"], zenith=d["zenith"], datetime_or_doy=d["doy"])
    dni, dhi = plant.erbs(
        ghi=torch.tensor(d["ghi"]), zenith_deg=torch.tensor(d["zenith"]), doy=torch.tensor(d["doy"])
    )
    np.testing.assert_allclose(
        dhi.numpy(), np.nan_to_num(np.asarray(split["dhi"])), rtol=1e-4, atol=1e-3
    )
    np.testing.assert_allclose(
        dni.numpy(), np.nan_to_num(np.asarray(split["dni"])), rtol=1e-4, atol=1e-2
    )


def test_unit_response_matches_experiment_310_at_its_tilt() -> None:
    d = _inputs()
    reference, _, _ = m310.unit_response(
        ghi=d["ghi"][:, None],
        temperature_c=d["temperature_c"][:, None],
        zenith=d["zenith"],
        azimuth=d["azimuth"],
        doy=d["doy"],
    )
    ours = plant.unit_response(
        ghi=torch.tensor(d["ghi"]),
        temperature_c=torch.tensor(d["temperature_c"]),
        zenith_deg=torch.tensor(d["zenith"]),
        azimuth_deg=torch.tensor(d["azimuth"]),
        doy=torch.tensor(d["doy"]),
        tilt_deg=c310.FLEET_TILT_DEG,
        panel_azimuth_deg=c310.FLEET_AZIMUTH_DEG,
    )
    np.testing.assert_allclose(ours.numpy(), reference[:, 0], rtol=1e-4, atol=1e-4)


def test_end_to_end_plant_has_gradients_for_all_four_parameters() -> None:
    d = _inputs(256)
    module = plant.Plant(tilt_deg=20.0, azimuth_deg=180.0, scale=0.9, clip=0.8, learn=True)
    out = module(
        ghi=torch.tensor(d["ghi"]),
        temperature_c=torch.tensor(d["temperature_c"]),
        zenith_deg=torch.tensor(d["zenith"]),
        azimuth_deg=torch.tensor(d["azimuth"]),
        doy=torch.tensor(d["doy"]),
    )
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad) for p in module.parameters())
