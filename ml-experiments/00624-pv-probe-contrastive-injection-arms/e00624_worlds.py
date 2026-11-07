# ruff: noqa: E501, ANN001
"""Experiment 624, tests 2 and 3: a clipped physics-curve world and a displaced-sky world (plain network).

Test 2 injects the physics unit curve of the Primary's own CAMS shortwave, clipped as an inverter does
at DC-to-AC ratios 1.0, 1.2, and 1.4: the shape is `min(ratio * u, 1)`, divided by its 99th percentile
over daylight rows (sun under 75 degrees of zenith) so that the injected series reaches the injected
megawatts at its 99th percentile, as the farm series do. Test 3 injects the unit curve driven by the CAMS
shortwave of the nth nearest other Primary (n = 1 and 2) instead of the Primary's own; the CAMS
downloads are point series per site, so a one-pixel or two-pixel displacement is not available
locally, and the distance between Primaries stands in for it.

The network is experiment 621's plain network (experiment 554's stage `pre` network); the rows, loop,
seeds, and probe are experiment 623's.

Usage: python e00624_worlds.py fit --cutoff 2026-06
       python e00624_worlds.py probe --cutoff 2026-06
"""

import argparse
import json
import re
import time

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00554_model as m  # ty: ignore[unresolved-import]
import e00623_common as k
import e00623_fit as f623  # ty: ignore[unresolved-import]
import e00623_probe as p623  # ty: ignore[unresolved-import]
import e00624_common as k4
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

WORLDS: tuple[str, ...] = (
    "physclip_dc1.0_mw4",
    "physclip_dc1.2_mw4",
    "physclip_dc1.4_mw4",
    "physdisp_n1_mw4",
    "physdisp_n2_mw4",
)
MIN_COS_ZENITH_FARM = float(np.cos(np.radians(75.0)))
WORK = k4.WORK


class WorldInjector(k.Injector):
    """The experiment 623 injector plus the clipped and displaced worlds."""

    def __init__(self, *, fold: k4.Fold) -> None:
        """Hold the fold's rows for the neighbour lookup."""
        pre = fold.pre
        super().__init__(
            primary=pre.primary,
            target_idx=pre.target_idx,
            x=pre.x,
            standardiser=fold.rows.standardiser,
        )
        self.fold = fold
        sites = prep.primaries()
        lat, lon = (
            np.radians(sites["latitude"].to_numpy()),
            np.radians(sites["longitude"].to_numpy()),
        )
        dlat, dlon = (
            lat[:, None] - lat[None, :],
            (lon[:, None] - lon[None, :]) * np.cos(lat[:, None]),
        )
        self.distance_km = 6371.0 * np.hypot(dlat, dlon)
        np.fill_diagonal(self.distance_km, np.inf)

    def neighbour(self, *, primary: int, n: int) -> int:
        """Return the nth nearest other Primary (n from 1)."""
        return int(np.argsort(self.distance_km[primary])[n - 1])

    def sky_of(self, *, other: int, rows_t: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        """Return the shortwave of Primary `other` at the given target indices, with the fallback where absent."""
        sel = self.primary == other
        t, ghi = self.target_idx[sel], self.ghi[sel]
        order = np.argsort(t, kind="stable")
        t, ghi = t[order], ghi[order]
        pos = np.clip(np.searchsorted(t, rows_t), 0, len(t) - 1)
        found = t[pos] == rows_t
        return np.where(found, ghi[pos], fallback)

    def change_points(self, *, condition) -> np.ndarray:
        """Return the target change in points for a clipped or displaced world, else defer to experiment 623."""
        match = re.fullmatch(
            r"physclip_dc(?P<r>[\d.]+)_mw(?P<mw>\d+)|physdisp_n(?P<n>\d)_mw(?P<mw2>\d+)",
            condition.name,
        )
        if match is None:
            return super().change_points(condition=condition)
        change = np.zeros(len(self.primary), dtype=np.float64)
        for site in k.SIX:
            p = self.six[site]
            sel = self.primary == p
            inc = 1.0 / self.cap[p]
            if match.group("r"):
                u = k.unit_curve(
                    ghi=self.ghi[sel], x=self.x[sel], times_us=self.t_us[sel], temp=self.temp[sel]
                )
                s = np.minimum(float(match.group("r")) * u, 1.0)
                day = self.x[sel][:, 14] > MIN_COS_ZENITH_FARM
                shape = s / float(np.percentile(s[day], 99))
                mw = float(match.group("mw"))
            else:
                other = self.neighbour(primary=p, n=int(match.group("n")))
                ghi = self.sky_of(other=other, rows_t=self.target_idx[sel], fallback=self.ghi[sel])
                shape = k.unit_curve(
                    ghi=ghi, x=self.x[sel], times_us=self.t_us[sel], temp=self.temp[sel]
                )
                mw = float(match.group("mw2"))
            change[sel] = -mw * inc * shape
        return change


def parse(name: str):  # noqa: ANN201
    """Return a condition-like object carrying only the name (the injector reads the name)."""
    from types import SimpleNamespace

    return SimpleNamespace(name=name, kind="world", mw=4.0)


def weights(*, cutoff: str, world: str, seed: int):  # noqa: ANN201
    """Return where a worlds network is stored."""
    return WORK / "pretrained_worlds" / cutoff / f"{world}_s{seed}.pt"


def fit(*, cutoff: str, worlds: list[str], seeds: list[int]) -> None:
    """Pretrain the plain network in each requested world."""
    device = "cuda"
    fold = k4.Fold(cutoff=cutoff)
    inj = WorldInjector(fold=fold)
    pre, keep = fold.pre, fold.keep
    tgt_mean, tgt_std = fold.rows.target_mean[pre.primary], fold.rows.target_std[pre.primary]
    x = torch.from_numpy(pre.x[keep]).to(device)
    p = torch.from_numpy(pre.primary[keep]).to(device)
    v = torch.from_numpy(pre.is_val[keep]).to(device)
    w = torch.ones(int(keep.sum()), device=device)
    for name in worlds:
        change = inj.change_points(condition=parse(name))
        z = torch.from_numpy(
            (((pre.y.astype(np.float64) + change) - tgt_mean) / tgt_std)[keep].astype(np.float32)
        ).to(device)
        for seed in seeds:
            target = weights(cutoff=cutoff, world=name, seed=seed)
            if target.exists():
                continue
            started = time.monotonic()
            torch.manual_seed(seed)
            model = m.ModularMdn(n_primaries=len(fold.rows.target_mean), stage="pre").to(device)
            epochs, best = f623.train(
                model=model,
                x=x,
                z=z,
                w=w,
                p=p,
                is_val=v,
                seed=seed,
                lr=c.PRETRAIN_LR,
                device=device,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            torch.save(model.state_dict(), tmp)
            tmp.rename(target)
            print(
                f"  {cutoff} {name} s{seed}: {epochs} epochs, val NLL {best:.4f}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
    meta = {}
    for site, i in fold.six.items():
        meta[site] = {
            f"n{n}": {
                "other": inj.neighbour(primary=i, n=n),
                "distance_km": round(float(inj.distance_km[i, inj.neighbour(primary=i, n=n)]), 1),
            }
            for n in (1, 2)
        }
    (WORK / f"worlds_neighbours_{cutoff}.json").write_text(json.dumps(meta, indent=1))


def probe(*, cutoff: str, worlds: list[str], seeds: list[int]) -> None:
    """Probe each worlds network at the real Primaries' vectors."""
    torch.set_num_threads(2)
    frame = p623.load_frame(cutoff=cutoff)
    six = k.six_indices()
    cap = k.effective_capacity_mw()
    probes = {site: p623.cells_and_donors(frame=frame, primary=i) for site, i in six.items()}
    for name in worlds:
        out = WORK / "probes_worlds" / cutoff / f"{name}.parquet"
        if out.exists():
            continue
        records = []
        for seed in seeds:
            net = m.ModularMdn(n_primaries=16, stage="pre")
            net.load_state_dict(
                torch.load(weights(cutoff=cutoff, world=name, seed=seed), map_location="cpu")
            )
            net.eval()
            for site, i in six.items():
                rng = np.random.default_rng(1000 * seed + i)
                res = p623.probe_network(
                    model=net,
                    frame=frame,
                    primary=i,
                    probe=probes[site],
                    cap=float(cap[i]),
                    rng=rng,
                )
                for kind in ("plain", "cloud_tpl"):
                    point, boot = res[kind]
                    records.append(
                        {
                            "cutoff": cutoff,
                            "condition": name,
                            "seed": seed,
                            "site": site,
                            "model": kind,
                            "capacity_mw": point,
                            "boot": boot.tolist(),
                        }
                    )
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(records).write_parquet(out)
        print("wrote", out, flush=True)


def main() -> None:
    """Run the `fit` or `probe` step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["fit", "probe"])
    parser.add_argument("--cutoff", required=True, choices=k.CUTOFFS)
    parser.add_argument("--worlds", nargs="+", default=list(WORLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(k.SEEDS))
    args = parser.parse_args()
    (fit if args.step == "fit" else probe)(cutoff=args.cutoff, worlds=args.worlds, seeds=args.seeds)


if __name__ == "__main__":
    main()
