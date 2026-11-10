# ruff: noqa: E501
"""Part B of experiment 531: score the stored L-1d weights (511, seeds 0 and 1) with a chosen weather feed. Inference only."""

import argparse
import time

import e00428_data as data  # ty: ignore[unresolved-import]
import e00510_common as c  # ty: ignore[unresolved-import]
import e00511_common as k  # ty: ignore[unresolved-import]
import e00511_fit as f511  # ty: ignore[unresolved-import]
import e00531_common as m  # ty: ignore[unresolved-import]
import polars as pl
import torch  # ty: ignore[unresolved-import]


def run(*, mode: str, months: list[str], device: str) -> None:
    """Write each month's per-seed forecast frames for the mode ("origin" check or "fresh")."""
    clean = c.clean_stack()
    dense = m.dense_by_primary() if mode == "fresh" else {}
    arm = f511.ARMS["L-1d"]
    for month in months:
        started = time.monotonic()
        rows = data.build_fold(refit=c.month_start(month=month), clean=clean)
        raw = m.raw_test(month=month, mode=mode, dense=dense)
        test = data.TestRows(
            x=rows.standardiser.apply(x=raw["x"]), primary=raw["primary"], origin_idx=raw["origin_idx"],
            target_idx=raw["target_idx"], lead=raw["lead"], origin=raw["origin"], valid=raw["valid"],
        )  # fmt: skip
        cuts = k.cutoffs(month=month, cadence="d")
        for seed in (0, 1):
            out = m.WORK / "seed_preds" / f"L1d-{mode}" / f"{month}_s{seed}.parquet"
            if out.exists():
                continue
            parts = []
            for i, cut in enumerate(cuts):
                day = f511.day_of(month=month, cut=cut)
                net = c.new_model(
                    outlier=False, n_inputs=test.x.shape[1], n_primaries=len(rows.target_mean)
                ).to(device)
                net.load_state_dict(
                    torch.load(
                        f511.model_file(arm=arm, month=month, day=day, seed=seed),
                        map_location=device,
                    )
                )
                net.eval()
                lo, hi = k.window_of(month=month, cadence="d", index=i)
                sel = (test.origin >= lo) & (test.origin < hi)
                if sel.any():
                    parts.append(
                        c.forecast_frame(
                            model=net,
                            outlier=False,
                            test=test,
                            select=sel,
                            rows=rows,
                            widen=1.0,
                            device=device,
                        )
                    )
            out.parent.mkdir(parents=True, exist_ok=True)
            pl.concat(parts).write_parquet(out)
            print(
                f"{mode}/{month}_s{seed}: {sum(p.height for p in parts):,} rows, dropped {int(raw['dropped'][0]):,}, {time.monotonic() - started:.0f} s",
                flush=True,
            )
        del rows, raw, test


def main() -> None:
    """Run part B."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("origin", "fresh"), required=True)
    p.add_argument("--months", nargs="+", default=list(m.MONTHS))
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    run(mode=a.mode, months=a.months, device=a.device)


if __name__ == "__main__":
    main()
