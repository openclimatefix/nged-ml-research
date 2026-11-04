# ruff: noqa: E501
"""Check that Part 3's identity arm reproduces experiment 448's stored fine-tuned L0PT forecasts.

Loads the fine-tuned L0PT network of fold 2026-01, seed 0, predicts the 2026-01 test rows with its own
embedding, and compares each quantile with `predictions/L0PT/2026-01_s0_w0.parquet` of experiment 448
(raw quantiles, fraction of capacity), row by row on (Primary, origin, valid time). Writes
`work/identity_check.json`.
"""

import json

import e00397_prep as prep  # ty: ignore[unresolved-import]
import e00428_data as data428  # ty: ignore[unresolved-import]
import e00510_common as c510  # ty: ignore[unresolved-import]
import e00522_common as common  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

LEVELS = ("q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99")


def main() -> None:
    """Compare the identity quantiles with the stored ones."""
    torch.set_num_threads(1)
    folder = common.E448 / "work" / "models" / "L0PT" / "2026-01" / "s0"
    z = np.load(folder / "standardiser.npz")
    test = data428.build_test(
        month="2026-01",
        standardiser=prep.Standardiser(mean=z["mean"], std=z["std"], mask=z["mask"]),
    )
    model = c510.new_model(outlier=False, n_inputs=test.x.shape[1], n_primaries=common.N_PRIMARIES)
    c510.load_pretrained(model=model, path=folder / "model.pt")
    model.eval()
    out = []
    for a in range(0, len(test.x), 65_536):
        sl = slice(a, a + 65_536)
        q = c510.serving_quantiles(
            model=model,
            x=torch.from_numpy(test.x[sl]),
            p=torch.from_numpy(test.primary[sl]),
            outlier=False,
        ).numpy()
        q = np.sort(
            q * z["target_std"][test.primary[sl]][:, None]
            + z["target_mean"][test.primary[sl]][:, None],
            axis=1,
        )
        out.append(q)
    mine = pl.DataFrame(np.concatenate(out), schema=list(LEVELS), orient="row").with_columns(
        primary=pl.Series(test.primary),
        origin_us=pl.Series(test.origin),
        valid_us=pl.Series(test.valid),
    )
    stored = pl.read_parquet(next(common.E448.glob("work/predictions/L0PT/2026-01_s0_w0.parquet")))
    joined = mine.join(
        stored.select("primary", "origin_us", "valid_us", *LEVELS),
        on=["primary", "origin_us", "valid_us"],
        suffix="_stored",
        how="inner",
    )
    diffs = {q: float((joined[q] - joined[f"{q}_stored"]).abs().max()) for q in LEVELS}  # ty: ignore[invalid-argument-type]
    report = {
        "rows_mine": mine.height,
        "rows_stored": stored.height,
        "rows_joined": joined.height,
        "max_abs_difference_by_level": diffs,
    }
    (common.WORK_DIR / "identity_check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
