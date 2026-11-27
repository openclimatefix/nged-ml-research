# ruff: noqa: E501
"""Experiment 191: in-sample gain of the price columns, and the fitted price coefficients in points.

For each monthly fit, the mean absolute error on the fit's own training rows of the model without
price columns (a0) and with them (joint, at the true price). The price coefficients are the
fitted output in points for each indicator; the table prints them in points and never in MW.

    uv run --no-sync python ml-experiments/00191-*/e00191_insample.py
"""

import pickle

import e00191_common as common
import e00191_engine as engine
import numpy as np
from studies.dp_gam import TrainingWindow, predict_gam, prepare_fit


def main() -> None:
    """Print the in-sample table for D8, the register's Primary, and D5."""
    cache = common.load_cache()
    for site in ("D8", common.REGISTER_PRIMARY, "D5"):
        work = engine.masked_cache(cache=cache, site=site)
        grid = engine.build_price_grid(cache=work, site=site, ungated=False)
        fits = pickle.loads((common.FITS_DIR / "gated" / f"{site}.pkl").read_bytes())
        print(f"\n== {common.shown(site)} (effective capacity [withheld])")
        print("cut        rows  a0 MAE  joint MAE  gain%   theta (points)")
        gains = []
        for cut_fits in fits:
            _, index = prepare_fit(
                cache=work, site=site, spec=engine.gam_spec(), cut=cut_fits.cut,
                window=TrainingWindow(months=common.WINDOW_MONTHS), season_years=common.SEASON_YEARS,
            )  # fmt: skip
            y = work.sites[site].y[index]
            a0 = np.abs(y - predict_gam(cache=work, fit=cut_fits.a0, index=index)).mean()
            joint_pred = predict_gam(
                cache=work, fit=cut_fits.joint.fit, index=index
            ) - engine.price_output(
                cut_fits=cut_fits, mode="true", cache=work, grid=grid, rows=index, origin=None
            )
            joint = np.abs(y - joint_pred).mean()
            gains.append((a0 - joint) / a0 * 100.0)
            theta = cut_fits.joint.theta_price
            # NGED withholds the fitted response at D8: it is the size of a gas engine's block.
            # Only the coefficients in points are printed, because with the response in MW as
            # well the ratio of the two would give the site's effective capacity.
            response = "[withheld]" if site == "D8" else f"{np.round(theta, 2)}"
            print(
                f"{cut_fits.cut:%Y-%m-%d} {len(index):6d} {a0:7.3f} {joint:9.3f} "
                f"{gains[-1]:6.2f}  {response}"
            )
        print(f"mean in-sample gain {np.mean(gains):.2f}%, last cut {gains[-1]:.2f}%")


if __name__ == "__main__":
    main()
