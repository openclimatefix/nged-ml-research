"""Stage 0 of experiment 479: signal share and minimum detectable capacity, before any fit is read.

For every Primary, prints the share of the PV shape's variance that survives the removal of the
month-year-hour-weekday means, the variance inflation of the capacity column against the
temperature columns, the ordinary-least-squares standard error of the read-out season-year's
capacity, and the minimum detectable capacity (80% power at the Bonferroni-corrected level).
It reads no injected series and no placebo.

    PYTHONPATH=<folders of 269, 129, 474> uv run --no-sync python e00479_signal.py
"""

import json

import e00269_common as c269  # ty: ignore[unresolved-import]
import e00479_anomaly as anomaly
import e00479_common as common
from scipy.stats import norm


def main() -> None:
    """Write `work/stage0_signal.json`."""
    cache = c269.load_cache()
    z_alpha = norm.ppf(1 - common.ALPHA / (2 * common.N_TESTS))
    z_power = norm.ppf(common.POWER)
    out: dict[str, dict] = {}
    for site in common.ALL_SITES:
        prepared = anomaly.prepare(cache=cache, site=site)
        for cloud in (False, True):
            design = anomaly.build_design(
                prepared=prepared, mask=prepared.usable, cloud_terms=cloud
            )
            gram, rhs = anomaly.monthly_normal_equations(design=design)
            col = anomaly.readout_column(design=design, season_year=common.LATEST_SEASON_YEAR)
            if col < 0:
                col = len(design.years) - 1
            se = anomaly.analytic_standard_error(design=design, gram=gram, rhs=rhs)
            vif = anomaly.variance_inflation(design=design, gram=gram)
            out[f"{site}|cloud={cloud}"] = {
                "n_rows": len(design.rows),
                "signal_share": anomaly.signal_share(prepared=prepared, mask=prepared.usable),
                "season_year_used": int(design.years[col]),
                "se_mw": float(se[col]),
                "vif": float(vif[col]),
                "mdc_mw": float((z_alpha + z_power) * se[col]),
                "eff_mw": prepared.eff_mw,
            }
            print(site, cloud, out[f"{site}|cloud={cloud}"], flush=True)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "stage0_signal.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
