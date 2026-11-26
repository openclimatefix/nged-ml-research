# ruff: noqa: E402, E501, PLR0915
"""Experiment 732: the absolute scores of every arm, for the write-up (new, read only, CPU).

Reads the stored predictions and rebuilds the test rows exactly as `e00732_report.stage_figures`
does, then writes `scores/tables.json`: the absolute Brier score, log loss, and harm of the sign by
arm, the solar error, the eight solar bars with their values, the step scores, tables T1 to T3 and
T5 to T6, and the BSP rows. It changes no stored file other than the new `scores/tables.json` and
no bar, margin, or constant. Run it as

    taskset -c 22-27 python e00732_tables.py

with `DATA_PATH_INTERNAL` set. Aggregates only.
"""

import json
import sys
from collections.abc import Mapping

import e00732_paths as paths

paths.pin_threads()

import e00730_score as score730
import e00732_cli as cli
import e00732_core as core
import e00732_report as report
import e00732_score as score
import numpy as np

ARMS_SIGN = ("S", "S+", "U*", "U-NC", "U-NCw", "U-SG", "U-N", "U-noBSP", "S-L", "U*-L")
ARMS_SOLAR = ("U*", "U-NC", "U-NCw", "U-SG", "U-N", "U-noBSP", "U*-L", "B-PV0", "B-PV1", "B-PV2")
ARMS_STEP = ("U*", "U-NC", "U-NCw", "U-SG", "U-N", "U-noBSP", "U*-L", "B-TN", "B-T")


def present(*, by_name: Mapping[str, np.ndarray | None]) -> dict[str, np.ndarray]:
    """Return the arrays of `by_name` that exist, dropping the arms whose array is `None`."""
    return {name: array for name, array in by_name.items() if array is not None}


def required(*, by_name: Mapping[str, np.ndarray | None]) -> dict[str, np.ndarray]:
    """Return `by_name` with every array present.

    Args:
        by_name: Each arm's array, or `None` where the arm did not produce one.

    Returns:
        The same arrays, now known not to be `None`.

    Raises:
        ValueError: If any arm's array is `None`.
    """
    missing = [name for name, array in by_name.items() if array is None]
    if missing:
        raise ValueError(f"arms without the required array: {missing}")
    return present(by_name=by_name)


def clean(value: object) -> object:
    """Make a nested value JSON-safe."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    """Build the tables and write them."""
    root, root730 = paths.data_root(), paths.root_730()
    ctx = cli.real_context()
    base = cli.real_base(ctx=ctx)
    env = cli.real_env(root=root, ctx=ctx, base=base, with_real=False)
    limits_path = root / "limits.json"
    noise = float(json.loads(limits_path.read_text())["noise_units"]["D8"])
    truth_, overlay, valid = report.build_truth(root730=root730, env=env, noise_d8=noise)
    arm_map = report.assemble_arms(
        root=root, root730=root730, env=env, base=base, truth_=truth_, ctx=ctx,
        test_overlay=overlay, test_valid=valid,
    )  # fmt: skip
    limits = core.read_limits_file(path=limits_path)
    s_share = float(json.loads(limits_path.read_text())["s"])
    rows = truth_.rows
    x, r = rows.x, rows.r
    dec = r >= core.DECIDABLE_READING_POINTS
    band13 = dec & (r < core.BAND_EDGE_POINTS)
    band3 = dec & (r >= core.BAND_EDGE_POINTS)
    masks = {
        "all": np.ones(rows.n, bool), "import": x >= 0, "reverse": x < 0,
        "decidable_1_3mva": band13, "decidable_3mva_plus": band3,
    }  # fmt: skip
    bsp = np.asarray(rows.has_bsp, dtype=bool)
    out: dict[str, object] = {"n_rows": rows.n, "s_share": s_share}

    sign = {}
    for name in ARMS_SIGN:
        p = arm_map[name].p
        if p is None:
            continue
        rb, rl = score730.row_brier(p=p, x=x), score730.row_log_loss(p=p, x=x)
        sign[name] = {
            "brier": {k: float(rb[m].mean()) for k, m in masks.items()},
            "log_loss": {k: float(rl[m].mean()) for k, m in masks.items()},
            "brier_bsp_months_rows": float(rb[bsp].mean()) if bsp.any() else None,
            "h1": score.harm_by_band(rows=rows, p=p, limits=limits),
            "detection": score730.detection(rows=rows, p=p),
            "calibration": {
                k: v
                for k, v in score730.calibration(rows=rows, p=p, bar=0.1 * s_share).items()
                if k in ("reliability", "max_gap", "pass", "bar")
            },
        }
    out["sign"] = sign
    out["p0_brier"] = {k: float(((x < 0).astype(float) * 1.0)[m].mean()) for k, m in masks.items()}
    out["p0_brier_note"] = "P0 gives P(X<0)=0, so its Brier score is the share of reverse-flow rows"

    up = score.sun_up(cosz=truth_.cosz) & ~truth_.exclude
    solar_rows = score.solar_world(rows=rows)
    pv = present(by_name={n: arm_map[n].pv for n in ARMS_SOLAR})
    out["solar_mae"] = {
        n: {
            "mae_sun_up": float(np.abs(v - rows.solar)[up].mean()),
            "rmse_sun_up": float(np.sqrt(((v - rows.solar) ** 2)[up].mean())),
            "mae_sun_up_solar_worlds": float(np.abs(v - rows.solar)[up & solar_rows].mean()),
        }
        for n, v in pv.items()
    }
    out["solar_mean_target_sun_up"] = float(rows.solar[up].mean())
    out["solar_mean_target_solar_worlds"] = float(rows.solar[up & solar_rows].mean())
    out["zero_output_mae_note"] = "a head that outputs nothing has MAE equal to the mean target"
    out["t1_solar_by_world_source"] = score.solar_error_table(
        rows=rows, cosz=truth_.cosz, pv_by_arm=pv
    )
    pv_tr = present(by_name={n: arm_map[n].pv_tr for n in ("U*", "U-N")})
    out["t1_transfer"] = score.transfer_solar_table(
        rows=rows, cosz=truth_.cosz, tr=truth_.tr, pv_tr_by_arm=pv_tr
    )

    keep = {
        a: score.injection_keep(delta=truth_.delta[a], amount=a, cosz=truth_.cosz, pair=truth_.pair[a].astype(float), exclude=truth_.exclude)
        for a in core.TEST_INJECTION_POINTS
    }  # fmt: skip
    bars = {}
    for n in ("U*", "U-NC", "U-NCw", "U-SG", "U*-L", "B-PV0", "B-PV1", "B-PV2"):
        a = arm_map[n]
        if a.pv is None:
            continue
        entry: dict[str, object] = {}
        if set(a.pv_inj) == set(core.TEST_INJECTION_POINTS):
            entry["recovery"] = {
                str(amt): score.recovery_bar(pv_inj=a.pv_inj[amt], pv_plain=a.pv, delta=truth_.delta[amt], amount=amt, keep=keep[amt])
                for amt in core.TEST_INJECTION_POINTS
            }  # fmt: skip
            entry["recovery_reverse"] = score.recovery_reverse_bar(
                pv_inj=a.pv_inj, pv_plain=a.pv, delta=truth_.delta, keep=keep, reverse=x < 0
            )
            entry["recovery_by_group"] = {
                str(amt): score.recovery_by_group(truth=truth_, pv_plain=a.pv, pv_inj=a.pv_inj, groups=score.sign_change_groups(truth=truth_, amount=amt))
                for amt in core.TEST_INJECTION_POINTS
            }  # fmt: skip
            entry["injection_median_abs_error_points"] = {
                str(amt): float(
                    np.median(np.abs(a.pv_inj[amt] - a.pv - truth_.delta[amt])[keep[amt]])
                )
                for amt in core.TEST_INJECTION_POINTS
            }
            entry["injection_mean_abs_error_points"] = {
                str(amt): float(
                    np.mean(np.abs(a.pv_inj[amt] - a.pv - truth_.delta[amt])[keep[amt]])
                )
                for amt in core.TEST_INJECTION_POINTS
            }
        entry["absolute"] = score.absolute_bar(
            pv=a.pv,
            solar=rows.solar,
            cosz=truth_.cosz,
            world_is_solar=solar_rows,
            exclude=truth_.exclude,
        )
        entry["night"] = score.night_bar(pv=a.pv, cosz=truth_.cosz, valid=np.ones(rows.n, bool))
        if a.pv_raised is not None:
            entry["monotone"] = score.monotone_bar(
                pv=a.pv, pv_raised=a.pv_raised, raisable=truth_.raisable
            )
            entry["response"] = score.response_bar(
                pv=a.pv, pv_raised=a.pv_raised, raisable=truth_.raisable, solar_target=rows.solar
            )
        bars[n] = entry
    out["solar_bars"] = bars

    tr = truth_.tr
    step = {}
    one = {
        "e": truth_.event[None, :],
        "s": truth_.size[None, :],
        "v": np.ones((1, truth_.event.size), bool),
    }
    for n in ARMS_STEP:
        a = arm_map[n]
        if a.step is None:
            continue
        f = score.step_f1(step_p=a.step[None, :], event=one["e"], size=one["s"], valid=one["v"])
        t6 = score.step_table(
            step_p=a.step, size_hat=a.size_hat, event=truth_.event, size=truth_.size, tr=tr
        )
        precision_all = score.precision_all(step_p=a.step, event=truth_.event)
        counts = {
            "below": int(((truth_.event > 0.5) & (tr.edge_noise < 4.0) & tr.edge).sum()),
            "above": int(((truth_.event > 0.5) & (tr.edge_noise >= 4.0) & tr.edge).sum()),
        }
        step[n] = {
            "f1_big": f,
            "brier_step": float(((a.step - truth_.event) ** 2).mean()),
            "size_error_points": None
            if a.size_hat is None
            else float(
                score.size_error(
                    size_hat=a.size_hat,
                    size=truth_.size,
                    event=truth_.event,
                    valid=np.ones_like(truth_.event, bool),
                )
            ),
            "precision_all": precision_all,
            "t6": t6,
            "detectability": score.detectability(
                table=t6["size_noise_units"], precision=precision_all, counts=counts
            ),
            "counts_by_4_noise_units": counts,
        }
    out["step"] = step
    r18_steps = present(by_name={n: arm_map[n].step for n in ("U*", "U-N", "B-TN", "B-T")})
    out["r18"] = score.recall_gap_r18(steps=r18_steps, event=truth_.event, size=truth_.size, tr=tr)
    out["n_edges_total"] = int(((truth_.event > 0.5) & tr.edge).sum())
    out["n_edges_5_points_plus"] = int(((truth_.event > 0.5) & (np.abs(truth_.size) >= 5.0)).sum())
    out["n_transfer_rows"] = int(truth_.event.size)

    out["t2_scenarios"] = score.scenario_table(
        truth=truth_,
        arms={
            n: arm_map[n]
            for n in ("U*", "S+", "S", "U-N")
            if arm_map[n].p_tr is not None or arm_map[n].step is not None
        },
    )
    p_by_arm = required(by_name={n: arm_map[n].p for n in ("U*", "S+", "U-SG", "S")})
    out["t3_ambiguity"] = score.ambiguity_table(rows=rows, p_by_arm=p_by_arm)
    # T5: U* with the BSP channel hidden
    a = arm_map["U*"]
    if a.pv is not None and a.p_nobsp is not None:
        out["u_star_p_nobsp_brier_bsp_rows"] = float(
            score730.row_brier(p=a.p_nobsp, x=x)[bsp].mean()
        )
        out["u_star_p_brier_bsp_rows"] = float(score730.row_brier(p=a.p, x=x)[bsp].mean())
    out["bsp_rows"] = int(bsp.sum())
    out["bsp_reverse_rows"] = int(((x < 0) & bsp).sum())
    # beta terciles are stored per segment in the overlay; not recomputed here
    # world mix and counts
    out["n_unreliable_excluded"] = int(truth_.exclude.sum())
    out["share_reverse_rows"] = float((x < 0).mean())
    out["share_import_below_1mva"] = float(((x >= 0) & ~dec).mean())
    p_reverse = required(by_name={n: arm_map[n].p for n in ("S", "S+", "U*")})
    out["mean_p_s_on_reverse"] = {n: float(p[x < 0].mean()) for n, p in p_reverse.items()}
    path = root / "scores" / "tables.json"
    path.write_text(json.dumps(clean(out), indent=1))
    print(f"wrote {path.name}: arms {sorted(sign)}")


if __name__ == "__main__":
    sys.exit(main())
