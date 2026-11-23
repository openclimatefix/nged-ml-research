# ruff: noqa: E501, PT018
"""Experiment 720, rung 0b: known-answer checks of the sign read-outs on synthetic worlds (CPU).

Run: `python e00720_check_sign.py`. The worlds are synthetic, so this reads no real data.
"""

import sys

import e00720_fold as fold
import e00720_sign as sign
import e00720_worlds as worlds
import numpy as np

N_DAYS = 900
SCORED = np.arange(540, 600)  # days of year 2 around midsummer
REFERENCE = np.arange(0, 540)
Q = 2.0


def run_world(
    *, seed: int, pv_fraction: float, estimate_scale: float = 1.0, sunny_share: float = 0.85
) -> tuple[worlds.SyntheticWorld, np.ndarray, np.ndarray, np.ndarray]:
    """Return the world and `ρ` of the anchored read-out with the real and the shifted sky."""
    w = worlds.synthetic_world(
        n_days=N_DAYS, capacity=100.0, seed=seed, pv_fraction=pv_fraction,
        estimate_scale=estimate_scale, sunny_share_summer=sunny_share,
    )  # fmt: skip
    v = fold.reading_to_magnitude(m=w.reading, q=Q)
    real, _ = sign.anchored_posterior(
        v=v, p=w.p, u=w.u, day_type=w.day_type, temperature=w.temperature,
        reference_days=REFERENCE, scored_days=SCORED,
    )  # fmt: skip
    shifted_p = worlds.shift_days(a=w.p, days=28)
    shifted, _ = sign.anchored_posterior(
        v=v, p=np.nan_to_num(shifted_p), u=w.u, day_type=w.day_type, temperature=w.temperature,
        reference_days=REFERENCE, scored_days=SCORED,
    )  # fmt: skip
    return w, v, real, shifted


def check_injected_reading() -> None:
    """A crossing half-hour reads above its mean magnitude, and a flat run reads sqrt(x²+Q²)."""
    x = np.array([[-20.0, -10.0, 0.0, 10.0, 20.0] + [20.0] * 43])
    r = worlds.reading_from_signed(x=x, q=Q)
    inst = worlds.reading_from_mean(x=x, q=Q)
    assert r[0, 2] > inst[0, 2] + 1.0, (r[0, 2], inst[0, 2])
    assert abs(r[0, 20] - inst[0, 20]) < 1e-9
    assert np.isnan(worlds.reading_from_signed(x=np.array([[np.nan, 1.0]]), q=Q)[0, 0])
    print(
        f"PASS injected reading: crossing half-hour reads {r[0, 2]:.2f} against {inst[0, 2]:.2f} for the half-hour mean"
    )


def check_metric_functions() -> None:
    """The area and recall functions agree with hand-built cases."""
    s = np.array([0.1, 0.2, 0.8, 0.9])
    pos = np.array([False, False, True, True])
    assert sign.auc(score=s, positive=pos) == 1.0
    assert sign.auc(score=s[::-1], positive=pos) == 0.0
    assert sign.auc(score=np.ones(4), positive=pos) == 0.5
    assert sign.recall_at_false_rate(score=s, positive=pos) == 1.0
    print("PASS metrics: perfect, reversed, and tied scores give 1, 0, and 0.5")


def check_anchored_exporter() -> None:
    """Anchored read-out on an exporting world: real sky well above shifted sky; stop rule holds."""
    w, _v, real, shifted = run_world(seed=11, pv_fraction=1.0)
    truth = w.x[SCORED]
    u = w.u[SCORED]
    a = sign.evaluate_sign(rho=real, x_true=truth, u=u)
    b = sign.evaluate_sign(rho=shifted, x_true=truth, u=u)
    assert a["n_reverse"] > 100, a
    assert a["auc"] > 0.95, a
    assert a["auc"] - b["auc"] >= 0.1, (a, b)
    print(
        f"PASS anchored exporter: area {a['auc']:.3f} (real sky) vs {b['auc']:.3f} (shifted); recall at 5% false {a['recall_at_5pct_false']:.3f}; {int(a['n_reverse'])} reverse rows"
    )


def check_no_injection_world() -> None:
    """A non-exporting world: false reverse flow stays under 1% of sunny middays with the gate open."""
    w, _v, real, _ = run_world(seed=12, pv_fraction=0.12)
    sunny_mid = (w.u[SCORED] >= 0.5) & (np.abs(np.arange(48) - 24)[None, :] <= 4)
    rate, n = sign.false_reverse_rate(rho=real, mask=sunny_mid)
    assert n > 100 and rate < 0.01, (rate, n)
    assert (w.x[SCORED] >= 0).all()
    print(f"PASS non-exporting world: false reverse flow {rate:.4f} on {n} sunny midday half-hours")


def check_unanchored_fails_when_export_heavy() -> None:
    """The unanchored control loses sign skill in an export-heavy window, the anchored one does not."""
    w, v, real, _ = run_world(seed=13, pv_fraction=1.0, sunny_share=0.95)
    passes = sign.unanchored_posterior(v=v, p=w.p, u=w.u, day_type=w.day_type, weights="sky")
    truth, u = w.x, w.u
    anch = sign.evaluate_sign(rho=real, x_true=truth[SCORED], u=u[SCORED])
    sel = np.zeros_like(truth, dtype=bool)
    sel[SCORED] = True
    res = []
    for k, rho_k in enumerate(passes, start=1):
        r = np.where(sel, rho_k, np.nan)
        res.append(sign.evaluate_sign(rho=r, x_true=truth, u=u))
        print(
            f"  unanchored pass {k}: area {res[-1]['auc']:.3f}, recall {res[-1]['recall_at_5pct_false']:.3f}"
        )
    assert anch["auc"] > 0.9, anch
    assert res[-1]["auc"] < anch["auc"] - 0.05, (res[-1], anch)
    print(
        f"PASS export-heavy window: anchored area {anch['auc']:.3f} beats unanchored {res[-1]['auc']:.3f} after three passes"
    )


def check_capacity_sensitivity() -> None:
    """The estimate scale moves the sign: report the area at 0.7, 1.0 and 1.3 times the capacity."""
    out = {}
    for scale in (0.7, 1.0, 1.3):
        w, _v, real, _ = run_world(seed=14, pv_fraction=1.0, estimate_scale=scale)
        out[scale] = sign.evaluate_sign(rho=real, x_true=w.x[SCORED], u=w.u[SCORED])["auc"]
    assert out[1.0] >= max(out[0.7], out[1.3]) - 1e-9, out
    print(
        "PASS capacity sensitivity (reported, no bar): "
        + ", ".join(f"{k}x -> {v:.3f}" for k, v in out.items())
    )


def check_leak_guard() -> None:
    """Scored days that overlap the reference days are refused."""
    w = worlds.synthetic_world(n_days=100, capacity=100.0, seed=1)
    v = fold.reading_to_magnitude(m=w.reading, q=Q)
    try:
        sign.anchored_posterior(
            v=v, p=w.p, u=w.u, day_type=w.day_type, temperature=w.temperature,
            reference_days=np.arange(0, 60), scored_days=np.arange(50, 70),
        )  # fmt: skip
    except AssertionError:
        print("PASS leak guard: overlapping reference and scored days are refused")
        return
    msg = "the leak guard did not fire"
    raise AssertionError(msg)


def main() -> int:
    """Run every check; return 0 when all pass."""
    for check in (
        check_injected_reading,
        check_metric_functions,
        check_anchored_exporter,
        check_no_injection_world,
        check_unanchored_fails_when_export_heavy,
        check_capacity_sensitivity,
        check_leak_guard,
    ):
        check()
    print("rung 0b (synthetic): all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
