# ruff: noqa: E501, D103, ANN001, RUF059, PERF401
"""Experiment 687: figures.

Writes SVGs to docs/ml-experiments/00687-*/. Run
`npx svgo@4 --multipass --precision=1 --final-newline` on them after.
"""

import json
from pathlib import Path

import e00650_common as c650  # ty: ignore[unresolved-import]
import e00650_fit as f650  # ty: ignore[unresolved-import]
import e00652_common as c652  # ty: ignore[unresolved-import]
import e00687_fit as f687
import e00687_score as s
import e00687_tables as t
import matplotlib as mpl
import numpy as np
import polars as pl

mpl.use("svg")
import matplotlib.pyplot as plt

mpl.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
    }
)
DOCS = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00687-causal-detector-features-for-the-prevailing-forecast-heads"
)
COL = {
    "B0": "#306BFF",
    "L1": "#B701FF",
    "F1": "#FF4901",
    "F1d46": "#17E58F",
    "F2": "#10C5F7",
    "F1h4": "#FC9700",
    "P": "#888888",
    "B0g": "#009C75",
    "K1": "#222222",
}
LABEL = {
    "B0": "B0 (experiment 652's head)",
    "L1": "L1 (extra lags)",
    "F1": "F1 (detector features)",
    "F1d46": "F1d46 (D4 to D6 only)",
    "F2": "F2 (features plus flag)",
    "F1h4": "F1h4 (4 h cutoff)",
    "P": "P (another Primary's features)",
    "B0g": "B0g (new age rule only)",
}
W = 7.4


def boot_mean(v: np.ndarray, rng) -> tuple[float, float, float]:
    b = v[rng.integers(0, len(v), (4000, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def reaction_contrast(r: pl.DataFrame, arm: str, col: str, rng) -> tuple[float, float, float]:
    ref = r.filter(pl.col("arm") == "B0").sort("month", "seed", "pass", "prim")
    x = r.filter(pl.col("arm") == arm).sort("month", "seed", "pass", "prim")
    key = (ref["month"] + "_" + ref["pass"].cast(str) + "_" + ref["prim"].cast(str)).to_numpy()
    _u, inv = np.unique(key, return_inverse=True)
    ev = np.bincount(inv, weights=x[col].to_numpy() - ref[col].to_numpy()) / np.bincount(inv)
    return boot_mean(ev, rng)


def fig_reaction() -> None:
    r = pl.read_parquet(s.OUT / "reaction.parquet")
    rng = np.random.default_rng(0)
    arms = ["F1", "F1d46", "F2", "F1h4", "L1", "B0g", "P"]
    fig, axes = plt.subplots(2, 1, figsize=(W, 6.2), sharex=True)
    for ax, col, title in ((axes[0], "start_days", "After a shift starts: origin days until the prevailing output covers half the shift"), (axes[1], "end_days", "After a shift ends: origin days until the prevailing output has released half of it")):  # fmt: skip
        for i, a in enumerate(arms):
            m, lo, hi = reaction_contrast(r, a, col, rng)
            ax.plot([lo, hi], [i, i], color=COL[a], lw=1)
            ax.plot([m], [i], "o", color=COL[a], ms=5)
        ax.axvline(0, color="#888", lw=1)
        ax.set_yticks(range(len(arms)), [LABEL[a] for a in arms])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9, loc="left")
    axes[1].set_xlabel("Reaction time minus B0's, days (negative = reacts sooner)\nDot: mean over 281 injected events. Line: 95% interval over events.")  # fmt: skip
    fig.suptitle("Detector features make the prevailing output react about 1.6 days sooner at the start of a shift; extra lags get most of the way", fontsize=10, x=0.02, ha="left", wrap=True)  # fmt: skip
    fig.tight_layout()
    fig.savefig(DOCS / "fig1_reaction.svg")
    plt.close(fig)


def fig_auroc() -> None:
    d = json.loads((t.WORK / "separability.json").read_text())
    fig, axes = plt.subplots(2, 1, figsize=(W, 5.6), sharex=True)
    for ax, fam in zip(axes, ("train_family", "held_out"), strict=True):
        pos = 0
        ticks, labels = [], []
        for bank, bname in (("M", "stored"), ("S", "mirror x0.6"), ("X", "no mirror")):
            for lab in ("y_day", "y_week"):
                for k, (sc, c) in enumerate(
                    (("base", "#306BFF"), ("plus", "#FF4901"), ("placebo", "#888888"))
                ):
                    ax.bar(pos + 0.27 * k, d[f"{bank}|{lab}|{sc}|{fam}"]["mean"] - 0.5, 0.25, bottom=0.5, color=c, label={"base": "experiment 606's summaries", "plus": "plus D1 to D5", "placebo": "plus D1 to D5 of another Primary"}[sc] if (pos == 0) else None)  # fmt: skip
                ticks.append(pos + 0.27)
                labels.append(f"{bname}\n{'day' if lab == 'y_day' else 'week'} lag")
                pos += 1
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("AUROC (higher is better)")
        ax.set_title(
            {
                "train_family": "Shifts of the training family (passes 0 to 11)",
                "held_out": "Held-out shifts (passes 200 to 203)",
            }[fam],
            fontsize=9,
            loc="left",
        )
        ax.set_xticks(ticks, labels, fontsize=8)
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=3, frameon=False, fontsize=8
    )
    fig.suptitle("Can a logistic regression tell that a lag lies inside an injected shift? Mean of 5 held-out months", fontsize=10, x=0.02, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(DOCS / "fig2_auroc.svg")
    plt.close(fig)


def fig_crps() -> None:
    agg = pl.read_parquet(s.OUT / "agg.parquet")
    rng = np.random.default_rng(0)
    arms = ["F1", "F1d46", "F2", "L1", "B0g", "P"]
    fig, axes = plt.subplots(4, 1, figsize=(W, 8.6), sharex=False)
    for ax, (stt, title) in zip(axes, (("inforce", "Shift in force at the origin"), ("near", "Within 24 h of the end"), ("just_ended", "Within 7 days after the end"), ("free", "No shift (event-free rows)")), strict=True):  # fmt: skip
        d = agg.filter(
            (pl.col("set") == "fam") & (pl.col("stratum") == stt) & (pl.col("band") != "other")
        )
        pm = d.group_by("arm", "seed", "month").agg((pl.col("crps_s").sum() / pl.col("n").sum()).alias("v")).group_by("arm", "month").agg(pl.col("v").mean())  # fmt: skip
        ref = pm.filter(pl.col("arm") == "B0").sort("month")["v"].to_numpy()
        for i, a in enumerate(arms):
            x = pm.filter(pl.col("arm") == a).sort("month")["v"].to_numpy() - ref
            m, lo, hi = boot_mean(x, rng)
            ax.plot([lo, hi], [i, i], color=COL[a])
            ax.plot([m], [i], "o", color=COL[a], ms=5)
        ax.axvline(0, color="#888", lw=1)
        ax.set_yticks(range(len(arms)), [LABEL[a] for a in arms], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9, loc="left")
    axes[-1].set_xlabel("Scaled CRPS minus B0's, points of capacity (negative = better)\nDot: mean over 5 months. Line: 95% interval over months.")  # fmt: skip
    fig.tight_layout()
    fig.savefig(DOCS / "fig3_crps.svg")
    plt.close(fig)


def load_month(month: str) -> tuple[dict, dict]:
    z = f650.sources(month=month, bank="new")["bank"]
    f = {a: np.load(f687.OUT / f"{a}_{month}_s0.npz") for a in ("B0", "L1", "F1")}
    return z, f


def event_panel(axs, *, month: str, pa: int, pr: int, label: str) -> None:
    z, f = load_month(month)
    ev = {(int(r["pass"]), int(r["primary"])): r for r in c652.events_table(month=month).iter_rows(named=True)}[(pa, pr)]  # fmt: skip
    idx = f["B0"]["held_idx"]
    sel = (
        (z["pass_"][idx] == pa)
        & (z["prim"][idx] == pr)
        & (z["lead"][idx] >= 24)
        & (z["lead"][idx] < 48)
    )
    ii = idx[sel]
    vi = z["vi"][ii].astype(float)
    order = np.argsort(vi)
    ii, vi = ii[order], vi[order]
    x = (vi - ev["onset_idx"]) / 48.0
    nra_truth = z["y"][ii] * 100
    obs = (z["y"][ii] + z["shift"][ii]) * 100
    k1 = z["q"][ii] * 100
    ax0, ax1 = axs
    for ax in axs:
        ax.axvspan(0, (ev["end_idx"] - ev["onset_idx"]) / 48.0, color="#e6e6e6", lw=0)
    ax0.fill_between(x, k1[:, 1], k1[:, 9], color="#222222", alpha=0.15, lw=0)
    ax0.plot(x, k1[:, 5], color="#222222", label="NRA output (K1), median and 90% band")
    ax0.plot(x, nra_truth, color="#17E58F", label="Demand with no shift (truth for the NRA output)")
    ax0.set_title(f"{label}: NRA output", fontsize=8, loc="left")
    for a in ("B0", "F1"):
        q = f[a]["held_q"][sel][order] * 100
        if a == "F1":
            ax1.fill_between(x, q[:, 1], q[:, 9], color=COL[a], alpha=0.15, lw=0)
        ax1.plot(x, q[:, 5], color=COL[a], label=f"{LABEL[a]}, median")
    ax1.plot(x, obs, color="#222222", label="Observed demand (with the shift)")
    ax1.set_title(f"{label}: prevailing output (day-ahead, 24 to 48 h)", fontsize=8, loc="left")
    ax1.set_xlabel("Days since the shift starts (grey: shift in force)")
    for ax in axs:
        ax.set_ylabel("% of capacity")


def fig_events() -> None:
    nb_of, _cap, is_mw = c650.world()
    picks = []
    for month in ("2025-12", "2026-01"):
        ev = (
            c652.events_table(month=month)
            .filter(pl.col("pass") >= 200)
            .filter(pl.col("end_idx") < c650.WINDOW_START_IDX)
        )
        for r in ev.sort("depth", descending=True).iter_rows(named=True):
            if (
                r["length_days"] in (10, 21, 45)
                and r["depth"] in (0.2, 0.4, 0.65)
                and (not r["out_of_support"] or True)
            ):
                picks.append(
                    (
                        month,
                        int(r["pass"]),
                        int(r["primary"]),
                        bool(is_mw[int(r["primary"])]),
                        r["depth"],
                        r["length_days"],
                    )
                )
    mw = [p for p in picks if p[3] and p[5] <= 21][:3]
    mva = [p for p in picks if not p[3] and p[5] <= 21][:3]
    chosen = mw + mva
    import e00578_common as k578  # ty: ignore[unresolved-import]

    labels = k578.sites()["label"].to_list()
    for part, grp in enumerate((chosen[:3], chosen[3:])):
        fig, axes = plt.subplots(6, 1, figsize=(W, 14), sharex=False)
        for k, (month, pa, pr, mw_, dep, ln) in enumerate(grp):
            event_panel(
                axes[2 * k : 2 * k + 2],
                month=month,
                pa=pa,
                pr=pr,
                label=f"{labels[pr]} ({'signed MW' if mw_ else 'MVA only'}), depth {dep:.2f}, {ln:.0f} days",
            )
        axes[1].legend(
            fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=2, frameon=False
        )
        fig.tight_layout()
        fig.savefig(DOCS / f"fig4_events_{part + 1}.svg")
        plt.close(fig)


def fig_real() -> None:
    z = f650.sources(month="2025-12", bank="new")["real"]
    f = {a: np.load(f687.OUT / f"{a}_2025-12_s0.npz") for a in ("B0", "L1", "F1")}
    idx = f["B0"]["real_idx"]
    sel = z["verified"][idx] & (z["lead"][idx] >= 24) & (z["lead"][idx] < 48)
    ii = idx[sel]
    vi = z["vi"][ii].astype(float)
    order = np.argsort(vi)
    ii, vi = ii[order], vi[order]
    prims = np.unique(z["prim"][ii])
    import e00578_common as k578  # ty: ignore[unresolved-import]

    labels = k578.sites()["label"].to_list()
    fig, axes = plt.subplots(2 * len(prims), 1, figsize=(W, 4.2 * len(prims) * 1.0))
    for k, p in enumerate(prims):
        m = z["prim"][ii] == p
        x = (vi[m] - vi[m].min()) / 48.0
        k1 = z["q"][ii][m] * 100
        axes[2 * k].fill_between(x, k1[:, 1], k1[:, 9], color="#222222", alpha=0.15, lw=0)
        axes[2 * k].plot(x, k1[:, 5], color="#222222", label="NRA output (K1), median and 90% band")
        axes[2 * k].set_title(
            f"{labels[p]}: NRA output, verified transfer of December 2025", fontsize=8, loc="left"
        )
        for a in ("B0", "L1", "F1"):
            q = f[a]["real_q"][sel][order][m] * 100
            axes[2 * k + 1].plot(x, q[:, 5], color=COL[a], label=f"{LABEL[a]}, median")
        axes[2 * k + 1].plot(
            x, z["y"][ii][m] * 100, color="#222222", ls=":", label="Observed demand"
        )
        axes[2 * k + 1].set_title(
            f"{labels[p]}: prevailing output (day-ahead)", fontsize=8, loc="left"
        )
        axes[2 * k + 1].set_xlabel("Days since the first scored valid time of the verified rows")
        for ax in axes[2 * k : 2 * k + 2]:
            ax.set_ylabel("% of capacity")
    axes[1].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.6), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(DOCS / "fig5_verified.svg")
    plt.close(fig)


def fig_horizon() -> None:
    month = "2026-01"
    z, f = load_month(month)
    nb_of, _cap, is_mw = c650.world()
    ev = (
        c652.events_table(month=month)
        .filter((pl.col("pass") >= 200) & (pl.col("length_days") == 21))
        .sort("depth", descending=True)
    )
    idx = f["B0"]["held_idx"]
    for r in ev.iter_rows(named=True):
        pa, pr = int(r["pass"]), int(r["primary"])
        if not (is_mw[pr] and r["depth"] >= 0.2):
            continue
        base = (z["pass_"][idx] == pa) & (z["prim"][idx] == pr)
        ois = np.unique(z["oi"][idx][base])
        want = (r["onset_idx"] // 48) * 48 + 18 + 10 * 48
        if len(ois) > 20 and np.abs(ois - want).min() <= 48:
            origin = int(ois[np.argmin(np.abs(ois - want))])
            break
    fig, axes = plt.subplots(2, 1, figsize=(W, 6.2))
    for ax, (lo, hi, title) in zip(axes, ((24, 48, "Day-ahead forecasts from 14 consecutive daily origins (24 to 48 h)"), (24, 240, "One forecast from a single origin (24 to 240 h; the stored rows stop at 10 days)")), strict=True):  # fmt: skip
        sel = base & (z["lead"][idx] >= lo) & (z["lead"][idx] < hi)
        if lo == 24 and hi == 48:
            sel &= (z["oi"][idx] >= origin - 7 * 48) & (z["oi"][idx] < origin + 7 * 48)
        else:
            sel &= z["oi"][idx] == origin
        ii = idx[sel]
        order = np.argsort(z["vi"][ii])
        ii = ii[order]
        x = (z["vi"][ii] - r["onset_idx"]) / 48.0
        ax.axvspan(0, (r["end_idx"] - r["onset_idx"]) / 48.0, color="#e6e6e6", lw=0)
        for a in ("L1", "F1"):
            q = f[a]["held_q"][sel][order] * 100
            ax.fill_between(x, q[:, 1], q[:, 9], color=COL[a], alpha=0.12, lw=0)
            ax.fill_between(
                x, q[:, 2], q[:, 8], color=COL[a], alpha=0.12, lw=0
            )  # 80% band, nested inside the 90%
            ax.plot(x, q[:, 5], color=COL[a], label=f"{LABEL[a]}: median, 80% and 90% bands")
        ax.plot(x, (z["y"][ii] + z["shift"][ii]) * 100, color="#222222", label="Observed demand")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("% of capacity")
    axes[1].set_xlabel("Days since the shift starts (grey: shift in force)")
    axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(DOCS / "fig6_horizon.svg")
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    fig_reaction()
    fig_auroc()
    fig_crps()
    fig_events()
    fig_real()
    fig_horizon()


if __name__ == "__main__":
    main()
