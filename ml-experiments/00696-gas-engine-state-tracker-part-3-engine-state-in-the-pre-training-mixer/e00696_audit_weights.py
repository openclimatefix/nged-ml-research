# ruff: noqa: E501, ANN001, ANN201, D103, F405, PIE810, PTH207, UP031
"""Audit of experiment 696, step 5: the state reaches the encoders; the fine-tune mixer has no state input."""

import glob

import torch  # ty: ignore[unresolved-import]
from e00696_audit_common import *  # noqa: F403

E5 = str(E5)
E6 = str(E6)


def pre(root, arm, m, s):
    return torch.load(
        f"{root}/runs/{arm}/chains/5/{m}/work/pretrained_P2/{m}_s{s}.pt", map_location="cpu"
    )


m = "2026-01"
sd = {k: pre(E6, k, m, 4) for k in ("A0", "S1", "SP")}
sd["A0s5"] = pre(E6, "A0", m, 5)
sd["A0_695_s0"] = pre(E5, "A0", m, 0)
sd["S1s0"] = pre(E6, "S1", m, 0)
k0 = sd["A0"]
print([(k, tuple(v.shape)) for k, v in k0.items() if "mixer_pre.0" in k or "mixer_ft.0" in k])
print("S1 mixer_pre.0 weight shape", tuple(sd["S1"]["mixer_pre.0.weight"].shape))


def rel(a, b, prefix):
    ks = [k for k in a if k.startswith(prefix)]
    num = sum(((a[k] - b[k]) ** 2).sum() for k in ks)
    den = sum((b[k] ** 2).sum() for k in ks)
    return float((num / den) ** 0.5)


for p in ("weather", "datetime", "embedding"):
    print(
        p,
        "S1s4 vs A0s4 %.3f" % rel(sd["S1"], sd["A0"], p),
        "SPs4 vs A0s4 %.3f" % rel(sd["SP"], sd["A0"], p),
        "A0s5 vs A0s4 %.3f" % rel(sd["A0s5"], sd["A0"], p),
        "S1s0 vs A0(695)s0 %.3f" % rel(sd["S1s0"], sd["A0_695_s0"], p),
    )
w = sd["S1"]["mixer_pre.0.weight"]
print(
    "S1 mixer_pre.0 |w| mean, original cols %.4f, state cols" % w[:, :-3].abs().mean(),
    w[:, -3:].abs().mean(0),
)
w = sd["SP"]["mixer_pre.0.weight"]
print("SP state cols", w[:, -3:].abs().mean(0))
# stage-2 and stage-3 files: find
for arm in ("S1", "A0"):
    fs = (
        sorted(glob.glob(f"{E6}/runs/{arm}/chains/5/{m}/work/step2/**/*.pt", recursive=True))[:1]
        + sorted(glob.glob(f"{E6}/runs/{arm}/arms/k1_5w_sp1/work/**/*{m}*s4*.pt", recursive=True))[
            :2
        ]
    )
    for f in fs:
        s = torch.load(f, map_location="cpu")
        s = s.get("state_dict", s) if isinstance(s, dict) else s
        print(
            arm,
            f.split("work/")[-1],
            {
                k: tuple(v.shape)
                for k, v in s.items()
                if k.endswith("mixer_ft.0.weight") or k.endswith("mixer_pre.0.weight")
            },
        )
        if arm == "S1":
            a = (
                torch.load(f.replace("/S1/", "/A0/"), map_location="cpu")
                if glob.glob(f.replace("/S1/", "/A0/"))
                else None
            )
        # frozen encoders: equal to pretrain?
        print(
            "  weather equal to pretrained:",
            all(
                torch.equal(s[k], pre(E6, arm, m, 4)[k])
                for k in s
                if k.startswith("weather") and k in pre(E6, arm, m, 4)
            ),
            " datetime equal:",
            all(torch.equal(s[k], pre(E6, arm, m, 4)[k]) for k in s if k.startswith("datetime")),
        )
