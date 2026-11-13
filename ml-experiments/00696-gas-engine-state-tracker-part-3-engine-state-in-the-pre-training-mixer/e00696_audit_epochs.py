# ruff: noqa: E501, F405, PTH123, PTH207, SIM115, UP031
"""Audit of experiment 696, step 6: epochs and validation loss of every stage, from the training logs."""

import collections
import glob
import pickle
import re

import numpy as np
from e00696_audit_common import *  # noqa: F403

E5 = str(E5)
E6 = str(E6)
pat = {
    "pre": re.compile(r"pretrain P5 (\S+) s(\d+): (\d+) epochs, val NLL ([\d.]+)"),
    "st2": re.compile(r"step 2 \(P5\) (\S+) s(\d+): [\d,]+ rows, (\d+) epochs, val NLL ([\d.]+)"),
    "st3": re.compile(r"P5/(\S+)_d0_s(\d+): (\d+) epochs"),
}
res = collections.defaultdict(dict)
for root, tag in ((E5, "695"), (E6, "696")):
    for arm in ("A0", "W0", "S1", "SP", "S1d", "WST", "WS1", "SZ", "G0", "G0W"):
        for f in glob.glob(f"{root}/runs/{arm}/logs/*.log"):
            st = (
                "pre"
                if "_pretrain" in f
                else "st2"
                if "_step2" in f
                else "st3"
                if re.search(r"_fit[_.]", f)
                else None
            )
            if not st:
                continue
            for line in open(f, errors="ignore"):
                m = pat[st].search(line)
                if m:
                    res[(tag, arm, st)][(m.group(1), int(m.group(2)))] = (
                        int(m.group(3)),
                        float(m.group(4)) if st != "st3" else np.nan,
                    )
for k in sorted(res):
    v = res[k]
    ep = np.array([x[0] for x in v.values()])
    nll = np.array([x[1] for x in v.values()])
    s03 = [x[0] for (mo, s), x in v.items() if s <= 2]
    s39 = [x[0] for (mo, s), x in v.items() if s >= 3]
    print(
        k,
        "n",
        len(v),
        "epochs mean %.1f" % ep.mean(),
        "at cap60 %d" % (ep == 60).sum() if k[2] == "pre" else "",
        "nll %.4f" % np.nanmean(nll),
        "s0-2 %.1f" % np.mean(s03) if s03 else "-",
        "s3-9 %.1f" % np.mean(s39) if s39 else "-",
    )
pickle.dump(dict(res), open(OUT / "epochs.pkl", "wb"))
