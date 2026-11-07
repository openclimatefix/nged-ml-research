"""Experiment 625 toy test: experiment 436's joint reshape scrambles members; the fix does not.

Builds paths whose value is the member index at every node and step. The joint vector of member m
must then hold only the value m. Experiment 436's `transpose(1, 0, 2).reshape(M, -1)` fails that
check, and the flattening used in this experiment (and in experiment 434) passes it. Run with
`uv run --no-sync python ml-experiments/00625-*/e00625_check_joint_reshape.py`; it exits non-zero
if the old form passes or the new form fails.
"""

import numpy as np

members, nodes, steps = 5, 4, 6
paths = np.broadcast_to(np.arange(members)[:, None, None], (members, nodes, steps)).astype(float)
sub = np.arange(0, steps, 2)
old = paths[:, :, sub].transpose(1, 0, 2).reshape(members, -1)
new = paths[:, :, sub].reshape(members, -1)
ok_new = all((new[m] == m).all() for m in range(members))
ok_old = all((old[m] == m).all() for m in range(members))
print("old reshape keeps members intact:", ok_old)
print("new reshape keeps members intact:", ok_new)
if ok_old or not ok_new:
    raise SystemExit(1)
