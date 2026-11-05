# ruff: noqa: E501, E402, ANN001
"""Experiment 565: real-event leak of M0 and D5 in experiment 556's harness (daily refits, seed 0).

Usage: python e00565_events.py ARM   with ARM `M0d` or `D5d`.

Experiment 556's `real_events` fits, for each labelled event, once on metered targets and once on
NRA-corrected targets at the first daily cutoff three or more days after the event. The absorbed share
is the sign of the event times the change in the median over the next 7 days at the event's Primary,
divided by the event's labelled depth. This script runs that function unchanged, with `Cell` replaced
by a subclass that builds the network, the extra input columns, and the fine-tune of the arm in
`e00565_fit` (frozen encoders, drift penalty, fresh mixer), and with controls stored under this
experiment's folder.
"""

import sys

ARM = sys.argv[1]
assert ARM in {"M0d", "D5d"}, ARM
sys.argv = [
    sys.argv[0],
    "fit",
    "--arm",
    ARM,
]  # the argument parser of e00565_fit reads this at import

import e00510_common as c  # ty: ignore[unresolved-import]
import e00556_fit as f556  # ty: ignore[unresolved-import]
import e00565_fit as f565  # ty: ignore[unresolved-import]
import torch  # ty: ignore[unresolved-import]

SEED = 0
f556.SEED = SEED
f556.WORK = f565.W / f"events_{ARM}"


class Cell565(f556.Cell):
    """556's `Cell` with the arm's network and fine-tune."""

    def new(self) -> torch.nn.Module:
        """Return the arm's network with the fold's pretrained weights and a fresh mixer."""
        model = f565.new_model(n_primaries=len(self.mr.base.target_mean)).to(self.device)
        f565.load_pretrained_fresh(
            model=model, path=f565.pretrained_file(month=self.month, seed=SEED)
        )
        return model.to(self.device)

    def stored_path(self, *, arm: f556.Arm, cut) -> None:
        """Return no stored weights: this experiment's controls are fitted here and saved."""
        return

    def fit(
        self, *, arm: f556.Arm, cut, scen: f556.Scenario | None, origin, sign: int, ref, mad_model
    ) -> torch.nn.Module:
        """Fit one fine-tune: the arm's rows at `cut`, with the scenario's injection if given."""
        assert arm.clip is None
        assert not arm.embargo_days
        r = self.select(arm=arm, cut=cut)
        shift = None
        if scen is not None:
            assert origin is not None
            shift, _ = self.injection(r=r, scen=scen, origin=origin, sign=sign, mad=None)
        refit = c.Refit(r.index, r.y, r.weight, r.is_val, r.report)
        shift_z = (
            None if shift is None else shift / self.mr.base.target_std[self.mr.primary[r.index]]
        )
        t = self.g.refit_tensors(refit=refit, shift_z=shift_z)
        model = self.new()
        spec = f565.SPEC
        f565.train(
            model=model, x=t["x"], z=t["z"], w=t["w"], p=t["p"], is_val=t["v"], seed=SEED,
            lr=spec.lr, max_epochs=spec.max_epochs, fresh_lr=f565.FRESH_LR, frozen=spec.frozen,
            anchor=None, lam=0.0, device=self.device,
        )  # fmt: skip
        del t
        return model


f556.Cell = Cell565
if __name__ == "__main__":
    f556.real_events(arms=["F1d"], months=list(c.MONTHS), device="cuda")
