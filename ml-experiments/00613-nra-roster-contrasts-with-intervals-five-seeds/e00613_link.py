# ruff: noqa: D103, E501
"""Experiment 613: lay out the private data folder (CPU).

Seeds 0 and 1 (experiment 612) sit beside seeds 2 to 4.

For every arm, this creates `arms/NAME/work/{fits,inject}/...` in this experiment's folder and
symlinks experiment 612's seed 0 and seed 1 forecast files into it. The new fits of seeds 2 to 4 are
written next to the links, so the scorer sees all five seeds in one folder. It also creates the two
chain folders (`chains/d5`, `chains/o2`) with symlinks to the stored seed 0 and 1 weights.

Experiment 577 stored O2 pre-training for seeds 0, 1, and 5 to 9 only, so the K3 O2 comparison uses seeds
0, 1, 5, 6, and 7, and `k3_d5b` holds the matching D5 fits.

Usage: python e00613_link.py
"""

from pathlib import Path

from studies.sources import ML_EXPERIMENTS_DATA_DIR

OLD = ML_EXPERIMENTS_DATA_DIR / "00612-nra-combined-networks-three-class-roster-completion"
NEW = ML_EXPERIMENTS_DATA_DIR / "00613-nra-roster-contrasts-with-intervals-five-seeds"
OLD_SEEDS = ("_s0.", "_s1.")
ARMS = ("k1_d5", "k1_o2", "k3_d5", "k3_o2", "noe3_d5")
"""Arms whose seeds 0 and 1 are reused. The candidate's seeds 0 and 1 are linked from `c2_e3`."""


def link_tree(*, src: Path, dst: Path) -> int:
    n = 0
    for f in sorted(src.rglob("*")):
        if f.is_dir() or not any(s in f.name for s in OLD_SEEDS):
            continue
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() and not target.is_symlink():
            target.symlink_to(f.resolve())
            n += 1
    return n


def main() -> None:
    for arm in (*ARMS, "c2_e3"):
        for sub in ("fits", "inject"):
            src = OLD / "arms" / arm / "work" / sub
            if src.exists():
                n = link_tree(src=src, dst=NEW / "arms" / arm / "work" / sub)
                print(f"{arm}/{sub}: {n} links")
    n = link_tree(
        src=OLD / "arms" / "k3_d5" / "work" / "fits", dst=NEW / "arms" / "k3_d5b" / "work" / "fits"
    )
    print(f"k3_d5b/fits: {n} links (the D5 twin for the K3 comparison on seeds 0, 1, 5, 6, 7)")
    d591 = (
        ML_EXPERIMENTS_DATA_DIR
        / "00591-shared-lag-encoder-auxiliary-shift-depth-loss/chain_shared/work"
    )
    for sub in ("pretrained_P2", "step2/P5"):
        n = link_tree(src=d591 / sub, dst=NEW / "chains" / "d5" / "work" / sub)
        print(f"chains/d5/{sub}: {n} links")
    (NEW / "chains" / "d5" / "work" / "stats" / "B1").mkdir(parents=True, exist_ok=True)
    n = link_tree(src=OLD / "arms/k1_o2/chain/work", dst=NEW / "chains" / "o2" / "work")
    print(f"chains/o2: {n} links")
    (NEW / "chains" / "o2" / "work" / "stats" / "B1").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
