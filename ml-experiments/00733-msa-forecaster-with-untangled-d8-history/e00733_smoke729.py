"""Experiment 733: run experiment 729's real-file checks against the banks this experiment reads.

Experiment 729's `tests/smoke_real_files.py` reads experiment 729's own test bank, and that file
on the data folder is the superseded run whose S2 solar output is silent on every clear base day
(0 of 321 clear base days have output). Experiment 733 reads experiment 730's banks. This driver
runs every check of experiment 729's script and points the bank check at experiment 730's test
bank. It runs as a subprocess of the smoke stage with experiment 729's folders on `PYTHONPATH`.
"""

import sys

import e00733_paths as paths


def main() -> int:
    """Run the checks of `smoke_real_files.CHECKS` on experiment 730's test bank.

    Returns:
        0 when every check passed, 1 otherwise.
    """
    import e00729_stages as stages  # ty: ignore[unresolved-import]
    import e00730_banks as banks730  # ty: ignore[unresolved-import]
    import smoke_real_files as script  # ty: ignore[unresolved-import]

    root_730 = paths.root_730()

    def bank_path(*, kind: str) -> object:
        return banks730.bank_path(root=root_730, kind=kind)

    stages.bank_path = bank_path
    failed = 0
    for name, check in script.CHECKS.items():
        try:
            line = check()
            if name == "bank_s2_solar" and "test bank present: True" not in line:
                msg = "experiment 730's test bank is absent, so the bank check read nothing"
                raise FileNotFoundError(msg)  # noqa: TRY301
            print(f"{name}: {line}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"{name}: FAILED with {type(error).__name__}: {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
