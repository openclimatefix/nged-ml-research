"""Experiment 708: reload a bundle in a clean interpreter and reproduce its fixtures.

Usage: `python e00708_check.py BUNDLE_DIR`. The bundle's own `reference_check.py` runs in a
subprocess with `-I` (no environment `PYTHONPATH`, no script-directory imports), so it can import
only the third-party packages `torch`, `numpy`, and `polars`, and reads only files of the bundle.
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the bundle's check."""
    bundle = Path(sys.argv[1]).resolve()
    result = subprocess.run(
        [sys.executable, "-I", str(bundle / "reference_check.py"), str(bundle)], check=False
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
