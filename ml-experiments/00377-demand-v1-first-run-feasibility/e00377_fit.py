"""Fit the running arm's T-all monthly fits (a wrapper, so pickled classes live in `e00377_tall`).

Set `E00377_SOURCE` to `control`, `mva`, or `d15`, then pass the sites:

    uv run --no-sync python ml-experiments/00377-*/e00377_fit.py --cadences monthly --sites D5
"""

import e00377_tall as tall

if __name__ == "__main__":
    tall.main()
