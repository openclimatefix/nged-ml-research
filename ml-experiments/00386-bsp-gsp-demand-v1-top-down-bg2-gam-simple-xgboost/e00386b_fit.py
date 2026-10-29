"""Fit the running arm's T-all monthly fits (a wrapper, so pickled classes live in `e00386b_tall`).

Set `E00386_SOURCE` to `control`, `mva`, or `d15`, then pass the sites:

    uv run --no-sync python ml-experiments/00377-*/e00386b_fit.py --cadences monthly --sites D5
"""

import e00386b_tall as tall

if __name__ == "__main__":
    tall.main()
