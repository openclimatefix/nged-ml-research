"""Fit the running arm's T-all monthly fits (a wrapper, so pickled classes live in `e00376_tall`).

Set `E00376_ARM` to the arm, then pass the sites:

    uv run --no-sync python ml-experiments/00180-*/e00376_fit.py --cadences monthly --sites D5
"""

import e00376_tall as tall

if __name__ == "__main__":
    tall.main()
