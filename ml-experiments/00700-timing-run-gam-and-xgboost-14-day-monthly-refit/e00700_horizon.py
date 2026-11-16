"""Experiment 700: lengthen the GAM's stage 1 window from 672 to 678 half-hours.

Import this module straight after `e00377_common` and before `e00377_ens` or `e00377_run`, because
`e00377_ens` binds `HORIZON_HALF_HOURS` by value when it is imported.

Experiment 377 sets the window to 672 half-hours (336 hours), counted from the 09:00 UTC native
origin. A contract origin of 12:00 UTC needs valid times up to 12:00 UTC 14 days later, which is
339 hours after the native origin, so the window has to be 678 half-hours. The stored ensemble
arrays hold 702 half-hours (351 hours), so no weather outside the origin's own run is needed.
"""

from typing import Final

import e00377_common as common  # ty: ignore[unresolved-import]

HORIZON_HALF_HOURS: Final[int] = 678
common.HORIZON_HALF_HOURS = HORIZON_HALF_HOURS  # ty: ignore[invalid-assignment]
common.dp_ens_weather.HORIZON_HALF_HOURS = HORIZON_HALF_HOURS  # ty: ignore[invalid-assignment]
