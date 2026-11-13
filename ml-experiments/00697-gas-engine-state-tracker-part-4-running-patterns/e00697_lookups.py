r"""A small open lookup near D8: EAC reserve and response auction results by post code.

    uv run --no-sync python $E/00697-*/e00697_lookups.py

Both are descriptive context. **A match is a match on location and technology, never a confirmed
identity**, and the matching rows are written only into the private data folder
(`lookups/`), never printed in full and never put in `result.md` with a name, an exact post code,
or a unit identifier. The script prints counts only.

D8's production metadata holds no post code. It holds the centre of the substation's supply area,
which this script uses locally as D8's position. **That position is never sent anywhere**: the
script sends only public post codes (those listed in the EAC results) to `postcodes.io` to get
their coordinates, then measures the distance to D8 on this machine.

The steps are: (1) read D8's area centre from `NGED/metadata.parquet` through the panel key, which
is the `time_series_id`; (2) ask NESO's SQL endpoint for the row count per post code and service
of the gas reciprocating
engines in the EAC results, a query that names only the technology; (3) geocode the post codes
here; (4) keep those within `RADIUS_KM`; (5) write the matching counts to the private folder and
print their total. A Capacity Market register
lookup is not made, because a private dossier already answers it.
"""

from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Final

import polars as pl
import requests
from e00694_common import load_panel  # ty: ignore[unresolved-import]
from e00697_common import DATA_DIR  # ty: ignore[unresolved-import]
from e00697_signals_download import NESO, get_json  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

LOOKUPS_DIR: Final[Path] = DATA_DIR / "lookups"
METADATA_PATH: Final = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet"
RADIUS_KM: Final[float] = 5.0
"""The distance from the centre of D8's supply area within which a post code counts as near."""
EARTH_RADIUS_KM: Final[float] = 6371.0
EAC_RESOURCE_ID: Final[str] = "a63ab354-7e68-44c2-ad96-c6f920c30e85"
EAC_TECHNOLOGY: Final[str] = "Gas Reciprocating Engines"
"""The only filter sent to NESO: the technology of a gas engine."""
POSTCODE_API: Final[str] = "https://api.postcodes.io/postcodes"
POSTCODE_BATCH: Final[int] = 100


def d8_position() -> tuple[float, float]:
    """Return the latitude and longitude of the centre of D8's supply area, kept local."""
    key = next(k for k, v in load_panel().items() if v["site"] == "D8")
    row = pl.read_parquet(METADATA_PATH).filter(pl.col("time_series_id") == key)
    if row.height != 1:
        raise SystemExit("D8's panel key does not match exactly one row of the production metadata")
    return float(row["area_center_lat"][0]), float(row["area_center_lon"][0])


def distance_km(*, a: tuple[float, float], b: tuple[float, float]) -> float:
    """Return the great-circle distance in kilometres between two latitude and longitude pairs."""
    lat1, lon1, lat2, lon2 = map(radians, (*a, *b))
    h = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(h))


def geocode(*, postcodes: list[str]) -> dict[str, tuple[float, float]]:
    """Return the coordinates of public post codes from postcodes.io; unknown ones are left out."""
    out: dict[str, tuple[float, float]] = {}
    for i in range(0, len(postcodes), POSTCODE_BATCH):
        batch = postcodes[i : i + POSTCODE_BATCH]
        reply = requests.post(POSTCODE_API, json={"postcodes": batch}, timeout=60)
        reply.raise_for_status()
        for item in reply.json()["result"]:
            found = item["result"]
            if found and found.get("latitude") is not None:
                out[item["query"]] = (found["latitude"], found["longitude"])
    return out


def near_postcodes(*, postcodes: list[str], centre: tuple[float, float]) -> set[str]:
    """Return the post codes within `RADIUS_KM` of the centre."""
    coordinates = geocode(postcodes=sorted(set(postcodes)))
    return {p for p, c in coordinates.items() if distance_km(a=centre, b=c) <= RADIUS_KM}


def eac(*, centre: tuple[float, float]) -> tuple[int, int]:
    """Write EAC aggregates near D8 to a private file; return (post codes, half-hourly rows).

    The query names only a technology and asks the server to count rows per post code and service,
    so the server receives nothing about D8's position. The distance filter runs here.
    """
    reply = get_json(
        url=f"{NESO}/datastore_search_sql",
        params={
            "sql": (
                'SELECT "postCode", "serviceType", COUNT(*) AS n_rows, '
                'AVG("executedQuantity") AS mean_executed_mw '
                f'FROM "{EAC_RESOURCE_ID}" WHERE "technologyType" = \'{EAC_TECHNOLOGY}\' '
                'GROUP BY "postCode", "serviceType"'
            )
        },
    )
    table = pl.DataFrame(reply["result"]["records"], infer_schema_length=None)
    table = table.filter(pl.col("postCode").is_not_null())
    near = near_postcodes(postcodes=table["postCode"].to_list(), centre=centre)
    hits = table.filter(pl.col("postCode").is_in(sorted(near)))
    hits.write_csv(LOOKUPS_DIR / "eac_near_d8_PRIVATE.csv")
    return hits["postCode"].n_unique(), int(hits["n_rows"].cast(pl.Int64).sum() or 0)


def main() -> None:
    """Run both lookups and print counts only."""
    LOOKUPS_DIR.mkdir(parents=True, exist_ok=True)
    centre = d8_position()
    n_codes, n_rows = eac(centre=centre)
    print(f"EAC post codes within {RADIUS_KM:.0f} km: {n_codes}; result rows at them: {n_rows}")
    print("Matches are on post code and technology only, not confirmed identities.")


if __name__ == "__main__":
    main()
