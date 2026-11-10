"""Count what NGED's Embedded Capacity Register lists behind each of the 16 Primaries we forecast.

The asset-inventory simulator page quotes these counts as the priors the register could give a
fitted simulator. The script reads the August 2026 release of the register (`studies.ecr_reader`),
splits every row into its up to three technology slots, keeps the connected slots whose `Primary`
column names one of our Primaries, and prints one markdown table: the number of slots and the
registered MW per asset class. It also prints the slot totals by connection voltage (a slot at
11 kV or below sits behind the Primary's meter), how many slots carry a connection date, and the
slots accepted to connect but not yet connected.
No customer name, site, or coordinate is printed.

Wrangle T1 (D15) and Wrangle T2 (D18) share one register name, so they share one row.

Usage (one core, a few seconds), from the repository root::

    uv run --no-sync python \
        docs/ml-experiments/themes/gross-demand-simulation/scripts/register_coverage.py
"""

from typing import Final, Literal

import polars as pl
from studies.ecr_reader import ECR_PATH, REGISTER_SHEETS, read_sheet

ConnectionStatusType = Literal["Connected", "Accepted to connect"]

AssetClassType = Literal[
    "solar", "wind", "storage", "gas or oil engine", "bioenergy", "type not given", "other"
]

PRIMARY_REGISTER_NAMES: Final[dict[str, str]] = {
    "D2": "Warth Lane Skegness",
    "D3": "Sleaford Road Boston",
    "D4": "Ingoldmells",
    "D5": "Kirton",
    "D8": "Marsh Lane",
    "D9": "Trusthorpe",
    "D10": "Spilsby",
    "D11": "Horncastle",
    "D12": "Stickney",
    "D14": "Mount Bridge Boston",
    "D15 and D18": "Wrangle",
    "D16": "Donington",
    "D19": "Chapel St Leonards",
    "D21": "Alford",
    "D22": "Langrick",
}
"""D code of each Primary and the start of its name in the register's `Primary` column."""

NOT_GIVEN: Final[str] = "data not available"
"""The register's text for a field it does not know."""

NOT_APPLICABLE: Final[str] = "data not applicable"
"""The register's text for an unused technology slot."""

BEHIND_METER_MAX_KV: Final[float] = 11.0
"""A slot connected at this voltage or below sits behind the Primary's 11 kV meter."""


def _asset_class() -> pl.Expr:
    text = pl.col("source").str.to_lowercase() + " " + pl.col("technology").str.to_lowercase()
    return (
        pl.when(pl.col("source") == NOT_GIVEN)
        .then(pl.lit("type not given"))
        .when(text.str.contains("solar"))
        .then(pl.lit("solar"))
        .when(text.str.contains("wind"))
        .then(pl.lit("wind"))
        .when(text.str.contains("stored energy|storage"))
        .then(pl.lit("storage"))
        .when(text.str.contains("bio|landfill|waste"))
        .then(pl.lit("bioenergy"))
        .when(text.str.contains("fossil|gas|oil|diesel"))
        .then(pl.lit("gas or oil engine"))
        .otherwise(pl.lit("other"))
    )


def read_slots(*, status: ConnectionStatusType = "Connected") -> pl.DataFrame:
    """Return one row per technology slot of `status` whose Primary is one we forecast.

    Args:
        status: The register's connection status to keep.

    Returns:
        Columns `d_code`, `asset_class`, `mw`, `voltage`, and `has_connection_date`.
    """
    pieces = [
        read_sheet(path=ECR_PATH, sheet_name=sheet).select(
            primary=pl.col("Primary"),
            source=pl.col(f"Energy Source {slot}"),
            technology=pl.col(f"Energy Conversion Technology {slot}").fill_null(""),
            mw=pl.col(
                f"Energy Source & Energy Conversion Technology {slot} - Registered Capacity (MW)"
            ).cast(pl.Float64, strict=False),
            status=pl.col("Connection Status"),
            kv=pl.col("Point of Connection (POC) Voltage (kV)").cast(pl.Float64, strict=False),
            connected=pl.col("Date Connected"),
        )
        for sheet in REGISTER_SHEETS
        for slot in (1, 2, 3)
    ]
    slots = pl.concat(pieces).filter(
        pl.col("source").is_not_null(),
        pl.col("status") == status,
        # Slots 2 and 3 hold the not-applicable text when a site has one technology.
        pl.col("source") != NOT_APPLICABLE,
    )
    d_code = pl.lit(None, dtype=pl.String)
    for code, name in PRIMARY_REGISTER_NAMES.items():
        d_code = (
            pl.when(pl.col("primary").str.starts_with(name)).then(pl.lit(code)).otherwise(d_code)
        )
    return (
        slots.with_columns(d_code=d_code, asset_class=_asset_class())
        .filter(pl.col("d_code").is_not_null())
        .select(
            "d_code",
            "asset_class",
            "mw",
            voltage=pl.when(pl.col("kv").is_null())
            .then(pl.lit("voltage not given"))
            .when(pl.col("kv") <= BEHIND_METER_MAX_KV)
            .then(pl.lit("11 kV or below"))
            .otherwise(pl.lit("above 11 kV")),
            has_connection_date=pl.col("connected").cast(pl.Float64, strict=False).is_not_null(),
        )
    )


def coverage_table(slots: pl.DataFrame) -> pl.DataFrame:
    """Return one cell per Primary and asset class: the slot count and the registered MW.

    Args:
        slots: The output of `read_slots`.

    Returns:
        One row per D code and one string column per asset class, each cell "count / MW".
    """
    cells = (
        slots.group_by("d_code", "asset_class")
        .agg(n=pl.len(), mw=pl.col("mw").sum())
        .with_columns(cell=pl.format("{} / {}", pl.col("n"), pl.col("mw").round(2)))
    )
    return (
        cells.pivot(on="asset_class", index="d_code", values="cell")
        .fill_null("0")
        .select("d_code", *sorted(cells["asset_class"].unique()))
        .sort(pl.col("d_code").str.extract(r"(\d+)").cast(pl.Int32))
    )


def main() -> None:
    """Print the coverage table, slot totals, and the accepted-to-connect slots by class."""
    slots = read_slots()
    table = coverage_table(slots)
    print("| " + " | ".join(table.columns) + " |")
    print("|" + "---|" * table.width)
    for row in table.iter_rows():
        print("| " + " | ".join(row) + " |")
    print(f"connected slots at our Primaries: {slots.height}")
    print(slots.group_by("voltage").agg(n=pl.len(), mw=pl.col("mw").sum()))
    print(f"slots with a connection date: {slots['has_connection_date'].sum()}")
    accepted = read_slots(status="Accepted to connect")
    print("accepted-to-connect slots at our Primaries, by asset class:")
    print(
        accepted.group_by("asset_class").agg(n=pl.len(), mw=pl.col("mw").sum()).sort("asset_class")
    )


if __name__ == "__main__":
    main()
