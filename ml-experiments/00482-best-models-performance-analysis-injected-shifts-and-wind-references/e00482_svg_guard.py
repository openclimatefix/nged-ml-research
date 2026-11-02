"""Fail when a chart draws a mark in the colour of the chart background.

A line, area, or legend symbol coloured like the background is invisible, and nothing else in the
chart-making code notices. ``assert_no_invisible_marks`` reads a saved SVG and raises if any element
other than the background rectangles has a ``stroke`` or ``fill`` equal to the background colour.
"""

from pathlib import Path
from xml.etree import ElementTree

import plotting.ocf_theme as ocf

_BACKGROUND_COLOUR = ocf.BACKGROUND.lower()


def invisible_mark_count(*, svg_path: Path) -> int:
    """Count the elements in an SVG whose stroke or fill equals the chart background colour.

    The two background rectangles that Vega writes (the canvas and the plot-area ``background``
    class) are not counted.

    Args:
        svg_path: The SVG file to read.

    Returns:
        The number of elements drawn in the background colour.
    """
    count = 0
    for element in ElementTree.parse(svg_path).iter():
        tag = element.tag.rpartition("}")[2]
        is_background = (
            tag == "rect"
            or "background" in element.get("class", "")
            or element.get("d", "").startswith("M0 0h")
        )
        if is_background:
            continue
        colours = (element.get("stroke", ""), element.get("fill", ""))
        count += any(colour.lower() == _BACKGROUND_COLOUR for colour in colours)
    return count


def assert_no_invisible_marks(*, svg_path: Path) -> None:
    """Raise if the SVG draws any mark in the chart background colour.

    Args:
        svg_path: The SVG file to check.

    Raises:
        ValueError: If one or more marks are drawn in the background colour.
    """
    count = invisible_mark_count(svg_path=svg_path)
    if count:
        message = (
            f"{svg_path.name}: {count} mark(s) are drawn in the background colour"
            f" {ocf.BACKGROUND} and so are invisible."
        )
        raise ValueError(message)
