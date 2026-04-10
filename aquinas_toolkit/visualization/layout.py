"""Canonical sensor parsing and analytical bridge layout helpers.

The structural semantics in this module come from the AQUINAS dataset
README and handbook:

- `OLD` / `NEW` deck identity
- `S1` / `S2` span identity
- `MID`, `INT`, and `SHE` section meanings
- `ACC_Y` existing on `DO` only in the released files

Important limitation:

- The handbook defines `MID` and `INT` section locations explicitly
  (`1/2`, `2/3`, and `1/3` of span length), but the `SHE` locations are
  only described as "near pier 1". In this viewer, `S1_SHE` and
  `S2_SHE` are therefore placed using a normalized approximation close
  to the span boundary, not a surveyed physical coordinate.
"""

from __future__ import annotations

from typing import Any

# These coordinates describe a normalized analytical schematic, not a
# surveyed bridge model. The viewer and exporter both rely on this same
# coordinate system, so changes here should be treated as schema-level.
#
# Provenance:
# - `S1_MID`, `S1_INT`, `S2_INT`, and `S2_MID` follow the handbook
#   section definitions directly.
# - `S1_SHE` and `S2_SHE` are intentional approximations because the
#   handbook only states that the shear sensors are located "near pier 1".
SECTION_X = {
    "S1_MID": 0.50,
    "S1_INT": 0.67,
    "S1_SHE": 0.92,
    "S2_SHE": 1.08,
    "S2_INT": 1.33,
    "S2_MID": 1.50,
}
SPAN_BOUNDARIES = {
    "S1": (0.0, 1.0),
    "S2": (1.0, 2.0),
}
DECK_Z_CENTERS = {
    "compact": {"OLD": 1.10, "NEW": -1.10},
    "exploded": {"OLD": 1.90, "NEW": -1.90},
}
SIDE_Z_OFFSET = {"UP": 0.24, "DO": -0.24}
HEIGHT_BY_GLYPH = {
    "vertical-arrow": 0.30,
    "transverse-arrow": 0.18,
    "lower-strain": -0.30,
    "upper-strain": 0.30,
    "shear-strain": 0.0,
}
X_NUDGE = {
    "MID_ACC_Z": -0.02,
    "MID_ACC_Y": 0.03,
    "INT_ACC_Z": -0.02,
    "INT_ACC_Y": 0.03,
    "INF_STR": -0.04,
    "SUP_STR": 0.04,
    "SHE_STR": 0.0,
}
GLYPH_BY_MEASUREMENT = {
    "MID_ACC_Z": "vertical-arrow",
    "INT_ACC_Z": "vertical-arrow",
    "MID_ACC_Y": "transverse-arrow",
    "INT_ACC_Y": "transverse-arrow",
    "INF_STR": "lower-strain",
    "SUP_STR": "upper-strain",
    "SHE_STR": "shear-strain",
}


def parse_sensor_name(sensor_name: str) -> dict[str, Any]:
    """Parse an AQUINAS sensor name into structured layout metadata."""
    parts = sensor_name.split("_")
    if len(parts) not in {5, 6}:
        raise ValueError(f"Unsupported AQUINAS sensor name: {sensor_name}")

    deck, span, side = parts[:3]
    suffix = parts[3:]

    # AQUINAS names encode either a positioned accelerometer
    # (`..._MID_ACC_Z`) or a strain gauge (`..._INF_STR` / `..._SHE_STR`).
    if len(suffix) == 3 and suffix[1] == "ACC":
        position = suffix[0]
        axis_or_fibre = suffix[2]
        section = position
        measurement_family = "ACC"
        measurement_code = f"{position}_ACC_{axis_or_fibre}"
    elif len(suffix) == 2 and suffix[1] == "STR":
        fibre = suffix[0]
        measurement_family = "STR"
        axis_or_fibre = fibre
        section = "MID" if fibre in {"INF", "SUP"} else "SHE"
        measurement_code = f"{fibre}_STR"
    else:
        raise ValueError(f"Unsupported AQUINAS sensor name: {sensor_name}")

    # The released dataset only includes transversal acceleration on the
    # downstream side. Keeping this rule here prevents invalid synthetic
    # layouts from leaking into the exported schema.
    if measurement_family == "ACC" and axis_or_fibre == "Y" and side != "DO":
        raise ValueError(
            f"Released AQUINAS layout does not include ACC_Y sensors on side {side}: {sensor_name}"
        )

    # Comparison groups intentionally ignore deck so homologous pairs can
    # be matched by swapping `OLD` <-> `NEW`.
    comparison_group_id = f"{span}_{side}_{measurement_code}"
    homologous_sensor_id = f"{'NEW' if deck == 'OLD' else 'OLD'}_{comparison_group_id}"

    return {
        "sensor_id": sensor_name,
        "deck": deck,
        "span": span,
        "side": side,
        "section": section,
        "section_key": f"{span}_{section}",
        "measurement_family": measurement_family,
        "measurement_code": measurement_code,
        "axis_or_fibre": axis_or_fibre,
        "glyph_type": GLYPH_BY_MEASUREMENT[measurement_code],
        "comparison_group_id": comparison_group_id,
        "homologous_sensor_id": homologous_sensor_id,
    }


def build_sensor_layout(sensor_names: list[str]) -> list[dict[str, Any]]:
    """Build normalized analytical 3D coordinates for the AQUINAS sensors."""
    known_sensors = set(sensor_names)
    layout_rows: list[dict[str, Any]] = []

    for sensor_name in sorted(sensor_names):
        parsed = parse_sensor_name(sensor_name)
        measurement_code = parsed["measurement_code"]

        # The viewer supports both compact and exploded deck spacing, so
        # both positions are exported up front instead of recomputed later.
        compact_z = DECK_Z_CENTERS["compact"][parsed["deck"]] + SIDE_Z_OFFSET[parsed["side"]]
        exploded_z = DECK_Z_CENTERS["exploded"][parsed["deck"]] + SIDE_Z_OFFSET[parsed["side"]]

        # Small x-offsets keep co-located markers legible without changing
        # their structural section assignment from the handbook topology.
        x = SECTION_X[parsed["section_key"]] + X_NUDGE[measurement_code]
        y = HEIGHT_BY_GLYPH[parsed["glyph_type"]]

        layout_rows.append(
            {
                **parsed,
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(exploded_z, 4),
                "compact_z": round(compact_z, 4),
                "exploded_z": round(exploded_z, 4),
                "homologous_sensor_id": (
                    parsed["homologous_sensor_id"]
                    if parsed["homologous_sensor_id"] in known_sensors
                    else None
                ),
            }
        )

    return layout_rows


def build_bridge_geometry() -> dict[str, Any]:
    """Return the normalized analytical bridge geometry used by the viewer."""
    deck_meshes = []
    for deck in ("OLD", "NEW"):
        segments = []
        for span, (x_start, x_end) in SPAN_BOUNDARIES.items():
            # Each segment is a simple box-girder volume. The viewer turns
            # this into an SVG projection rather than a true mesh render.
            # Span extents are normalized for comparison; they are not meant
            # to represent exact bridge dimensions in meters.
            segments.append(
                {
                    "deck": deck,
                    "span": span,
                    "x_start": x_start,
                    "x_end": x_end,
                    "y_center": 0.0,
                    "depth": 0.48,
                    "width": 0.58,
                    "compact_center_z": DECK_Z_CENTERS["compact"][deck],
                    "exploded_center_z": DECK_Z_CENTERS["exploded"][deck],
                }
            )
        deck_meshes.append({"deck": deck, "segments": segments})

    label_anchors = [
        {"label": "OLD deck", "x": 0.10, "y": -0.55, "compact_z": 1.10, "exploded_z": 1.90},
        {"label": "NEW deck", "x": 0.10, "y": -0.55, "compact_z": -1.10, "exploded_z": -1.90},
        {"label": "Span 1", "x": 0.50, "y": 0.65, "compact_z": 0.0, "exploded_z": 0.0},
        {"label": "Span 2", "x": 1.50, "y": 0.65, "compact_z": 0.0, "exploded_z": 0.0},
    ]

    section_anchors = [
        {"section_key": key, "x": x, "label": key.replace("_", " "), "y": 0.52}
        for key, x in SECTION_X.items()
    ]

    return {
        "coordinate_system": {
            "x": "span progression",
            "y": "vertical elevation",
            "z": "upstream/downstream transverse axis",
        },
        "view_modes": {
            "compact": {"deck_centers": DECK_Z_CENTERS["compact"]},
            "exploded": {"deck_centers": DECK_Z_CENTERS["exploded"]},
        },
        "span_boundaries": [
            {"span": span, "x_start": bounds[0], "x_end": bounds[1]}
            for span, bounds in SPAN_BOUNDARIES.items()
        ],
        "pier_anchors": [
            {"pier_id": "abutment_1", "x": 0.0},
            {"pier_id": "pier_1", "x": 1.0},
            {"pier_id": "abutment_2", "x": 2.0},
        ],
        "deck_meshes": deck_meshes,
        "label_anchors": label_anchors,
        "section_anchors": section_anchors,
    }
