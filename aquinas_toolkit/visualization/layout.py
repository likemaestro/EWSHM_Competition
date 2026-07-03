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

from math import sqrt
from typing import Any

SPAN_LENGTH_M = 45.0


def _norm(meters: float) -> float:
    """Convert a physical dimension in meters to normalized span units."""
    return meters / SPAN_LENGTH_M


# These coordinates describe a normalized analytical schematic, not a
# surveyed bridge model. The viewer and exporter both rely on this same
# coordinate system, so changes here should be treated as schema-level.
#
# Provenance:
# - `S1_MID`, `S1_INT`, `S2_INT`, and `S2_MID` follow the handbook
#   section definitions directly.
# - `S1_SHE` and `S2_SHE` are intentional approximations because the
#   handbook only states that the shear sensors are located "near pier 1".
# - The shared cross-section below is a refined analytical box-girder
#   derived from the handbook-scale interpretation used by the viewer:
#   45 m span, 2.0 m depth, and a *trapezoidal* box that is wider at the
#   top (where the webs meet the top slab) and narrower at the bottom
#   slab, matching the structural drawings. Field names below are stated
#   explicitly as "top" / "bottom" of the box so the viewer render and the
#   sensor anchors stay in agreement.
GIRDER_DEPTH = _norm(2.0)
TOP_SLAB_WIDTH = _norm(7.5)
WEB_OUTER_TOP_WIDTH = _norm(5.0)      # box outer width where webs meet the top slab
WEB_OUTER_BOTTOM_WIDTH = _norm(3.2)   # box outer width at the bottom slab (narrower)
BOTTOM_SLAB_WIDTH = WEB_OUTER_BOTTOM_WIDTH
SLAB_THICKNESS = _norm(0.30)
WEB_THICKNESS = _norm(0.35)
WEB_INNER_TOP_WIDTH = WEB_OUTER_TOP_WIDTH - (2 * WEB_THICKNESS)
WEB_INNER_BOTTOM_WIDTH = WEB_OUTER_BOTTOM_WIDTH - (2 * WEB_THICKNESS)
OVERHANG_WIDTH = (TOP_SLAB_WIDTH - WEB_OUTER_TOP_WIDTH) / 2
SENSOR_READABILITY_OFFSET = _norm(0.12)

HALF_DEPTH = GIRDER_DEPTH / 2
HALF_TOP_SLAB = TOP_SLAB_WIDTH / 2
HALF_BOTTOM_SLAB = BOTTOM_SLAB_WIDTH / 2
HALF_WEB_OUTER_TOP = WEB_OUTER_TOP_WIDTH / 2
HALF_WEB_OUTER_BOTTOM = WEB_OUTER_BOTTOM_WIDTH / 2
HALF_WEB_INNER_TOP = WEB_INNER_TOP_WIDTH / 2
HALF_WEB_INNER_BOTTOM = WEB_INNER_BOTTOM_WIDTH / 2
TOP_SLAB_UNDERSIDE_Y = HALF_DEPTH - SLAB_THICKNESS
BOTTOM_SLAB_TOP_Y = -HALF_DEPTH + SLAB_THICKNESS

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
    "compact": {"OLD": 0.14, "NEW": -0.14},
    "exploded": {"OLD": 0.22, "NEW": -0.22},
}
X_NUDGE = {
    "MID_ACC_Z": -0.008,
    "MID_ACC_Y": 0.008,
    "INT_ACC_Z": -0.008,
    "INT_ACC_Y": 0.008,
    "INF_STR": -0.012,
    "SUP_STR": 0.012,
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


def _normalize_vector(x: float, y: float, z: float) -> tuple[float, float, float]:
    magnitude = sqrt((x * x) + (y * y) + (z * z))
    if magnitude == 0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return (x / magnitude, y / magnitude, z / magnitude)


def _vector_dict(vector: tuple[float, float, float]) -> dict[str, float]:
    return {
        "x": round(vector[0], 4),
        "y": round(vector[1], 4),
        "z": round(vector[2], 4),
    }


def _point_dict(point: tuple[float, float, float]) -> dict[str, float]:
    return {
        "x": round(point[0], 4),
        "y": round(point[1], 4),
        "z": round(point[2], 4),
    }


def _side_sign(side: str) -> int:
    return 1 if side == "UP" else -1


def _web_outer_half_width_at(y: float) -> float:
    """Return the half-width of the outer web face at a given local Y.

    The web is wider at the top (HALF_WEB_OUTER_TOP) and narrower at the
    bottom (HALF_WEB_OUTER_BOTTOM), matching the trapezoidal box-girder
    profile shown in the structural drawings.
    """
    top_y = TOP_SLAB_UNDERSIDE_Y
    bottom_y = BOTTOM_SLAB_TOP_Y
    span = bottom_y - top_y
    if span == 0:
        return HALF_WEB_OUTER_BOTTOM
    t = (y - top_y) / span
    return HALF_WEB_OUTER_TOP + ((HALF_WEB_OUTER_BOTTOM - HALF_WEB_OUTER_TOP) * t)


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


def _sensor_mount(parsed: dict[str, Any], *, x: float) -> dict[str, Any]:
    """Return mount-aware local coordinates and orientation for one sensor."""
    side_sign = _side_sign(parsed["side"])
    measurement_code = parsed["measurement_code"]

    if measurement_code == "SUP_STR":
        # Placed on the exterior top surface of the top slab so it is
        # visible and raycaster-reachable from above.
        anchor = (
            x,
            HALF_DEPTH,
            side_sign * (HALF_WEB_INNER_TOP - _norm(0.15)),
        )
        normal = (0.0, 1.0, 0.0)
        orientation = (1.0, 0.0, 0.0)
        surface = "top_slab_exterior"
    elif measurement_code == "INF_STR":
        # Placed on the exterior bottom surface of the bottom slab so it
        # protrudes below the deck and can be clicked.
        anchor = (
            x,
            -HALF_DEPTH,
            side_sign * (HALF_WEB_INNER_BOTTOM - _norm(0.12)),
        )
        normal = (0.0, -1.0, 0.0)
        orientation = (1.0, 0.0, 0.0)
        surface = "bottom_slab_exterior"
    elif measurement_code.endswith("ACC_Z"):
        # Placed on the outer z-face of the bottom slab, anchored at the
        # exterior bottom face so the arrow sits flush at the deck edge.
        anchor = (
            x,
            -HALF_DEPTH,
            side_sign * HALF_BOTTOM_SLAB,
        )
        normal = (0.0, 0.0, float(side_sign))
        orientation = (0.0, 1.0, 0.0)
        surface = "web_outer_face"
    elif measurement_code.endswith("ACC_Y"):
        anchor = (
            x,
            -HALF_DEPTH,
            side_sign * HALF_BOTTOM_SLAB,
        )
        normal = (0.0, 0.0, float(side_sign))
        orientation = (0.0, 0.0, -side_sign)
        surface = "web_outer_face"
    elif measurement_code == "SHE_STR":
        anchor_y = 0.0
        anchor = (
            x,
            anchor_y,
            side_sign * _web_outer_half_width_at(anchor_y),
        )
        normal = _normalize_vector(0.0, 0.25, float(side_sign))
        orientation = _normalize_vector(1.0 if parsed["span"] == "S1" else -1.0, 1.0, 0.0)
        surface = "web_outer_face"
    else:  # pragma: no cover - guarded by parse_sensor_name / glyph map
        raise ValueError(f"Unsupported measurement code: {measurement_code}")

    final_position = (
        anchor[0] + (normal[0] * SENSOR_READABILITY_OFFSET),
        anchor[1] + (normal[1] * SENSOR_READABILITY_OFFSET),
        anchor[2] + (normal[2] * SENSOR_READABILITY_OFFSET),
    )

    return {
        "anchor": anchor,
        "normal": normal,
        "orientation": orientation,
        "final_position": final_position,
        "mount_surface": surface,
    }


def build_sensor_layout(sensor_names: list[str]) -> list[dict[str, Any]]:
    """Build normalized analytical 3D coordinates for the AQUINAS sensors."""
    known_sensors = set(sensor_names)
    layout_rows: list[dict[str, Any]] = []

    for sensor_name in sorted(sensor_names):
        parsed = parse_sensor_name(sensor_name)
        measurement_code = parsed["measurement_code"]

        x = SECTION_X[parsed["section_key"]] + X_NUDGE[measurement_code]
        mount = _sensor_mount(parsed, x=x)
        local_position = mount["final_position"]

        compact_z = DECK_Z_CENTERS["compact"][parsed["deck"]] + local_position[2]
        exploded_z = DECK_Z_CENTERS["exploded"][parsed["deck"]] + local_position[2]

        layout_rows.append(
            {
                **parsed,
                "x": round(local_position[0], 4),
                "y": round(local_position[1], 4),
                "z": round(exploded_z, 4),
                "local_position": _point_dict(local_position),
                "anchor_local": _point_dict(mount["anchor"]),
                "surface_normal": _vector_dict(mount["normal"]),
                "glyph_orientation": _vector_dict(mount["orientation"]),
                "mount_surface": mount["mount_surface"],
                "readability_offset": round(SENSOR_READABILITY_OFFSET, 4),
                "local_z": round(local_position[2], 4),
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
            segments.append(
                {
                    "deck": deck,
                    "span": span,
                    "x_start": x_start,
                    "x_end": x_end,
                    "compact_center_z": DECK_Z_CENTERS["compact"][deck],
                    "exploded_center_z": DECK_Z_CENTERS["exploded"][deck],
                }
            )
        deck_meshes.append({"deck": deck, "segments": segments})

    label_anchors = [
        {
            "label": "OLD deck",
            "x": 0.12,
            "y": _norm(2.8),
            "compact_z": DECK_Z_CENTERS["compact"]["OLD"],
            "exploded_z": DECK_Z_CENTERS["exploded"]["OLD"],
        },
        {
            "label": "NEW deck",
            "x": 0.12,
            "y": _norm(2.8),
            "compact_z": DECK_Z_CENTERS["compact"]["NEW"],
            "exploded_z": DECK_Z_CENTERS["exploded"]["NEW"],
        },
        {"label": "Span 1", "x": 0.50, "y": _norm(3.4), "compact_z": 0.0, "exploded_z": 0.0},
        {"label": "Span 2", "x": 1.50, "y": _norm(3.4), "compact_z": 0.0, "exploded_z": 0.0},
    ]

    section_anchors = [
        {"section_key": key, "x": x, "label": key.replace("_", " "), "y": _norm(2.6)}
        for key, x in SECTION_X.items()
    ]

    return {
        "coordinate_system": {
            "x": "span progression",
            "y": "vertical elevation",
            "z": "upstream/downstream transverse axis",
        },
        "world": {
            "meters_per_normalized_unit": SPAN_LENGTH_M,
            "bridge_length_m": SPAN_LENGTH_M * 2,
        },
        "cross_section": {
            "depth": GIRDER_DEPTH,
            "top_slab_width": TOP_SLAB_WIDTH,
            "bottom_slab_width": BOTTOM_SLAB_WIDTH,
            "slab_thickness": SLAB_THICKNESS,
            "web_thickness": WEB_THICKNESS,
            # Box outer/inner widths stated explicitly at the top and bottom
            # of the web so the trapezoid tapers inward going down.
            "web_outer_top_width": WEB_OUTER_TOP_WIDTH,
            "web_outer_bottom_width": WEB_OUTER_BOTTOM_WIDTH,
            "web_inner_top_width": WEB_INNER_TOP_WIDTH,
            "web_inner_bottom_width": WEB_INNER_BOTTOM_WIDTH,
            "overhang_width": OVERHANG_WIDTH,
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
