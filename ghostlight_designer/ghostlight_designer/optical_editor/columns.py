"""Column indices by name, for callers that address columns as
``Column.RADIUS`` rather than by position.

The authority is ``row_schemas.CANONICAL_COLUMNS``, which the model and
delegate dispatch through; this enum mirrors those positions.
"""
from __future__ import annotations

import enum


class Column(enum.IntEnum):
    NAME = 0
    IDENTIFIER = 1
    RADIUS = 2
    POS_Z = 3
    APERTURE_RAD = 4
    # Off-axis element placement. OFF_AXIS holds the ">>>" reveal toggle and
    # is always visible; the rest are hidden until some element is revealed.
    OFF_AXIS = 5
    POS_X = 6
    POS_Y = 7
    ROT_X = 8
    ROT_Y = 9
    ROT_Z = 10
    PIVOT_X = 11
    PIVOT_Y = 12
    PIVOT_Z = 13
