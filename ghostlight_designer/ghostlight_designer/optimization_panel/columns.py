"""Column schema for the optimization panel tree.

Same canonicalisation idea as the optical editor: a tuple of (col, label)
plus a header lookup. The model exposes the canonical width and the
trailing slot count is computed dynamically per goal kind (one trailing
column per :class:`ParamDef`).
"""
from __future__ import annotations

from enum import IntEnum


class Column(IntEnum):
    NAME = 0          # MF name / Goal kind + comment
    TYPE = 1          # MF status / Goal kind label
    TARGET = 2        # Goal target (blank on MF row)
    WEIGHT = 3        # Goal weight
    VALUE = 4         # MF last_total / Goal cached_value
    RESIDUAL = 5      # Goal cached_residual


HEADERS: dict[Column, str] = {
    Column.NAME: "Name",
    Column.TYPE: "Type",
    Column.TARGET: "Target",
    Column.WEIGHT: "Weight",
    Column.VALUE: "Value",
    Column.RESIDUAL: "Residual",
}


# Width of trailing "param" columns. We don't know per-goal-kind param
# counts at module-load time, so the model widens itself dynamically; this
# is just the default reservation for the param-strip header.
PARAM_COLUMN_OFFSET = len(Column)


def base_column_count() -> int:
    return len(Column)


def header_text(section: int) -> str:
    try:
        return HEADERS[Column(section)]
    except ValueError:
        return ""
