"""Column schema for the System Setup tree view."""
from __future__ import annotations

from enum import IntEnum


class Column(IntEnum):
    NAME = 0
    VALUE = 1


HEADERS = {
    Column.NAME: "Name",
    Column.VALUE: "Value",
}


def column_count() -> int:
    return len(Column)


def header_text(section: int) -> str:
    try:
        return HEADERS[Column(section)]
    except ValueError:
        return ""
