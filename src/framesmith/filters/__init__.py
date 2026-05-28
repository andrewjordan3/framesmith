# framesmith/filters/__init__.py
"""Row-filter expression builders.

Functions in this package return ``pl.Expr`` boolean masks intended
for ``df.filter(...)``. They never mutate frames themselves — the
user applies the mask.

Public surface is exported from this init; the internal file
structure (``dates.py``, etc.) is private. Import from
``framesmith.filters``.
"""

from framesmith.filters.dates import (
    within_complete_month,
    within_complete_period,
)

__all__: list[str] = [
    'within_complete_month',
    'within_complete_period',
]
