# framesmith/group/__init__.py
"""Group-resolution and group-fill operations.

Resolvers collapse duplicate-key rows via ``df.group_by().agg``; fillers fill
nulls in place via ``df.with_columns``. Public surface is exported here;
internal file layout (``resolve.py``, ``fill.py``, ``mode.py``) is private —
import from ``framesmith.group``.
"""

from framesmith.group.fill import (
    CentralTendencyStrategy,
    fill_null_numeric_by_group,
    fill_null_string_by_group,
)
from framesmith.group.resolve import (
    first_non_null_per_group,
    mode_then_first_per_group,
)

__all__: list[str] = [
    'CentralTendencyStrategy',
    'fill_null_numeric_by_group',
    'fill_null_string_by_group',
    'first_non_null_per_group',
    'mode_then_first_per_group',
]
