# framesmith/validate/__init__.py
"""Column validation guards.

Each guard inspects a column and either returns ``None`` (the data satisfies
the constraint) or raises ``ValueError`` describing the violation. Guards do
not transform the frame — they are called for their loud-failure side effect,
to fail a pipeline early on bad data rather than letting it flow downstream.
Public surface is exported here; internal file layout is private.
"""

from framesmith.validate.length import assert_string_length
from framesmith.validate.nulls import assert_no_nulls

__all__: list[str] = [
    'assert_no_nulls',
    'assert_string_length',
]
