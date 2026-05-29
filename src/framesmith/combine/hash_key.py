# framesmith/combine/hash_key.py
"""
Deterministic surrogate keys via struct hashing.

``hash_key`` builds a ``pl.Expr`` that hashes one or more columns into a single
``UInt64`` key, applied via ``df.with_columns(...)``. The key is a pure
function of the input values, so the same values produce the same key
regardless of what else is in the frame — keys computed on separate frames in
the same run line up for joins without coordinating a shared key space.
"""

from collections.abc import Sequence

import polars as pl

__all__: list[str] = ['hash_key']


def hash_key(
    column_names: Sequence[str],
    output_column_name: str,
) -> pl.Expr:
    """Build a deterministic ``UInt64`` surrogate key from one or more columns.

    Hashes a struct of the named columns, so each distinct combination of
    values maps to a stable key. The key depends only on the values, not
    on the frame: the same combination produces the same key in any frame
    within a run, which is what lets independently-keyed frames join.
    Apply via ``df.with_columns(hash_key(...))``.

    Determinism scope: keys are reproducible within a single polars
    version. The hash algorithm may change across polars releases, so do
    not persist these keys and expect them to match after an upgrade —
    they are for within-run / within-version use.

    Collision risk (read this): the key is 64-bit, so distinct inputs can
    in principle hash to the same key, silently merging two entities in
    any join — wrong results, no error. Probability scales with the number
    of distinct key values (not rows): negligible through the low millions
    (~1 in tens of millions of distinct values), about 0.03% at 100
    million distinct, and about 2.7% at 1 billion distinct. Use where that
    risk is acceptable. To assert it did not happen for a given frame,
    check that the key's distinct count equals the struct's distinct count
    (``df.select(pl.struct(cols).n_unique(), key.n_unique())`` — equal
    means no collision occurred).

    Nulls are hashed, not propagated: a null value (or a null in any
    struct field) produces a stable key like any other value, and all
    nulls map to the same key. This differs from ``combine_columns`` /
    ``coalesce_blank_columns``, which preserve nulls.

    Args:
        column_names: Columns to hash, in order. One or more; a single
            name produces a key for that column's values. Must be
            non-empty.
        output_column_name: Name of the resulting key column. Required —
            no default. If it matches an existing column, ``with_columns``
            overwrites it, per polars semantics.

    Returns:
        A ``UInt64`` ``pl.Expr`` for ``df.with_columns(...)``.

    Raises:
        ValueError: If ``column_names`` is empty.

    Note:
        Column existence is not checked here — the builder has no frame. A
        missing column raises ``polars.exceptions.ColumnNotFoundError`` (or
        the polars struct-construction error) when the expression is
        applied.

    Example:
        >>> import polars as pl
        >>> from framesmith.combine import hash_key
        >>> df = pl.DataFrame(
        ...     {'region': ['W', 'W', 'E'], 'branch': ['a', 'b', 'a']}
        ... )
        >>> keyed = df.with_columns(hash_key(['region', 'branch'], 'k'))
        >>> keyed['k'].dtype
        UInt64
    """
    if len(column_names) == 0:
        raise ValueError('column_names must not be empty')

    return pl.struct(list(column_names)).hash(seed=0).alias(output_column_name)
