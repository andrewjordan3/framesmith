# framesmith/transforms/missing.py
"""
Missing-value sentinel handling.

Many sources encode missing data as placeholder strings (``'N/A'``,
``'NULL'``, ``'NONE'``, etc.) rather than true nulls.
``nullify_sentinels`` is a transform factory: give it the set of
strings that mean "missing" for your source, and it returns a transform
that nulls matching cells.

Opt-in by design. This is NOT part of any default recipe — whether a
given token means "missing" depends entirely on the data, and
defaulting it on risks silently nulling valid values (e.g. ``'NA'`` as
Namibia). Add it to your own recipe explicitly::

    recipe = (*NORMALIZE_TEXT, nullify_sentinels(DEFAULT_MISSING_SENTINELS))
"""

from collections.abc import Collection

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'DEFAULT_MISSING_SENTINELS',
    'nullify_sentinels',
]


# Conservative default set, roughly aligned with the common pandas
# ``read_csv`` ``na_values`` convention. Stored uppercase because the
# factory's default case-insensitive comparison uppercases both sides.
#
# Deliberately EXCLUDES ``'INF'`` / ``'INFINITY'`` and other numeric
# special forms: those are float concerns surfaced by casting, not
# string-missing markers, and a text column may legitimately contain
# ``'inf'``.
#
# ``'NA'`` is included to match the dominant convention, but it is the
# most ambiguous entry — sources where ``'NA'`` is valid data (Namibia,
# initials, "North America") should pass a custom set instead.
DEFAULT_MISSING_SENTINELS: frozenset[str] = frozenset(
    {
        '',
        'N/A',
        'NA',
        'NAN',
        'NULL',
        'NONE',
    }
)


def nullify_sentinels(
    sentinels: Collection[str],
    *,
    case_insensitive: bool = True,
) -> ExpressionTransform:
    """Build a transform that nulls cells matching any missing sentinel.

    A cell matches if its whitespace-stripped form equals a sentinel
    (case-insensitively by default). Matching is exact after stripping
    — ``'NABISCO'`` does not match ``'NA'``. Non-matching cells pass
    through completely unchanged, including their original case and
    surrounding whitespace; the strip and uppercase are applied only
    for the comparison, never to the output.

    Typically applied early, against raw values, before other
    normalization has a chance to alter a sentinel's spelling.

    Args:
        sentinels: The strings that count as missing for this source.
            Must be non-empty.
        case_insensitive: If True (default), match without regard to
            case. Both the input cell and each sentinel are uppercased
            for the comparison.

    Returns:
        An ``ExpressionTransform`` that nulls matching cells and
        leaves everything else untouched. Nulls pass through as null.

    Raises:
        ValueError: If ``sentinels`` is empty.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> df = pl.DataFrame({'x': ['ok', 'N/A', 'null', 'Chicago']})
        >>> df.with_columns(
        ...     fs.compose_column(
        ...         'x',
        ...         [fs.nullify_sentinels(fs.DEFAULT_MISSING_SENTINELS)],
        ...     )
        ... )['x'].to_list()
        ['ok', None, None, 'Chicago']
    """
    if len(sentinels) == 0:
        raise ValueError('sentinels must not be empty')

    # Build the comparison set once, in the factory body. The returned
    # closure must not rebuild or re-normalize this on every call.
    comparison_set: frozenset[str] = (
        frozenset(sentinel.strip().upper() for sentinel in sentinels)
        if case_insensitive
        else frozenset(sentinel.strip() for sentinel in sentinels)
    )

    def _nullify_sentinels(expr: pl.Expr) -> pl.Expr:
        normalized: pl.Expr = expr.str.strip_chars()
        if case_insensitive:
            normalized = normalized.str.to_uppercase()
        return (
            pl.when(normalized.is_in(comparison_set))
            .then(None)
            .otherwise(expr)
        )

    return _nullify_sentinels
