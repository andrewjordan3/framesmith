# framesmith/transforms/split.py
"""
String-splitting transforms.

``split_on_delimiters`` splits a string column into a ``List(String)`` on any
of a set of single-character delimiters, trimming each token and preserving
empty fields as explicit nulls so positional information survives. It is the
inverse direction of ``framesmith.combine.combine_columns`` (which joins).
"""

import re
from collections.abc import Sequence

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'DEFAULT_SPLIT_DELIMITERS',
    'split_on_delimiters',
]


# Default split delimiters: comma, semicolon, pipe, slash, hyphen. Each is a
# single character (the mechanic builds a regex character class, which matches
# one character per position). The hyphen is included deliberately — many
# multi-value fields use it as a separator.
DEFAULT_SPLIT_DELIMITERS: tuple[str, ...] = (',', ';', '|', '/', '-')

# Sentinel placed at each split point: the ASCII Unit Separator control
# character (U+001F), chosen because it effectively never appears in tabular
# string data. polars ``str.split`` is literal (not regex), so the several
# delimiter characters are first rewritten to this single sentinel and the
# value is then split on it. A value that genuinely contains U+001F will
# mis-split — a documented limitation, since detecting it would require
# scanning the data and break the pure-expression model.
_SPLIT_SENTINEL: str = '\x1f'


def split_on_delimiters(
    delimiters: Sequence[str] = DEFAULT_SPLIT_DELIMITERS,
    *,
    dedup_delimiters: bool = False,
) -> ExpressionTransform:
    """Split a string column into a ``List(String)`` on any of several delimiters.

    Each value is split wherever any character in ``delimiters`` occurs.
    Every resulting token is trimmed of surrounding whitespace; a token
    that is empty or whitespace-only becomes an explicit ``null`` rather
    than being dropped, so positional information is preserved
    (``"a,, b ,"`` → ``["a", null, "b", null]``). A genuine null input
    passes through as a null list; an empty or whitespace-only input
    becomes a single-element ``[null]`` (so null and empty string are
    distinguished).

    By default consecutive delimiters are **not** collapsed: ``"a,,b"``
    yields ``["a", null, "b"]`` because the empty middle field is
    meaningful (a missing value, not noise). Set
    ``dedup_delimiters=True`` to collapse runs of *adjacent* delimiter
    characters, so ``"a,,b"`` yields ``["a", "b"]``. Whitespace between
    delimiters breaks a run even when dedup is on, so ``"a, ,b"`` still
    yields ``["a", null, "b"]`` — the space-separated gap is a real empty
    field.

    Matching is on single characters via a regex character class.
    Internally the delimiters are rewritten to a single control-character
    sentinel and the value is split on that, because polars'
    ``str.split`` takes a literal delimiter, not a regex.

    Args:
        delimiters: Single-character delimiters to split on. Each must be
            exactly one character (the character-class mechanic does not
            support multi-character delimiters). Must be non-empty.
            Defaults to ``(',', ';', '|', '/', '-')``.
        dedup_delimiters: When ``True``, runs of adjacent delimiter
            characters collapse to a single split point. When ``False``
            (default), each delimiter produces a split, preserving empty
            fields as nulls.

    Returns:
        An ``ExpressionTransform`` (producing ``List(String)``) applied
        via ``compose_column``.

    Raises:
        ValueError: If ``delimiters`` is empty, or if any delimiter is not
            exactly one character.

    Example:
        >>> import polars as pl
        >>> import framesmith as fs
        >>> from framesmith.transforms import split_on_delimiters
        >>> df = pl.DataFrame({'tags': ['a,, b ,']})
        >>> df.with_columns(
        ...     fs.compose_column('tags', [split_on_delimiters()])
        ... )['tags'].to_list()
        [['a', None, 'b', None]]
    """
    if len(delimiters) == 0:
        raise ValueError('delimiters must not be empty')
    multi_char_delimiters: list[str] = [
        delimiter for delimiter in delimiters if len(delimiter) != 1
    ]
    if multi_char_delimiters:
        raise ValueError(
            'each delimiter must be exactly one character; got '
            f'multi-character or empty delimiters: {multi_char_delimiters!r}'
        )

    # Build a regex character class from the delimiters. re.escape makes each
    # character safe inside a class (e.g. '-' -> r'\-', '|' -> r'\|'), which
    # the polars Rust regex engine accepts. The '+' quantifier in dedup mode
    # collapses runs of adjacent delimiters into a single split point.
    quantifier: str = '+' if dedup_delimiters else ''
    delimiter_class: str = (
        '['
        + ''.join(re.escape(character) for character in delimiters)
        + ']'
        + quantifier
    )

    def _split_on_delimiters(expr: pl.Expr) -> pl.Expr:
        return (
            expr.str.replace_all(delimiter_class, _SPLIT_SENTINEL)
            .str.split(_SPLIT_SENTINEL)
            .list.eval(
                # Trim each token; an empty or whitespace-only token
                # becomes null, preserving the position of a missing field.
                pl.when(pl.element().str.strip_chars().str.len_chars() == 0)
                .then(None)
                .otherwise(pl.element().str.strip_chars())
            )
        )

    return _split_on_delimiters
