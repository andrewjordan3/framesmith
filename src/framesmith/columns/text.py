# framesmith/columns/text.py
"""
Polars expression builders for text normalization.

These helpers return ``pl.Expr`` objects intended for use inside
``DataFrame.with_columns(...)`` (or ``LazyFrame.with_columns(...)``).
They compose: each is a single, narrowly-scoped transformation, so
callers chain them — or chain other ``str`` methods on the returned
expression — to assemble higher-level pipelines.

All builders auto-alias their output back to the input column name so
that the common case (``df.with_columns(normalize_text('name'))``)
overwrites the column in place without an explicit ``.alias(...)``.
"""

import polars as pl

from framesmith._internal.regex_patterns import (
    BLANK_OR_WHITESPACE_ONLY_PATTERN,
    WHITESPACE_RUN_PATTERN,
)
from framesmith._internal.unicode_maps import ASCII_COMPAT_MAP

__all__: list[str] = [
    'normalize_text',
    'to_snake_case',
]


def normalize_text(column: str) -> pl.Expr:
    """Multi-stage text normalization, returns a ``pl.Expr``.

    Stages, applied in order:
        1. Blank or whitespace-only strings are coerced to null.
        2. NFKC Unicode normalization.
        3. ASCII compatibility folding via ``ASCII_COMPAT_MAP``
           (smart quotes, em-dashes, currency symbols, trademark and
           registered symbols, non-standard whitespace, etc.).
        4. Whitespace-run collapse to a single space, then strip.
        5. Literal ``&`` is replaced with ``and``.
        6. Apostrophes (``'``) and periods (``.``) are removed.

    Args:
        column: Name of the input column to transform.

    Returns:
        ``pl.Expr`` that, when applied via ``with_columns``, produces
        the normalized text. The output is auto-aliased to ``column``.
        Nulls propagate: null in → null out. Blank or whitespace-only
        strings are converted to null before the normalization stages
        run, so they also come out as null.

    Note:
        Case is preserved. For case-insensitive matching, chain
        ``.str.to_lowercase()`` on the returned expression — lowercasing
        is a separate concern (matching vs. preservation) and is left
        to the caller.
    """
    blank_to_null: pl.Expr = (
        pl.when(pl.col(column).str.contains(BLANK_OR_WHITESPACE_ONLY_PATTERN))
        .then(None)
        .otherwise(pl.col(column))
    )
    return (
        blank_to_null.str.normalize('NFKC')
        .str.replace_many(ASCII_COMPAT_MAP)
        .str.replace_all(WHITESPACE_RUN_PATTERN, ' ')
        .str.strip_chars()
        .str.replace_all('&', 'and', literal=True)
        .str.replace_all("'", '', literal=True)
        .str.replace_all('.', '', literal=True)
        .alias(column)
    )


def to_snake_case(column: str) -> pl.Expr:
    """Replace whitespace runs with underscores.

    Assumes the input is already normalized (e.g. via
    :func:`normalize_text`): this builder does not strip, lowercase,
    fold Unicode, or collapse anything other than whitespace runs.

    Args:
        column: Name of the input column to transform.

    Returns:
        ``pl.Expr`` auto-aliased to ``column``. Nulls propagate
        unchanged.
    """
    return (
        pl.col(column)
        .str.replace_all(WHITESPACE_RUN_PATTERN, '_')
        .alias(column)
    )
