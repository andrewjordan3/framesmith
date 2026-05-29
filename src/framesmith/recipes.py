# framesmith/recipes.py
"""
Curated recipes: ordered tuples of transforms for common pipelines.

A recipe is plain data — a ``tuple[ExpressionTransform, ...]`` — that
callers pass to ``compose_column``. Splice and extend with tuple
unpacking: ``(*NORMALIZE_TEXT, to_snake_case)``. Recipes may compose
other recipes the same way (see ``NORMALIZE_TEXT`` below).

UPPERCASE naming signals "reusable predefined sequence," distinct from
the lowercase transform functions they contain.
"""

from framesmith.transforms import (
    accounting_parens_to_negative,
    cast_to_float64,
    collapse_whitespace,
    extract_email_local_part,
    fold_to_ascii,
    normalize_unicode_nfkc,
    nullify_blank_strings,
    percent_to_fraction,
    periods_to_spaces,
    remove_apostrophes,
    remove_periods,
    remove_thousands_separators,
    replace_ampersand_with_and,
    strip_whitespace,
    to_titlecase,
    trailing_minus_to_prefix,
    underscores_to_spaces,
)
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'CLEAN_NUMERIC_STRING',
    'EMAIL_TO_DISPLAY_NAME',
    'NORMALIZE_NUMERIC',
    'NORMALIZE_PERCENT',
    'NORMALIZE_TEXT',
    'SNAKE_TO_TITLE',
    'UNICODE_TO_ASCII',
]


# NFKC normalization followed by ASCII compatibility folding. Order
# matters: NFKC decomposes compatibility forms before the ASCII map
# runs. Use this when you want canonical ASCII-ish text; use the
# individual transforms when you want only one half.
UNICODE_TO_ASCII: tuple[ExpressionTransform, ...] = (
    normalize_unicode_nfkc,
    fold_to_ascii,
)


# Reproduces the legacy ``normalize_text`` pipeline exactly, in order.
# Splices UNICODE_TO_ASCII so the Unicode-canonicalization order is not
# duplicated here.
NORMALIZE_TEXT: tuple[ExpressionTransform, ...] = (
    nullify_blank_strings,
    *UNICODE_TO_ASCII,
    collapse_whitespace,
    strip_whitespace,
    replace_ampersand_with_and,
    remove_apostrophes,
    remove_periods,
)


# Clean a messy numeric/currency string into a bare numeric string
# ready to cast. Reuses UNICODE_TO_ASCII for NFKC + minus/currency/
# invisible/whitespace-variant handling, then applies the numeric-
# specific rewrites. Returns a STRING — cast it yourself if you want a
# dtype other than Float64 (e.g.
# ``compose_column(col, CLEAN_NUMERIC_STRING).cast(pl.Int64, strict=False)``).
CLEAN_NUMERIC_STRING: tuple[ExpressionTransform, ...] = (
    *UNICODE_TO_ASCII,
    accounting_parens_to_negative,
    trailing_minus_to_prefix,
    remove_thousands_separators,
)


# Full numeric normalization: clean the string, then cast to Float64.
# Unparseable values become null (no fill — caller decides). Splices
# CLEAN_NUMERIC_STRING so the cleaning order has one source of truth.
NORMALIZE_NUMERIC: tuple[ExpressionTransform, ...] = (
    *CLEAN_NUMERIC_STRING,
    cast_to_float64,
)


# Parse a possibly-messy percent string into a Float64 fraction.
# Cleans the string via CLEAN_NUMERIC_STRING (handles unicode minus,
# accounting parens, trailing minus, commas/whitespace), then strips
# the optional '%' and casts. '50%' → 0.5; '(12%)' → -0.12.
NORMALIZE_PERCENT: tuple[ExpressionTransform, ...] = (
    *CLEAN_NUMERIC_STRING,
    percent_to_fraction,
)


# Extract a human-readable display name from an email address.
# Strips surrounding whitespace, takes the local part (before the first
# '@'), replaces periods with spaces, and collapses any resulting
# whitespace runs so doubled dots don't produce doubled spaces.
# 'john.doe@example.com' → 'john doe'.
EMAIL_TO_DISPLAY_NAME: tuple[ExpressionTransform, ...] = (
    strip_whitespace,
    extract_email_local_part,
    periods_to_spaces,
    collapse_whitespace,
)


# snake_case identifier → human Title Case label. Underscores become
# spaces, then each word is title-cased. Title casing mangles acronyms
# ('primary_lob' → 'Primary Lob'); splice apply_replacements to fix
# specific tokens, e.g.
# (*SNAKE_TO_TITLE, apply_replacements({'Lob': 'LOB'})).
SNAKE_TO_TITLE: tuple[ExpressionTransform, ...] = (
    underscores_to_spaces,
    to_titlecase,
)
