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

from framesmith.transforms.numeric import (
    accounting_parens_to_negative,
    cast_to_float64,
    percent_to_fraction,
    remove_thousands_separators,
    trailing_minus_to_prefix,
)
from framesmith.transforms.text import (
    collapse_whitespace,
    fold_to_ascii,
    normalize_unicode_nfkc,
    nullify_blank_strings,
    remove_apostrophes,
    remove_periods,
    replace_ampersand_with_and,
    strip_whitespace,
)
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'CLEAN_NUMERIC_STRING',
    'NORMALIZE_NUMERIC',
    'NORMALIZE_PERCENT',
    'NORMALIZE_TEXT',
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
