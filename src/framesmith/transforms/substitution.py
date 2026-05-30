# framesmith/transforms/substitution.py
"""Character- and substring-substitution transforms.

Single-character substitutions (``remove_apostrophes``, ``remove_periods``,
``periods_to_spaces``, ``underscores_to_spaces``, ``replace_ampersand_with_and``)
and the configurable literal-substring ``apply_replacements`` factory. All
follow the ``ExpressionTransform`` contract.
"""

import polars as pl

from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'apply_replacements',
    'periods_to_spaces',
    'remove_apostrophes',
    'remove_periods',
    'replace_ampersand_with_and',
    'underscores_to_spaces',
]


def replace_ampersand_with_and(expr: pl.Expr) -> pl.Expr:
    """Replace literal ``&`` with ``and``."""
    return expr.str.replace_all('&', 'and', literal=True)


def remove_apostrophes(expr: pl.Expr) -> pl.Expr:
    """Remove all apostrophe characters (``'``)."""
    return expr.str.replace_all("'", '', literal=True)


def remove_periods(expr: pl.Expr) -> pl.Expr:
    """Remove all period characters (``.``)."""
    return expr.str.replace_all('.', '', literal=True)


def periods_to_spaces(expr: pl.Expr) -> pl.Expr:
    """Replace each period with a single space.

    Atomic: one period → one space, even in runs. ``'U.S.A'`` becomes
    ``'U S A'`` and ``'john..doe'`` becomes ``'john  doe'`` (two
    spaces). Compose with :func:`collapse_whitespace` downstream if
    you want repeated dots to collapse to a single space.

    Nulls pass through as null.
    """
    return expr.str.replace_all('.', ' ', literal=True)


def underscores_to_spaces(expr: pl.Expr) -> pl.Expr:
    """Replace each underscore with a single space.

    Atomic: one underscore → one space, even in runs. ``'john_smith'``
    becomes ``'john smith'`` and ``'a__b'`` becomes ``'a  b'`` (two
    spaces). Compose with :func:`collapse_whitespace` downstream if you
    want repeated underscores to collapse to a single space. Parallel to
    :func:`periods_to_spaces`. Nulls pass through unchanged.
    """
    return expr.str.replace_all('_', ' ', literal=True)


def apply_replacements(replacements: dict[str, str]) -> ExpressionTransform:
    """Build a transform applying literal substring replacements.

    Each key found anywhere in the string is replaced with its mapped
    value, in one pass (``str.replace_many``). Useful for fixing
    acronyms after title casing, expanding abbreviations, or mapping a
    controlled vocabulary — e.g. ``apply_replacements({'Lob': 'LOB',
    'Ccms': 'CCMS'})`` turns ``'Primary Lob'`` into ``'Primary LOB'``.

    Matching is literal substring, not word-boundary: ``{'Lob': 'LOB'}``
    also rewrites ``'Lobster'`` to ``'LOBster'``. Choose keys that won't
    collide with substrings you mean to keep. Nulls pass through
    unchanged.

    Args:
        replacements: Map of literal substrings to their replacements.
            Must be non-empty.

    Returns:
        An ``ExpressionTransform``. Applied via ``compose_column``.

    Raises:
        ValueError: If ``replacements`` is empty — a no-op replacer is
            almost certainly a bug.

    Example:
        >>> import polars as pl
        >>> from framesmith import compose_column
        >>> from framesmith.transforms import apply_replacements
        >>> fix = apply_replacements({'Lob': 'LOB'})
        >>> df = pl.DataFrame({'x': ['Primary Lob', 'Rep Lob']})
        >>> df.with_columns(compose_column('x', [fix]))['x'].to_list()
        ['Primary LOB', 'Rep LOB']
    """
    if len(replacements) == 0:
        raise ValueError('replacements must not be empty')

    def _apply_replacements(expr: pl.Expr) -> pl.Expr:
        return expr.str.replace_many(replacements)

    return _apply_replacements
