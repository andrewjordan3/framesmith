# framesmith/transforms/names.py
"""Atomic transforms for person-name normalization."""

import re
from collections.abc import Sequence

import polars as pl

from framesmith._internal import (
    DEFAULT_CREDENTIALS,
    DEFAULT_NAME_PREFIXES,
    DEFAULT_NAME_SUFFIXES,
    TRAILING_JR_PATTERN,
)
from framesmith.types import ExpressionTransform

__all__: list[str] = [
    'DEFAULT_CREDENTIALS',
    'DEFAULT_NAME_PREFIXES',
    'DEFAULT_NAME_SUFFIXES',
    'extract_email_local_part',
    'remove_credentials',
    'remove_jr_suffix',
    'strip_name_prefixes',
    'strip_name_suffixes',
]


def remove_jr_suffix(expr: pl.Expr) -> pl.Expr:
    """Remove a trailing ``jr`` or ``jr.`` suffix.

    Case-insensitive, with an optional comma and/or whitespace before
    the suffix. The pattern is end-anchored, so interior ``jr`` (e.g.
    ``"Jr Bakery"``) is not matched.

    Atomic: removes only the suffix. Does not collapse interior
    whitespace or strip ends. ``TRAILING_JR_PATTERN`` consumes the
    separator adjacent to the suffix, so normal input (``"John Smith
    Jr"``) comes out clean (``"John Smith"``) without further steps;
    pathological interior runs require :func:`collapse_whitespace` and
    :func:`strip_whitespace` for tidy-up.
    """
    return expr.str.replace(TRAILING_JR_PATTERN, '')


def extract_email_local_part(expr: pl.Expr) -> pl.Expr:
    """Take everything before the first ``'@'``; no ``'@'`` → unchanged.

    Splits the string on ``'@'`` and returns the first segment. A
    string with no ``'@'`` passes through as-is. A string starting
    with ``'@'`` yields the empty string. Strings with multiple
    ``'@'`` characters return only the part before the first one,
    matching the pandas reference behavior of
    ``str.split('@', n=1).str[0]``.

    Atomic: does NOT strip surrounding whitespace, lowercase, or
    further normalize. Compose with :func:`strip_whitespace` upstream
    if you need to handle whitespace-padded email strings, and with
    :func:`periods_to_spaces` downstream if you want display-name
    shape.

    Nulls pass through as null.
    """
    return expr.str.split('@').list.first()


def _name_token_alternation(tokens: Sequence[str]) -> str:
    """Build a regex alternation from affix tokens, longest-first and escaped.

    Longest-first so a token that is a prefix of a longer one does not
    shadow it ('iii' before 'ii', 'mrs' before 'mr'); each token is
    escaped so it matches literally.
    """
    ordered_tokens: list[str] = sorted(tokens, key=len, reverse=True)
    return '|'.join(re.escape(token) for token in ordered_tokens)


def strip_name_suffixes(
    suffixes: Sequence[str] = DEFAULT_NAME_SUFFIXES,
) -> ExpressionTransform:
    """Remove a trailing name suffix (Jr, Sr, II-IV, Esq, …).

    Case-insensitive. The suffix must be a separate trailing token —
    preceded by a comma and/or space and at the end of the string — so
    ``"John Smith, Jr."`` and ``"Jane Doe III"`` are stripped while
    ``"Hawaii"`` (which merely ends in "ii") is left alone. The separator
    requirement is mandatory for exactly this reason.

    Atomic: removes only the suffix and its adjacent separator; does not
    collapse interior whitespace or strip ends. Nulls pass through. The
    default set excludes bare ``I`` and ``V`` to avoid clobbering
    single-letter middle initials; pass ``suffixes`` to customize.

    Args:
        suffixes: Suffix tokens (bare, lowercase or any case). Must be
            non-empty.

    Returns:
        An ``ExpressionTransform`` for ``compose_column``.

    Raises:
        ValueError: If ``suffixes`` is empty.
    """
    if len(suffixes) == 0:
        raise ValueError('suffixes must not be empty')

    pattern: str = rf'(?i)[, ]+(?:{_name_token_alternation(suffixes)})\.?$'

    def _strip_name_suffixes(expr: pl.Expr) -> pl.Expr:
        return expr.str.replace(pattern, '')

    return _strip_name_suffixes


def strip_name_prefixes(
    prefixes: Sequence[str] = DEFAULT_NAME_PREFIXES,
) -> ExpressionTransform:
    """Remove a leading honorific prefix (Mr, Mrs, Ms, Dr, Prof, …).

    Case-insensitive. The prefix must be a leading token followed by a
    separator (optional period then whitespace), so ``"Dr. John Smith"``
    and ``"Mrs Jane Doe"`` are stripped while ``"Drake Smith"`` is left
    alone. The default set excludes ``St`` (a surname element, e.g.
    ``"St. John"``); pass ``prefixes`` to customize.

    Atomic: removes only the prefix and its adjacent separator; does not
    collapse interior whitespace or strip ends. Nulls pass through.

    Args:
        prefixes: Prefix tokens (bare, any case). Must be non-empty.

    Returns:
        An ``ExpressionTransform`` for ``compose_column``.

    Raises:
        ValueError: If ``prefixes`` is empty.
    """
    if len(prefixes) == 0:
        raise ValueError('prefixes must not be empty')

    pattern: str = rf'(?i)^(?:{_name_token_alternation(prefixes)})\.?\s+'

    def _strip_name_prefixes(expr: pl.Expr) -> pl.Expr:
        return expr.str.replace(pattern, '')

    return _strip_name_prefixes


def _credential_alternation(tokens: Sequence[str]) -> str:
    """Build a period-tolerant regex alternation from credential tokens.

    Longest-first (so a token that prefixes a longer one does not shadow
    it), each character escaped and followed by an optional period, so
    ``'phd'`` becomes ``p\\.?h\\.?d\\.?`` and matches ``'PhD'``,
    ``'Ph.D.'``, and ``'P.H.D'`` alike.
    """
    ordered_tokens: list[str] = sorted(tokens, key=len, reverse=True)
    return '|'.join(
        ''.join(re.escape(character) + r'\.?' for character in token)
        for token in ordered_tokens
    )


def remove_credentials(
    credentials: Sequence[str] = DEFAULT_CREDENTIALS,
) -> ExpressionTransform:
    """Remove trailing comma-separated professional credentials.

    Strips one or more credentials at the end of the string, each
    preceded by a comma: ``"Jane Smith, MD, PhD, FACS"`` → ``"Jane
    Smith"``. Case-insensitive, and tolerant of internal periods so
    ``"Jane Smith, Ph.D."`` is handled.

    A comma is required before each credential — this is what keeps a
    space-preceded surname that happens to match a credential token safe
    (``"Mary Do, MD"`` → ``"Mary Do"``). As a consequence, credentials
    written without a comma (``"John Smith MD"``) are left unchanged.

    Atomic: removes only the trailing credential run and its adjacent
    commas/whitespace; does not collapse interior whitespace or strip
    other ends. Nulls pass through. The default list omits collision-prone
    short tokens (``do``, ``pa``, bare degrees); pass ``credentials`` to
    customize — note that adding a token equal to a possible comma-set-off
    middle name (``"Smith, Do, MD"``) will strip that token too.

    Args:
        credentials: Credential tokens (bare, any case). Must be
            non-empty.

    Returns:
        An ``ExpressionTransform`` for ``compose_column``.

    Raises:
        ValueError: If ``credentials`` is empty.
    """
    if len(credentials) == 0:
        raise ValueError('credentials must not be empty')

    pattern: str = (
        rf'(?i)(?:\s*,\s*(?:{_credential_alternation(credentials)}))+$'
    )

    def _remove_credentials(expr: pl.Expr) -> pl.Expr:
        return expr.str.replace(pattern, '')

    return _remove_credentials
