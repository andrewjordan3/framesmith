# tests/transforms/test_names.py
"""Tests for ``framesmith.transforms.names.remove_jr_suffix``."""

import polars as pl
import pytest

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import remove_jr_suffix


def _apply(
    values: list[str | None], transform: ExpressionTransform
) -> pl.Series:
    """Run a single transform on a 1-column ``pl.String`` frame."""
    df = pl.DataFrame({'x': values}, schema={'x': pl.String})
    return df.with_columns(compose_column('x', [transform]))['x']


class TestRemoveJrSuffix:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('John Smith Jr', 'John Smith'),
            ('John Smith Jr.', 'John Smith'),
            ('John Smith, Jr.', 'John Smith'),
            ('John Smith jr', 'John Smith'),
            ('John Smith JR', 'John Smith'),
        ],
    )
    def test_strips_trailing_jr_variants(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], remove_jr_suffix)
        assert result.to_list() == [expected]

    def test_interior_jr_unchanged(self) -> None:
        # The pattern is end-anchored, so a 'Jr' that is not at the end
        # of the string is left alone.
        result = _apply(['Jr Bakery'], remove_jr_suffix)
        assert result.to_list() == ['Jr Bakery']

    def test_no_suffix_unchanged(self) -> None:
        result = _apply(['Smith'], remove_jr_suffix)
        assert result.to_list() == ['Smith']

    def test_null_propagates(self) -> None:
        result = _apply([None], remove_jr_suffix)
        assert result.to_list() == [None]

    def test_interior_whitespace_preserved(self) -> None:
        # Single-responsibility behavior: remove_jr_suffix removes only
        # the suffix and does not touch interior whitespace. Compose
        # collapse_whitespace if tidying is required.
        result = _apply(['John  Smith Jr'], remove_jr_suffix)
        assert result.to_list() == ['John  Smith']
