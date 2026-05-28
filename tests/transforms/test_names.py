# tests/transforms/test_names.py
"""Tests for transforms in ``framesmith.transforms.names``."""

import polars as pl
import pytest

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import extract_email_local_part, remove_jr_suffix


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


class TestExtractEmailLocalPart:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('john@example.com', 'john'),
            # Periods preserved at this stage — that's periods_to_spaces' job.
            ('john.doe@example.com', 'john.doe'),
            ('jane.q.smith@example.com', 'jane.q.smith'),
        ],
    )
    def test_takes_part_before_first_at(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], extract_email_local_part)
        assert result.to_list() == [expected]

    def test_no_at_sign_unchanged(self) -> None:
        result = _apply(['noatsign'], extract_email_local_part)
        assert result.to_list() == ['noatsign']

    def test_leading_at_yields_empty_string(self) -> None:
        result = _apply(['@example.com'], extract_email_local_part)
        assert result.to_list() == ['']

    def test_multiple_at_signs_split_on_first(self) -> None:
        # Matches the pandas reference: str.split('@', n=1).str[0].
        result = _apply(['john@host@subhost'], extract_email_local_part)
        assert result.to_list() == ['john']

    def test_empty_string_unchanged(self) -> None:
        result = _apply([''], extract_email_local_part)
        assert result.to_list() == ['']

    def test_null_propagates(self) -> None:
        result = _apply([None], extract_email_local_part)
        assert result.to_list() == [None]

    def test_does_not_strip_surrounding_whitespace(self) -> None:
        # Atomic-contract proof: whitespace is preserved verbatim in
        # the local part. Compose strip_whitespace upstream if needed.
        result = _apply([' padded@example.com'], extract_email_local_part)
        assert result.to_list() == [' padded']

    def test_output_dtype_is_string(self) -> None:
        result = _apply(['john@example.com'], extract_email_local_part)
        assert result.dtype == pl.String
