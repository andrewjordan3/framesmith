# tests/transforms/test_whitespace.py
"""Tests for whitespace transforms in ``framesmith.transforms.whitespace``.

Each transform is exercised in isolation through ``compose_column`` so
the tests also cover the integration point.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    collapse_whitespace,
    nullify_blank_strings,
    replace_whitespace_with,
    strip_whitespace,
)


def _apply(
    values: list[str | None], transform: ExpressionTransform
) -> pl.Series:
    """Run a single transform on a 1-column ``pl.String`` frame.

    An explicit schema lets all-null inputs through without dtype
    inference failing.
    """
    df = pl.DataFrame({'x': values}, schema={'x': pl.String})
    return df.with_columns(compose_column('x', [transform]))['x']


class TestCollapseWhitespace:
    def test_multiple_internal_spaces_collapsed(self) -> None:
        result = _apply(['a   b'], collapse_whitespace)
        assert result.to_list() == ['a b']

    def test_tabs_and_newlines_collapsed_to_space(self) -> None:
        result = _apply(['a\t\nb'], collapse_whitespace)
        assert result.to_list() == ['a b']

    def test_leading_and_trailing_whitespace_not_stripped(self) -> None:
        # Boundary proof: collapse is not strip. The runs at the ends
        # collapse to a single space each, but they are not removed.
        result = _apply(['   a   b   '], collapse_whitespace)
        assert result.to_list() == [' a b ']

    def test_null_propagates(self) -> None:
        result = _apply([None], collapse_whitespace)
        assert result.to_list() == [None]


class TestStripWhitespace:
    def test_leading_and_trailing_stripped(self) -> None:
        result = _apply(['  hello  '], strip_whitespace)
        assert result.to_list() == ['hello']

    def test_interior_runs_not_collapsed(self) -> None:
        # Boundary proof: strip is not collapse. Interior runs survive.
        result = _apply(['  a   b  '], strip_whitespace)
        assert result.to_list() == ['a   b']

    def test_null_propagates(self) -> None:
        result = _apply([None], strip_whitespace)
        assert result.to_list() == [None]


class TestNullifyBlankStrings:
    def test_empty_string_becomes_null(self) -> None:
        result = _apply([''], nullify_blank_strings)
        assert result.to_list() == [None]

    def test_whitespace_only_becomes_null(self) -> None:
        result = _apply(['   '], nullify_blank_strings)
        assert result.to_list() == [None]

    def test_tab_only_becomes_null(self) -> None:
        result = _apply(['\t'], nullify_blank_strings)
        assert result.to_list() == [None]

    def test_non_blank_unchanged(self) -> None:
        result = _apply([' hello '], nullify_blank_strings)
        assert result.to_list() == [' hello ']

    def test_existing_null_unchanged(self) -> None:
        result = _apply([None], nullify_blank_strings)
        assert result.to_list() == [None]


class TestReplaceWhitespaceWith:
    @pytest.mark.parametrize(
        ('separator', 'expected'),
        [
            ('_', 'hello_world'),
            ('-', 'hello-world'),
            ('.', 'hello.world'),
            ('__', 'hello__world'),
            ('', 'helloworld'),
            (' ', 'hello world'),
        ],
    )
    def test_separator_variations(
        self, separator: str, expected: str
    ) -> None:
        transform = replace_whitespace_with(separator)
        result = _apply(['hello world'], transform)
        assert result.to_list() == [expected]

    def test_multiple_spaces_collapse_to_one_separator(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply(['hello   world'], transform)
        assert result.to_list() == ['hello_world']

    def test_tab_replaced(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply(['a\tb'], transform)
        assert result.to_list() == ['a_b']

    def test_newline_replaced(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply(['a\nb'], transform)
        assert result.to_list() == ['a_b']

    def test_mixed_tabs_and_spaces_collapse_to_one_separator(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply(['a \t b'], transform)
        assert result.to_list() == ['a_b']

    def test_leading_and_trailing_whitespace_replaced_not_stripped(
        self,
    ) -> None:
        # Atomic-behavior contract: ends become separators just like
        # interior whitespace.
        transform = replace_whitespace_with('_')
        result = _apply([' hello world '], transform)
        assert result.to_list() == ['_hello_world_']

    def test_no_interior_text_change(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply(['HelloWorld'], transform)
        assert result.to_list() == ['HelloWorld']

    def test_null_propagates(self) -> None:
        transform = replace_whitespace_with('_')
        result = _apply([None], transform)
        assert result.to_list() == [None]

    def test_factory_returns_callable(self) -> None:
        transform = replace_whitespace_with('_')
        assert callable(transform)

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['hello world', 'a  b', ' edges ', None]},
            schema={'x': pl.String},
        )
        transform = replace_whitespace_with('_')
        expr = compose_column('x', [transform])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestNullPropagationBatch:
    """Batch null propagation across this module's transforms."""

    def test_null_propagates_for_module_transforms(self) -> None:
        transforms: list[ExpressionTransform] = [
            collapse_whitespace,
            strip_whitespace,
            nullify_blank_strings,
            replace_whitespace_with('_'),
        ]
        expected = pl.Series('x', [None], dtype=pl.String)
        for transform in transforms:
            result = _apply([None], transform)
            assert_series_equal(result, expected)
