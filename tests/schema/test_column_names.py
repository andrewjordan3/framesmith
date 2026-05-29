# tests/schema/test_column_names.py
"""Tests for ``framesmith.schema.normalize_column_names``.

Imports go through the directory's public surface
(``from framesmith.schema import ...``), not the internal
``column_names`` module, so the tests exercise the same contract
callers see.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import framesmith.schema.column_names as column_names_module
from framesmith.schema import normalize_column_names

# ---------------------------------------------------------------------
# Normalization basics
# ---------------------------------------------------------------------


class TestNormalizationBasics:
    @pytest.mark.parametrize(
        ('source_label', 'expected_label'),
        [
            (' First Name ', 'first_name'),
            ('id', 'id'),
            ('a b', 'a_b'),
            ('LAST NAME', 'last_name'),
        ],
    )
    def test_strip_lower_and_separator(
        self, source_label: str, expected_label: str
    ) -> None:
        df = pl.DataFrame({source_label: [1]})
        result = df.select(normalize_column_names())
        assert result.columns == [expected_label]

    @pytest.mark.parametrize(
        ('separator', 'expected_label'),
        [
            ('_', 'first_name'),
            ('-', 'first-name'),
            ('', 'firstname'),
            ('__', 'first__name'),
        ],
    )
    def test_separator_variations(
        self, separator: str, expected_label: str
    ) -> None:
        df = pl.DataFrame({'first name': [1]})
        result = df.select(normalize_column_names(separator))
        assert result.columns == [expected_label]


# ---------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------


class TestDataIntegrity:
    def test_values_unchanged(self) -> None:
        df = pl.DataFrame({'A B': [1, 2], 'C': [3, 4]})
        result = df.select(normalize_column_names())
        assert result.columns == ['a_b', 'c']
        assert result['a_b'].to_list() == [1, 2]
        assert result['c'].to_list() == [3, 4]


# ---------------------------------------------------------------------
# Collisions and lazy/eager equivalence
# ---------------------------------------------------------------------


class TestCollisionAndEvaluation:
    def test_colliding_names_raise(self) -> None:
        df = pl.DataFrame({'first name': [1], 'First Name': [2]})
        with pytest.raises(pl.exceptions.DuplicateError):
            df.select(normalize_column_names())

    def test_lazy_eager_equivalence(self) -> None:
        df = pl.DataFrame({'A B': [1, 2], 'C D': [3, 4]})
        expr = normalize_column_names()
        assert_frame_equal(
            df.select(expr), df.lazy().select(expr).collect()
        )


# ---------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------


class TestPublicSurface:
    def test_reexport_identity(self) -> None:
        assert (
            normalize_column_names
            is column_names_module.normalize_column_names
        )
