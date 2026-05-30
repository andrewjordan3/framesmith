# tests/transforms/test_names.py
"""Tests for transforms in ``framesmith.transforms.names``."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import (
    extract_email_local_part,
    remove_credentials,
    remove_jr_suffix,
    standardize_initials,
    strip_name_prefixes,
    strip_name_suffixes,
)


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


class TestStripNameSuffixes:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('John Smith Jr', 'John Smith'),
            ('John Smith, Jr.', 'John Smith'),
            ('Jane Doe III', 'Jane Doe'),
            ('Bob IV', 'Bob'),
            ('Mary II', 'Mary'),
            ('John Smith Esq', 'John Smith'),
            ('Robert Snr', 'Robert'),
            ('JOHN SMITH JR', 'JOHN SMITH'),
        ],
    )
    def test_strips_trailing_suffix(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], strip_name_suffixes())
        assert result.to_list() == [expected]

    @pytest.mark.parametrize(
        'value',
        [
            'Hawaii',  # ends in 'ii' but no separator
            'Hawaii Bank',  # 'ii' is interior, not a trailing token
            'John V',  # bare V is a middle initial, excluded
            'Smith',  # no suffix
        ],
    )
    def test_leaves_lookalikes_unchanged(self, value: str) -> None:
        result = _apply([value], strip_name_suffixes())
        assert result.to_list() == [value]

    def test_null_propagates(self) -> None:
        result = _apply([None], strip_name_suffixes())
        assert result.to_list() == [None]

    def test_custom_suffix_list(self) -> None:
        # A token not in the default set.
        result = _apply(['Jane Doe PhD'], strip_name_suffixes(['phd']))
        assert result.to_list() == ['Jane Doe']

    def test_empty_suffixes_raises(self) -> None:
        with pytest.raises(ValueError, match='empty'):
            strip_name_suffixes([])

    def test_factory_returns_callable(self) -> None:
        assert callable(strip_name_suffixes())

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['John Smith Jr', 'Hawaii', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', [strip_name_suffixes()])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestStripNamePrefixes:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('Dr. John Smith', 'John Smith'),
            ('Mr John', 'John'),
            ('Mrs. Jane Doe', 'Jane Doe'),
            ('Prof Bob', 'Bob'),
            ('Ms Lee', 'Lee'),
            ('Miss Jane', 'Jane'),
            ('DR JOHN', 'JOHN'),
        ],
    )
    def test_strips_leading_prefix(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], strip_name_prefixes())
        assert result.to_list() == [expected]

    @pytest.mark.parametrize(
        'value',
        [
            'Drake Smith',  # 'Dr' not followed by a separator
            'St. John',  # St is a surname element, excluded
            'Mr',  # no following separator
        ],
    )
    def test_leaves_lookalikes_unchanged(self, value: str) -> None:
        result = _apply([value], strip_name_prefixes())
        assert result.to_list() == [value]

    def test_null_propagates(self) -> None:
        result = _apply([None], strip_name_prefixes())
        assert result.to_list() == [None]

    def test_empty_prefixes_raises(self) -> None:
        with pytest.raises(ValueError, match='empty'):
            strip_name_prefixes([])

    def test_factory_returns_callable(self) -> None:
        assert callable(strip_name_prefixes())

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['Dr. John Smith', 'Drake Smith', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', [strip_name_prefixes()])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestRemoveCredentials:
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('Jane Smith, MD', 'Jane Smith'),
            ('Jane Smith, M.D.', 'Jane Smith'),  # period-tolerance
            ('Jane Smith, MD, PhD', 'Jane Smith'),  # stacking
            ('Jane Smith, MD, PhD, FACS', 'Jane Smith'),  # 3-stack
            ('Jane Smith , MD', 'Jane Smith'),  # space before comma
            ('Smith,MD', 'Smith'),  # no space after comma
            ('jane smith, phd', 'jane smith'),  # case-insensitive
            ('Smith, PhD.', 'Smith'),  # trailing period
        ],
    )
    def test_strips_trailing_credentials(
        self, value: str, expected: str
    ) -> None:
        result = _apply([value], remove_credentials())
        assert result.to_list() == [expected]

    @pytest.mark.parametrize(
        'value',
        [
            'John Smith MD',  # no comma -> not stripped
            'Nguyen Do',  # no comma
            'John Doe',  # no credential
        ],
    )
    def test_leaves_no_comma_forms_unchanged(self, value: str) -> None:
        result = _apply([value], remove_credentials())
        assert result.to_list() == [value]

    def test_space_preceded_surname_not_stripped_even_with_custom_token(
        self,
    ) -> None:
        # The comma-required separator is what protects 'Do': even when
        # 'do' is in the credential list, the space-preceded surname is
        # safe, while the comma-set-off ', MD' is stripped.
        result = _apply(['Mary Do, MD'], remove_credentials(['md', 'do']))
        assert result.to_list() == ['Mary Do']

    def test_null_propagates(self) -> None:
        result = _apply([None], remove_credentials())
        assert result.to_list() == [None]

    def test_custom_credential_list(self) -> None:
        # A token not in the default set.
        result = _apply(['Jane Smith, ASA'], remove_credentials(['asa']))
        assert result.to_list() == ['Jane Smith']

    def test_empty_credentials_raises(self) -> None:
        with pytest.raises(ValueError, match='empty'):
            remove_credentials([])

    def test_factory_returns_callable(self) -> None:
        assert callable(remove_credentials())

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['Jane Smith, MD, PhD', 'John Smith MD', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', [remove_credentials()])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)


class TestStandardizeInitials:
    @pytest.mark.parametrize(
        'value',
        ['J. R. Smith', 'J R Smith', 'J.R. Smith', 'J.R.Smith'],
    )
    def test_normalizes_to_canonical_form(self, value: str) -> None:
        result = _apply([value], standardize_initials)
        assert result.to_list() == ['J. R. Smith']

    def test_middle_initial_normalized(self) -> None:
        result = _apply(['Mary J. Smith'], standardize_initials)
        assert result.to_list() == ['Mary J. Smith']

    @pytest.mark.parametrize(
        'value',
        [
            'Jo Smith',  # multi-letter token, not an initial
            'Smith',  # multi-letter token
            'JR Smith',  # glued pair, ambiguous with name / Jr suffix
        ],
    )
    def test_leaves_non_initials_unchanged(self, value: str) -> None:
        result = _apply([value], standardize_initials)
        assert result.to_list() == [value]

    def test_case_preserved(self) -> None:
        result = _apply(['j. r. smith'], standardize_initials)
        assert result.to_list() == ['j. r. smith']

    def test_null_propagates(self) -> None:
        result = _apply([None], standardize_initials)
        assert result.to_list() == [None]

    def test_output_dtype_is_string(self) -> None:
        result = _apply(['J. R. Smith'], standardize_initials)
        assert result.dtype == pl.String

    def test_lazy_and_eager_produce_identical_results(self) -> None:
        df = pl.DataFrame(
            {'x': ['J R Smith', 'Jo Smith', None]},
            schema={'x': pl.String},
        )
        expr = compose_column('x', [standardize_initials])
        eager = df.with_columns(expr)
        lazy = df.lazy().with_columns(expr).collect()
        assert_frame_equal(eager, lazy)
