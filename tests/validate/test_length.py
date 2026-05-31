import polars as pl
import pytest

from framesmith.validate import assert_string_length


class TestAssertStringLength:
    def test_passes_when_all_within_bounds(self) -> None:
        df = pl.DataFrame({'v': ['1HGCM82633A004352']}, schema={'v': pl.String})
        assert assert_string_length(df, 'v', min_length=17, max_length=17) is None

    def test_raises_on_too_short(self) -> None:
        df = pl.DataFrame({'v': ['100682']}, schema={'v': pl.String})
        with pytest.raises(ValueError, match='outside'):
            assert_string_length(df, 'v', min_length=17, max_length=17)

    def test_raises_on_too_long(self) -> None:
        df = pl.DataFrame({'v': ['123456789012345678']}, schema={'v': pl.String})
        with pytest.raises(ValueError, match='outside'):
            assert_string_length(df, 'v', max_length=17)

    def test_min_only(self) -> None:
        df = pl.DataFrame({'v': ['abcde', 'ab']}, schema={'v': pl.String})
        with pytest.raises(ValueError, match='outside'):
            assert_string_length(df, 'v', min_length=3)

    def test_null_is_violation_by_default(self) -> None:
        df = pl.DataFrame({'v': [None]}, schema={'v': pl.String})
        with pytest.raises(ValueError, match='outside'):
            assert_string_length(df, 'v', max_length=17)

    def test_ignore_null_skips_nulls(self) -> None:
        df = pl.DataFrame({'v': ['1HGCM82633A004352', None]}, schema={'v': pl.String})
        assert assert_string_length(df, 'v', min_length=17, max_length=17, ignore_null=True) is None

    def test_neither_bound_raises(self) -> None:
        df = pl.DataFrame({'v': ['x']}, schema={'v': pl.String})
        with pytest.raises(ValueError, match='at least one'):
            assert_string_length(df, 'v')

    def test_empty_frame_passes(self) -> None:
        df = pl.DataFrame({'v': []}, schema={'v': pl.String})
        assert assert_string_length(df, 'v', min_length=17, max_length=17) is None

    def test_message_reports_count_and_examples(self) -> None:
        df = pl.DataFrame({'v': ['ab', 'cd']}, schema={'v': pl.String})
        with pytest.raises(ValueError, match=r'2 value\(s\)'):
            assert_string_length(df, 'v', min_length=5)

    def test_non_string_column_raises(self) -> None:
        df = pl.DataFrame({'v': [1, 2]}, schema={'v': pl.Int64})
        with pytest.raises(pl.exceptions.SchemaError):
            assert_string_length(df, 'v', max_length=17)
