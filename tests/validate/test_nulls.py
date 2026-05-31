import polars as pl
import pytest

from framesmith.validate import assert_no_nulls


class TestAssertNoNulls:
    def test_passes_when_no_nulls(self) -> None:
        assert assert_no_nulls(pl.DataFrame({'id': [1, 2, 3]}), 'id') is None

    def test_raises_on_nulls_with_count(self) -> None:
        df = pl.DataFrame({'id': [1, None, None]}, schema={'id': pl.Int64})
        with pytest.raises(ValueError, match='2 null'):
            assert_no_nulls(df, 'id')

    def test_empty_frame_passes(self) -> None:
        df = pl.DataFrame({'id': []}, schema={'id': pl.Int64})
        assert assert_no_nulls(df, 'id') is None

    def test_missing_column_raises(self) -> None:
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            assert_no_nulls(pl.DataFrame({'id': [1]}), 'nope')
