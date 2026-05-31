import polars as pl
import pytest

from framesmith.canonicalize import canonicalize_vin


class TestCanonicalizeVin:
    def test_pads_short_vin(self) -> None:
        df = pl.DataFrame({'vin': ['100682']}, schema={'vin': pl.String})
        assert canonicalize_vin(df, 'vin')['vin'].to_list() == ['00000000000100682']

    def test_exact_length_unchanged(self) -> None:
        df = pl.DataFrame({'vin': ['1HGCM82633A004352']}, schema={'vin': pl.String})
        assert canonicalize_vin(df, 'vin')['vin'].to_list() == ['1HGCM82633A004352']

    def test_over_length_raises(self) -> None:
        df = pl.DataFrame({'vin': ['123456789012345678']}, schema={'vin': pl.String})
        with pytest.raises(ValueError, match='outside'):
            canonicalize_vin(df, 'vin')

    def test_null_raises_by_default(self) -> None:
        df = pl.DataFrame({'vin': ['100682', None]}, schema={'vin': pl.String})
        with pytest.raises(ValueError, match='outside'):
            canonicalize_vin(df, 'vin')

    def test_ignore_null_allows_null(self) -> None:
        df = pl.DataFrame({'vin': ['100682', None]}, schema={'vin': pl.String})
        out = canonicalize_vin(df, 'vin', ignore_null=True)
        assert out['vin'].to_list() == ['00000000000100682', None]

    def test_returns_dataframe(self) -> None:
        df = pl.DataFrame({'vin': ['100682']}, schema={'vin': pl.String})
        assert isinstance(canonicalize_vin(df, 'vin'), pl.DataFrame)
