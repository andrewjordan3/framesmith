import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith import ExpressionTransform, compose_column
from framesmith.transforms import pad_left


def _apply(values: list[str | None], transform: ExpressionTransform) -> pl.Series:
    df = pl.DataFrame({'x': values}, schema={'x': pl.String})
    return df.with_columns(compose_column('x', [transform]))['x']


class TestPadLeft:
    def test_pads_short_value(self) -> None:
        assert _apply(['100682'], pad_left(17)).to_list() == ['00000000000100682']

    def test_over_width_is_not_truncated(self) -> None:
        long_value = '123456789012345678'  # 18 chars
        assert _apply([long_value], pad_left(17)).to_list() == [long_value]

    def test_exact_width_unchanged(self) -> None:
        assert _apply(['1HGCM82633A004352'], pad_left(17)).to_list() == ['1HGCM82633A004352']

    def test_null_propagates(self) -> None:
        assert _apply([None], pad_left(17)).to_list() == [None]

    def test_custom_fill(self) -> None:
        assert _apply(['12'], pad_left(5, fill='x')).to_list() == ['xxx12']

    def test_output_dtype_is_string(self) -> None:
        assert _apply(['12'], pad_left(5)).dtype == pl.String

    def test_width_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match='width must be'):
            pad_left(0)

    def test_multi_char_fill_raises(self) -> None:
        with pytest.raises(ValueError, match='exactly one character'):
            pad_left(5, fill='00')

    def test_factory_returns_callable(self) -> None:
        assert callable(pad_left(5))

    def test_lazy_matches_eager(self) -> None:
        df = pl.DataFrame({'x': ['12', None, '123456']}, schema={'x': pl.String})
        expr = compose_column('x', [pad_left(5)])
        assert_frame_equal(df.with_columns(expr), df.lazy().with_columns(expr).collect())
