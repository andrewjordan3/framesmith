# tests/filters/test_dates.py
"""Tests for ``framesmith.filters.within_complete_period``.

Imports go through the directory's public surface
(``from framesmith.filters import ...``), not the internal ``dates``
module, so the tests exercise the same contract callers see.
"""

from datetime import date, datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from framesmith.filters import within_complete_month, within_complete_period

# ---------------------------------------------------------------------
# Completeness logic
# ---------------------------------------------------------------------


class TestCompletenessLogic:
    def test_complete_trailing_period_keeps_all_rows(self) -> None:
        # Last date is Feb 28; Feb (2025) ends Feb 28 → 0 days from
        # period-end → complete by any non-negative threshold.
        df = pl.DataFrame(
            {'date': [date(2025, 1, 15), date(2025, 2, 28)]}
        )
        result = df.filter(
            within_complete_period('date', period='1mo', threshold_days=5)
        )
        assert result['date'].to_list() == [date(2025, 1, 15), date(2025, 2, 28)]

    def test_incomplete_trailing_period_dropped(self) -> None:
        # Last date is March 15; March ends March 31 → 16 days from
        # period-end → incomplete vs threshold 5.
        df = pl.DataFrame(
            {
                'date': [
                    date(2024, 1, 15),
                    date(2024, 2, 14),
                    date(2024, 3, 1),
                    date(2024, 3, 15),
                ]
            }
        )
        result = df.filter(
            within_complete_period('date', period='1mo', threshold_days=5)
        )
        assert result['date'].to_list() == [
            date(2024, 1, 15),
            date(2024, 2, 14),
        ]

    def test_exactly_at_threshold_period_is_complete(self) -> None:
        # Last date is Feb 23, 2024; Feb ends Feb 29 → 6 days from
        # period-end. Strict ``>``: threshold=6 means 6 > 6 is False,
        # so the period is complete and all rows are kept.
        df = pl.DataFrame(
            {'date': [date(2024, 1, 15), date(2024, 2, 23)]}
        )
        result = df.filter(
            within_complete_period('date', period='1mo', threshold_days=6)
        )
        assert result['date'].to_list() == [
            date(2024, 1, 15),
            date(2024, 2, 23),
        ]

    def test_single_period_incomplete_yields_empty(self) -> None:
        # Only March data, max date is March 15 → incomplete.
        df = pl.DataFrame(
            {'date': [date(2024, 3, 1), date(2024, 3, 15)]}
        )
        result = df.filter(
            within_complete_period('date', period='1mo', threshold_days=5)
        )
        assert result.height == 0

    def test_single_period_complete_keeps_all(self) -> None:
        # Only Feb 2024 data, max date is Feb 29 → 0 days from end.
        df = pl.DataFrame(
            {'date': [date(2024, 2, 1), date(2024, 2, 29)]}
        )
        result = df.filter(
            within_complete_period('date', period='1mo', threshold_days=5)
        )
        assert result['date'].to_list() == [
            date(2024, 2, 1),
            date(2024, 2, 29),
        ]


# ---------------------------------------------------------------------
# Period variations
# ---------------------------------------------------------------------


class TestPeriodVariations:
    def test_week_period_incomplete_drops_trailing_week(self) -> None:
        # Polars truncates weeks to Monday. The trailing date is a
        # Wednesday so the week ends 4 days later → incomplete vs
        # threshold 1.
        df = pl.DataFrame(
            {
                'date': [
                    date(2024, 1, 1),  # Monday, week of Jan 1
                    date(2024, 1, 8),  # Monday, week of Jan 8
                    date(2024, 1, 10),  # Wednesday of week of Jan 8
                ]
            }
        )
        result = df.filter(
            within_complete_period('date', period='1w', threshold_days=1)
        )
        assert result['date'].to_list() == [date(2024, 1, 1)]

    def test_quarter_period_incomplete_drops_trailing_quarter(self) -> None:
        # Q2 2024 ends June 30. Max date June 1 → 29 days from end →
        # incomplete vs threshold 5.
        df = pl.DataFrame(
            {
                'date': [
                    date(2024, 2, 15),
                    date(2024, 3, 31),
                    date(2024, 6, 1),
                ]
            }
        )
        result = df.filter(
            within_complete_period('date', period='1q', threshold_days=5)
        )
        assert result['date'].to_list() == [
            date(2024, 2, 15),
            date(2024, 3, 31),
        ]

    def test_year_period_incomplete_drops_trailing_year(self) -> None:
        # 2024 ends Dec 31. Max date Mar 15 → 291 days from end →
        # incomplete.
        df = pl.DataFrame(
            {
                'date': [
                    date(2022, 6, 1),
                    date(2023, 6, 1),
                    date(2024, 3, 15),
                ]
            }
        )
        result = df.filter(
            within_complete_period('date', period='1y', threshold_days=5)
        )
        assert result['date'].to_list() == [date(2022, 6, 1), date(2023, 6, 1)]

    def test_day_period_threshold_zero_keeps_all(self) -> None:
        # Degenerate case: with period='1d', period-end equals max_date,
        # so days_until is 0. Strict ``>``: 0 > 0 is False → all rows
        # are kept regardless of how recent the trailing day is. The
        # only way to trigger a drop with period='1d' is a negative
        # threshold, exercised in the next test.
        df = pl.DataFrame(
            {
                'date': [
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                ]
            }
        )
        result = df.filter(
            within_complete_period('date', period='1d', threshold_days=0)
        )
        assert result['date'].to_list() == [
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 3),
        ]

    def test_day_period_negative_threshold_drops_trailing_day(self) -> None:
        # Documented degenerate behavior: with period='1d', a negative
        # threshold flips the condition to True (0 > -1) and drops the
        # trailing day. Locked in so the degenerate case stays
        # explicit, not an accident.
        df = pl.DataFrame(
            {
                'date': [
                    date(2024, 1, 1),
                    date(2024, 1, 2),
                    date(2024, 1, 3),
                ]
            }
        )
        result = df.filter(
            within_complete_period('date', period='1d', threshold_days=-1)
        )
        assert result['date'].to_list() == [date(2024, 1, 1), date(2024, 1, 2)]


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_frame_returns_empty(self) -> None:
        df = pl.DataFrame({'date': []}, schema={'date': pl.Date})
        result = df.filter(
            within_complete_period('date', period='1mo', threshold_days=5)
        )
        assert result.height == 0

    def test_all_null_column_keeps_all_rows(self) -> None:
        # max(date) is null → days_until is null → ``when`` condition
        # is null → ``otherwise`` branch (True) keeps all rows.
        df = pl.DataFrame({'date': [None, None]}, schema={'date': pl.Date})
        result = df.filter(
            within_complete_period('date', period='1mo', threshold_days=5)
        )
        assert result.height == 2

    def test_single_row_at_period_end_kept(self) -> None:
        # Last day of Feb 2024 (leap year) → 0 days from period-end.
        df = pl.DataFrame({'date': [date(2024, 2, 29)]})
        result = df.filter(
            within_complete_period('date', period='1mo', threshold_days=5)
        )
        assert result['date'].to_list() == [date(2024, 2, 29)]

    def test_only_trailing_period_is_incomplete(self) -> None:
        # Multiple complete months plus an incomplete trailing month.
        df = pl.DataFrame(
            {
                'date': [
                    date(2024, 1, 15),
                    date(2024, 2, 14),
                    date(2024, 2, 29),
                    date(2024, 3, 1),
                    date(2024, 3, 14),
                ]
            }
        )
        result = df.filter(
            within_complete_period('date', period='1mo', threshold_days=5)
        )
        # Jan and Feb survive; March (max=Mar 14, 17 days from end) drops.
        assert result['date'].to_list() == [
            date(2024, 1, 15),
            date(2024, 2, 14),
            date(2024, 2, 29),
        ]


# ---------------------------------------------------------------------
# Data-driven (not wall-clock) completeness
# ---------------------------------------------------------------------


class TestDataDrivenCompleteness:
    def test_far_past_max_date_behaves_identically(self) -> None:
        # A frame whose max date is years in the past must behave the
        # same way as a recent frame — completeness is from the data,
        # not today. If today's date were used, the result would
        # depend on the wall clock.
        df = pl.DataFrame(
            {
                'date': [
                    date(2010, 1, 15),
                    date(2010, 2, 14),
                    date(2010, 3, 15),
                ]
            }
        )
        result = df.filter(
            within_complete_period('date', period='1mo', threshold_days=5)
        )
        assert result['date'].to_list() == [
            date(2010, 1, 15),
            date(2010, 2, 14),
        ]


# ---------------------------------------------------------------------
# Convenience: within_complete_month
# ---------------------------------------------------------------------


class TestWithinCompleteMonth:
    def test_matches_generic_with_explicit_arguments(self) -> None:
        df = pl.DataFrame(
            {
                'date': [
                    date(2024, 1, 15),
                    date(2024, 2, 14),
                    date(2024, 3, 15),
                ]
            }
        )
        from_generic = df.filter(
            within_complete_period('date', period='1mo', threshold_days=5)
        )
        from_convenience = df.filter(within_complete_month('date'))
        assert_frame_equal(from_generic, from_convenience)

    def test_default_threshold_days_is_5(self) -> None:
        # Explicit threshold_days=5 must equal the no-argument default.
        df = pl.DataFrame(
            {'date': [date(2024, 1, 15), date(2024, 2, 23)]}
        )
        default = df.filter(within_complete_month('date'))
        explicit = df.filter(within_complete_month('date', threshold_days=5))
        assert_frame_equal(default, explicit)

    def test_custom_threshold_passed_through(self) -> None:
        # Feb 23 is 6 days from Feb 29; threshold=6 should keep Feb.
        df = pl.DataFrame(
            {'date': [date(2024, 1, 15), date(2024, 2, 23)]}
        )
        result = df.filter(within_complete_month('date', threshold_days=6))
        assert result['date'].to_list() == [
            date(2024, 1, 15),
            date(2024, 2, 23),
        ]


# ---------------------------------------------------------------------
# Lazy / eager equivalence
# ---------------------------------------------------------------------


class TestLazyEagerEquivalence:
    def test_lazy_filter_matches_eager(self) -> None:
        df = pl.DataFrame(
            {
                'date': [
                    date(2024, 1, 15),
                    date(2024, 2, 14),
                    date(2024, 3, 15),
                ]
            }
        )
        mask = within_complete_period('date', period='1mo', threshold_days=5)
        eager = df.filter(mask)
        lazy = df.lazy().filter(mask).collect()
        assert_frame_equal(eager, lazy)


# ---------------------------------------------------------------------
# Composition with other filter expressions
# ---------------------------------------------------------------------


class TestComposition:
    def test_combined_with_other_boolean_expression(self) -> None:
        df = pl.DataFrame(
            {
                'date': [
                    date(2024, 1, 15),
                    date(2024, 2, 14),
                    date(2024, 3, 15),
                ],
                'amount': [10, -5, 20],
            }
        )
        result = df.filter(
            within_complete_month('date') & (pl.col('amount') > 0)
        )
        # March dropped by the date filter; Feb (-5) dropped by amount.
        # Only Jan 15 (amount=10) survives.
        assert result['date'].to_list() == [date(2024, 1, 15)]
        assert result['amount'].to_list() == [10]


# ---------------------------------------------------------------------
# Date vs Datetime dtype acceptance
# ---------------------------------------------------------------------


class TestDateAndDatetimeAcceptance:
    @pytest.mark.parametrize('dtype', [pl.Date, pl.Datetime('us')])
    def test_accepts_date_and_datetime_columns(
        self, dtype: pl.DataType
    ) -> None:
        if dtype == pl.Date:
            values = [date(2024, 1, 15), date(2024, 3, 15)]
        else:
            # Naive datetimes are intentional here: pl.Datetime('us')
            # is the timezone-naive variant and is the common shape
            # users land on. Timezone correctness is not under test.
            values = [
                datetime(2024, 1, 15, 12, 0),  # noqa: DTZ001
                datetime(2024, 3, 15, 12, 0),  # noqa: DTZ001
            ]
        df = pl.DataFrame({'date': values}, schema={'date': dtype})
        result = df.filter(
            within_complete_period('date', period='1mo', threshold_days=5)
        )
        # March dropped; January kept.
        assert result.height == 1
        assert result['date'][0] == values[0]
