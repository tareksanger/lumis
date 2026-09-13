"""Small deterministic tests for number, string, and date utilities."""

from datetime import date, datetime, timedelta, timezone
from unittest.mock import Mock

from lumis.core.utils import string as string_utils, time as time_utils
from lumis.core.utils.number import format_number

import pytest


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (1, "1"),
        (-42, "-42"),
        (12.25, "12.2"),
        (999, "999"),
        (1000, "1K"),
        (-1500, "-1.5K"),
        (1_234_567, "1.2M"),
        (1_000_000_000, "1B"),
        (9_876_543_210, "9.9B"),
        (1_000_000_000_000, "1T"),
        (1_250_000_000_000, "1.2T"),
        (1_000_000_000_000_000, "1000T"),
        (-1_500_000_000_000_000, "-1500T"),
        (1_000_500_000_000_000, "1000.5T"),
    ],
)
def test_format_number_uses_magnitude_and_omits_whole_decimals(value, expected):
    assert format_number(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "zero"),
        (1, "one"),
        (9, "nine"),
        (10, "ten"),
        (11, "eleven"),
        (19, "nineteen"),
        (20, "twenty"),
        (21, "twenty one"),
        (42, "forty two"),
        (80, "eighty"),
        (99, "ninety nine"),
    ],
)
def test_number_to_words(value, expected):
    assert string_utils.number_to_words(value) == expected


@pytest.mark.parametrize("value", [-100, -1, 100, 101])
def test_number_to_words_rejects_out_of_range_values(value):
    with pytest.raises(ValueError, match="between 0 and 99"):
        string_utils.number_to_words(value)


def test_random_string_uses_default_length_and_alphabet(monkeypatch):
    choice = Mock(side_effect=list("ABC123"))
    monkeypatch.setattr(string_utils.random, "choice", choice)

    assert string_utils.get_random_string() == "ABC123"
    assert choice.call_count == 6
    assert all(call.args == ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",) for call in choice.call_args_list)


def test_random_string_uses_custom_alphabet(monkeypatch):
    choice = Mock(side_effect=list("abba"))
    monkeypatch.setattr(string_utils.random, "choice", choice)

    assert string_utils.get_random_string(size=4, chars="ab") == "abba"
    assert choice.call_count == 4
    assert all(call.args == ("ab",) for call in choice.call_args_list)


def test_zero_length_random_string_does_not_sample(monkeypatch):
    choice = Mock()
    monkeypatch.setattr(string_utils.random, "choice", choice)

    assert string_utils.get_random_string(size=0, chars="") == ""
    choice.assert_not_called()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", ""), ("hello", "Hello"), ("hello_world", "Hello World"), ("HELLO_WORLD", "Hello World"), ("agent_42_tools", "Agent 42 Tools")],
)
def test_snake_to_title(value, expected):
    assert string_utils.snake_to_title(value) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (1, "1 second"),
        (2, "2 seconds"),
        (59, "59 seconds"),
        (60, "1 minute"),
        (120, "2 minutes"),
        (3600, "1 hour"),
        (7200, "2 hours"),
        (3661, "1 hour, 1 minute, 1 second"),
        (86400, "1 day"),
        (172800, "2 days"),
        (604800, "1 week"),
        (1_209_600, "2 weeks"),
        (694861, "1 week, 1 day, 1 hour, 1 minute, 1 second"),
    ],
)
def test_seconds_to_readable_decomposes_and_pluralizes(seconds, expected):
    assert time_utils.seconds_to_readable(seconds) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2024-02-29", datetime(2024, 2, 29, tzinfo=timezone.utc)),
        ("2024-02-29T13:45:12Z", datetime(2024, 2, 29, 13, 45, 12, tzinfo=timezone.utc)),
        (date(2024, 2, 29), datetime(2024, 2, 29, tzinfo=timezone.utc)),
        (datetime(2024, 2, 29, 13, 45, 12), datetime(2024, 2, 29, 13, 45, 12, tzinfo=timezone.utc)),
        (datetime(2024, 2, 29, tzinfo=timezone.utc), datetime(2024, 2, 29, tzinfo=timezone.utc)),
    ],
)
def test_to_datetime_normalizes_supported_inputs_to_utc(value, expected):
    assert time_utils.to_datetime(value) == expected


@pytest.mark.parametrize(
    "value",
    ["2024-03-01T00:30:00+02:00", datetime(2024, 3, 1, 0, 30, tzinfo=timezone(timedelta(hours=2)))],
)
def test_to_datetime_preserves_instant_when_converting_offset(value):
    actual = time_utils.to_datetime(value)
    assert actual == datetime(2024, 2, 29, 22, 30, tzinfo=timezone.utc)
    assert actual.tzinfo is timezone.utc


def test_to_datetime_converts_negative_offset_across_day_boundary():
    assert time_utils.to_datetime("2024-02-29T23:30:00-05:00") == datetime(2024, 3, 1, 4, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("value", ["not-a-date", "2024-02-30", ""])
def test_to_datetime_reports_invalid_strings(value):
    with pytest.raises(ValueError, match="Unable to parse"):
        time_utils.to_datetime(value)


@pytest.mark.parametrize(
    ("target", "expected"),
    [("2024-02-28", -1), ("2024-02-29", 0), ("2024-03-01", 1), (date(2025, 2, 28), 365), ("2024-03-01T00:30:00+02:00", 0)],
)
def test_days_from_now_uses_calendar_days_and_handles_leap_day(monkeypatch, target, expected):
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz is timezone.utc
            return cls(2024, 2, 29, 23, 59, 59, tzinfo=tz)

    monkeypatch.setattr(time_utils, "datetime", FrozenDatetime)
    assert time_utils.days_from_now(target) == expected


@pytest.mark.parametrize(
    ("ranges", "expected"),
    [
        ([], 0.0),
        ([("2024-02-28", "2024-03-01")], 2.0),
        ([("2024-01-01", "2024-01-01")], 0.0),
        ([("2024-01-03", "2024-01-01")], -2.0),
        ([("2024-01-01", "2024-01-02"), ("2024-01-01", "2024-01-03"), ("2024-01-01", "2024-01-03")], 1.67),
    ],
)
def test_calculate_average_days(ranges, expected):
    assert time_utils.calculate_average_days(ranges) == expected


def test_calculate_average_days_propagates_invalid_dates():
    with pytest.raises(ValueError, match="Unable to parse"):
        time_utils.calculate_average_days([("invalid", "2024-01-01")])
