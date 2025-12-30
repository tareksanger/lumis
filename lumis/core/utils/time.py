from datetime import date, datetime, timedelta, timezone
from typing import Sequence, Tuple, Union

from dateutil.parser import parse

def seconds_to_readable(seconds: int) -> str:
    """
    This function works by iterating over the periods list and using integer division and the
    modulus operator to calculate the number of each period. If the number of a period is greater
    than 1, it adds an 's' to the end of the period name to make it plural. Finally, it joins
    the strings with commas and returns the result.

    Examples:
      print(seconds_to_readable(60))  # Outputs: "1 minute"
      print(seconds_to_readable(3600))  # Outputs: "1 hour"
      print(seconds_to_readable(3661))  # Outputs: "1 hour, 1 minute, 1 second"

    """

    periods = [
        ("week", 60 * 60 * 24 * 7),
        ("day", 60 * 60 * 24),
        ("hour", 60 * 60),
        ("minute", 60),
        ("second", 1),
    ]

    strings = []
    for period_name, period_seconds in periods:
        if seconds >= period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            if period_value == 1:
                strings.append(f"{period_value} {period_name}")
            else:
                strings.append(f"{period_value} {period_name}s")

    return ", ".join(strings)


DateObject = Union[str, datetime, date]


def to_datetime(input: DateObject) -> datetime:
    """
    Convert a string, datetime, or date object to a datetime object.

    Args:
    input_string (Union[str, datetime, date]): The input to convert.

    Returns:
    datetime: The converted datetime object.

    Raises:
    ValueError: If the input string cannot be parsed into a valid datetime.
    """
    if isinstance(input, datetime):
        return input.replace(tzinfo=timezone.utc)
    elif isinstance(input, date):
        return datetime.combine(input, datetime.min.time(), tzinfo=timezone.utc)

    try:
        dt = parse(input)
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"Unable to parse '{input}' into a valid datetime")


def days_from_now(target_date: DateObject) -> int:
    """
    Calculate the number of days between now and a given date.

    Args:
    target_date (Union[str, date, datetime]): The target date. Can be a string in 'YYYY-MM-DD' format,
                                              a date object, or a datetime object.

    Returns:
    int: The number of days between now and the target date.
         Positive if the target date is in the future, negative if it's in the past.
    """
    now = datetime.now(timezone.utc).date()
    target_date = to_datetime(target_date).date()
    delta = target_date - now
    return delta.days


def calculate_average_days(date_ranges: Sequence[Tuple[str, str]]) -> float:
    """
    Calculate the average number of days between start and end dates in the given list of date ranges.

    Args:
    date_ranges (List[Tuple[str, str]]): A list of tuples, each containing start_date and end_date as strings in 'YYYY-MM-DD' format.

    Returns:
    float: The average number of days between the date ranges.
    """
    total_days = 0
    for start_date_str, end_date_str in date_ranges:
        start_date = to_datetime(start_date_str)
        end_date = to_datetime(end_date_str)
        days_difference = (end_date - start_date).days
        total_days += days_difference

    if len(date_ranges) > 0:
        average_days = total_days / len(date_ranges)
        return round(average_days, 2)
    else:
        return 0.0
