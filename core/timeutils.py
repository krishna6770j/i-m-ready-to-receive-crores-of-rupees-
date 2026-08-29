"""Timezone-aware time handling for Indian market data.

Every timestamp inside this project is tz-aware and expressed in IST
(Asia/Kolkata). Naive timestamps are rejected rather than silently localised,
because a naive timestamp that is silently assumed to be IST when it is
actually UTC shifts data by 5h30m and corrupts every downstream result.

FYERS returns candle timestamps as epoch seconds. Per the FYERS Data API
knowledge base, "The timestamp provided for each candle indeed marks the
beginning of that candle's time interval" -- so a candle stamped 09:15 covers
09:15:00 to 09:15:59 inclusive. This project preserves that convention
everywhere: a candle's timestamp is its OPEN time.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
IST_NAME = "Asia/Kolkata"


class NaiveTimestampError(ValueError):
    """Raised when a timestamp lacks timezone information."""


def epoch_to_ist(epoch_seconds: int | float) -> datetime:
    """Convert an epoch-seconds value to a tz-aware IST datetime."""
    return datetime.fromtimestamp(float(epoch_seconds), tz=IST)


def epoch_series_to_ist(values) -> pd.Series:
    """Convert a sequence of epoch seconds to a tz-aware IST datetime Series.

    Uses UTC as the intermediate representation because epoch seconds are
    unambiguously UTC-based; converting afterwards avoids any DST or offset
    guesswork. India has no DST, but doing it correctly costs nothing.
    """
    return pd.to_datetime(pd.Series(values), unit="s", utc=True).dt.tz_convert(IST_NAME)


def ensure_ist(ts: datetime) -> datetime:
    """Return ``ts`` converted to IST, rejecting naive datetimes."""
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise NaiveTimestampError(
            f"Naive datetime {ts!r} rejected. Attach a timezone explicitly; "
            "this project never assumes a timezone for naive input."
        )
    return ts.astimezone(IST)


def is_tz_aware_ist(series: pd.Series) -> bool:
    """True if a pandas Series is tz-aware and localised to IST."""
    tz = getattr(series.dtype, "tz", None)
    if tz is None:
        return False
    return str(tz) == IST_NAME


def ist_datetime(d: date, t: time) -> datetime:
    """Build a tz-aware IST datetime from a date and a wall-clock time."""
    return datetime(d.year, d.month, d.day, t.hour, t.minute, t.second, tzinfo=IST)


def to_api_date(d: date) -> str:
    """Format a date as 'yyyy-mm-dd'.

    This is the format FYERS expects when ``date_format=1``, per the
    ``FyersModel.history`` docstring in fyers-apiv3 3.1.16.
    """
    return d.strftime("%Y-%m-%d")
