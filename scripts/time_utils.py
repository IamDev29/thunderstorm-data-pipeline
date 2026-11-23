import datetime
from typing import Dict, Union
from zoneinfo import ZoneInfo

# TAI93 epoch definition: 1993-01-01 00:00:00 UTC
TAI93_BASE_UTC = datetime.datetime(1993, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)

# In the 2023 epoch, exactly 10 leap seconds were inserted between 1993 and 2023.
# (1993-07-01, 1994-07-01, 1996-01-01, 1997-07-01, 1999-01-01,
#  2005-12-31, 2008-12-31, 2012-06-30, 2015-06-30, 2016-12-31).
# TAI93 = UTC_seconds_since_1993 + 10.0
# Verified across dataset orbit_summary_UTC_start and orbit_summary_TAI93_start.
TAI93_TO_UTC_OFFSET_SECONDS = 10.0

IST_TZ = ZoneInfo("Asia/Kolkata")


def tai93_to_utc_datetime(tai93_seconds: float) -> datetime.datetime:
    """
    Convert LIS TAI93 seconds (seconds since 1993-01-01 00:00:00 TAI)
    into standard timezone-aware UTC datetime with microsecond precision.
    """
    utc_seconds = float(tai93_seconds) - TAI93_TO_UTC_OFFSET_SECONDS
    return TAI93_BASE_UTC + datetime.timedelta(seconds=utc_seconds)


def format_observation_timestamps(
    tai93_seconds: float,
    ist_tz: ZoneInfo = IST_TZ,
) -> Dict[str, Union[str, int]]:
    """
    Generate normalized timestamp fields required by the science dataset:
    - timestamp_utc (ISO-8601 UTC with subsecond precision)
    - timestamp_ist (ISO-8601 IST with subsecond precision)
    - date_utc (YYYY-MM-DD)
    - date_ist (YYYY-MM-DD)
    - hour_utc (0-23)
    - hour_ist (0-23)
    """
    dt_utc = tai93_to_utc_datetime(tai93_seconds)
    dt_ist = dt_utc.astimezone(ist_tz)

    # Format UTC string as ISO-8601 with Z
    # E.g. 2023-01-01T02:32:14.300000Z
    iso_utc = dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    iso_ist = dt_ist.isoformat()

    return {
        "timestamp_utc": iso_utc,
        "timestamp_ist": iso_ist,
        "date_utc": dt_utc.strftime("%Y-%m-%d"),
        "date_ist": dt_ist.strftime("%Y-%m-%d"),
        "hour_utc": dt_utc.hour,
        "hour_ist": dt_ist.hour,
    }
