"""
Unit Tests for Member 2 - Temporal Analysis Module
Validates date and time parsing, boundary conditions, and feature engineering.
"""

import unittest
from datetime import datetime

def parse_acq_time(acq_time):
    """
    Safely parses NASA FIRMS acq_time (integer or string representation of HHMM).
    Returns (valid: bool, formatted_time: str, hour: int, minute: int).
    """
    if acq_time is None:
        return False, None, None, None
    try:
        # Normalize to 4-character string with leading zeros
        if isinstance(acq_time, (int, float)):
            val = int(acq_time)
            if val < 0 or val > 2359:
                return False, None, None, None
            time_str = f"{val:04d}"
        else:
            time_str = str(acq_time).strip().zfill(4)
            if len(time_str) != 4 or not time_str.isdigit():
                return False, None, None, None
        
        hour = int(time_str[:2])
        minute = int(time_str[2:])
        
        # Validate ranges
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return True, time_str, hour, minute
        return False, None, None, None
    except Exception:
        return False, None, None, None

def parse_acq_date(acq_date_str):
    """
    Safely parses NASA FIRMS acq_date string (YYYY-MM-DD).
    Returns (valid: bool, year: int, month: int, day: int, year_month: str).
    """
    if not acq_date_str or not isinstance(acq_date_str, str):
        return False, None, None, None, None
    try:
        dt = datetime.strptime(acq_date_str.strip(), "%Y-%m-%d")
        return True, dt.year, dt.month, dt.day, f"{dt.year:04d}-{dt.month:02d}"
    except ValueError:
        return False, None, None, None, None

def get_severity_tier(frp):
    """
    Maps Fire Radiative Power (MW) to WARD operational tier.
    """
    if frp is None or frp < 0:
        return "Unknown"
    if frp >= 50.0:
        return "Extreme"
    elif frp >= 20.0:
        return "High"
    elif frp >= 5.0:
        return "Moderate"
    else:
        return "Low"

class TestTemporalParsing(unittest.TestCase):
    def test_acq_time_standard_integer(self):
        valid, time_str, hour, minute = parse_acq_time(1345)
        self.assertTrue(valid)
        self.assertEqual(time_str, "1345")
        self.assertEqual(hour, 13)
        self.assertEqual(minute, 45)

    def test_acq_time_leading_zero_integer(self):
        # In FIRMS, 0053 is often stored as integer 53
        valid, time_str, hour, minute = parse_acq_time(53)
        self.assertTrue(valid)
        self.assertEqual(time_str, "0053")
        self.assertEqual(hour, 0)
        self.assertEqual(minute, 53)

    def test_acq_time_single_digit_integer(self):
        # e.g., 0005 stored as 5
        valid, time_str, hour, minute = parse_acq_time(5)
        self.assertTrue(valid)
        self.assertEqual(time_str, "0005")
        self.assertEqual(hour, 0)
        self.assertEqual(minute, 5)

    def test_acq_time_string_with_leading_zero(self):
        valid, time_str, hour, minute = parse_acq_time("0053")
        self.assertTrue(valid)
        self.assertEqual(time_str, "0053")
        self.assertEqual(hour, 0)
        self.assertEqual(minute, 53)

    def test_acq_time_midnight(self):
        valid, time_str, hour, minute = parse_acq_time(0)
        self.assertTrue(valid)
        self.assertEqual(time_str, "0000")
        self.assertEqual(hour, 0)
        self.assertEqual(minute, 0)

    def test_acq_time_end_of_day(self):
        valid, time_str, hour, minute = parse_acq_time(2359)
        self.assertTrue(valid)
        self.assertEqual(time_str, "2359")
        self.assertEqual(hour, 23)
        self.assertEqual(minute, 59)

    def test_acq_time_invalid_values(self):
        # Invalid hour: 2400
        valid, _, _, _ = parse_acq_time(2400)
        self.assertFalse(valid)
        # Invalid minute: 1260
        valid, _, _, _ = parse_acq_time(1260)
        self.assertFalse(valid)
        # Negative
        valid, _, _, _ = parse_acq_time(-10)
        self.assertFalse(valid)
        # Non-numeric string
        valid, _, _, _ = parse_acq_time("ABCD")
        self.assertFalse(valid)
        # None
        valid, _, _, _ = parse_acq_time(None)
        self.assertFalse(valid)

    def test_acq_date_valid(self):
        valid, year, month, day, ym = parse_acq_date("2025-10-12")
        self.assertTrue(valid)
        self.assertEqual(year, 2025)
        self.assertEqual(month, 10)
        self.assertEqual(day, 12)
        self.assertEqual(ym, "2025-10")

    def test_acq_date_invalid(self):
        valid, _, _, _, _ = parse_acq_date("invalid-date")
        self.assertFalse(valid)
        valid, _, _, _, _ = parse_acq_date("2025-02-31")
        self.assertFalse(valid)
        valid, _, _, _, _ = parse_acq_date(None)
        self.assertFalse(valid)

    def test_severity_tier_mapping(self):
        self.assertEqual(get_severity_tier(2.5), "Low")
        self.assertEqual(get_severity_tier(5.0), "Moderate")
        self.assertEqual(get_severity_tier(19.99), "Moderate")
        self.assertEqual(get_severity_tier(20.0), "High")
        self.assertEqual(get_severity_tier(49.99), "High")
        self.assertEqual(get_severity_tier(50.0), "Extreme")
        self.assertEqual(get_severity_tier(250.0), "Extreme")

if __name__ == "__main__":
    unittest.main()
