"""Tests for date parsing logic in search_events().

These tests validate the date extraction without needing Elasticsearch
or any external services running.
"""
import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# We need to mock heavy dependencies before importing app
import sys

# Mock the heavy modules so we can import app without starting services
mock_modules = {
    'chainlit': MagicMock(),
    'sentence_transformers': MagicMock(),
    'psycopg2': MagicMock(),
    'psycopg2.pool': MagicMock(),
    'psycopg2.extras': MagicMock(),
    'elasticsearch': MagicMock(),
    'elasticsearch.helpers': MagicMock(),
    'scripts.evaluation': MagicMock(),
}

os.environ.setdefault("GROQ_API_KEY", "test-key")
with patch.dict(sys.modules, mock_modules):
    import app  # noqa: F401


class TestExtractDateRange:
    """Test the extract_date_range() helper function."""

    def test_today(self):
        from app import extract_date_range
        start, end = extract_date_range("free food today")
        today = datetime.now().strftime("%Y-%m-%d")
        assert start == today
        assert end == today

    def test_tomorrow(self):
        from app import extract_date_range
        start, end = extract_date_range("career fairs tomorrow")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        assert start == tomorrow
        assert end == tomorrow

    def test_this_weekend(self):
        from app import extract_date_range
        start, end = extract_date_range("events this weekend")
        assert start is not None
        assert end is not None
        # Weekend should be Sat-Sun
        from datetime import datetime as dt
        start_dt = dt.strptime(start, "%Y-%m-%d")
        end_dt = dt.strptime(end, "%Y-%m-%d")
        assert start_dt.weekday() == 5  # Saturday
        assert end_dt.weekday() == 6    # Sunday

    def test_this_month(self):
        from app import extract_date_range
        start, end = extract_date_range("events this month")
        now = datetime.now()
        assert start == now.replace(day=1).strftime("%Y-%m-%d")
        assert start is not None and end is not None

    def test_next_month(self):
        from app import extract_date_range
        start, end = extract_date_range("events next month")
        assert start is not None and end is not None

    def test_upcoming(self):
        from app import extract_date_range
        start, end = extract_date_range("upcoming events")
        today = datetime.now().strftime("%Y-%m-%d")
        assert start == today
        assert end is not None

    def test_specific_month_name(self):
        from app import extract_date_range
        start, end = extract_date_range("events in march")
        assert start is not None
        assert "03" in start  # March

    def test_no_date_returns_none(self):
        from app import extract_date_range
        start, end = extract_date_range("career fairs")
        assert start is None
        assert end is None

    def test_month_year_rollover(self):
        """If querying a month that has passed, should return next year."""
        from app import extract_date_range
        now = datetime.now()
        # Pick a month that's already passed
        if now.month > 1:
            past_month = "january"
            start, end = extract_date_range(f"events in {past_month}")
            if now.month > 1:  # January has passed
                assert str(now.year + 1) in start
