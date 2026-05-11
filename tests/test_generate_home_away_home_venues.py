"""Tests for multi-venue handling in ``team_home_venues.json``."""

from __future__ import annotations

import pytest

from src.scripts.generate_home_away_reports import (
    _home_venue_list_for_franchise,
    _pair_report_jobs,
)


def test_home_venue_list_single_string() -> None:
    """A string entry becomes a one-element list."""

    assert _home_venue_list_for_franchise("X", " Eden ") == ["Eden"]


def test_home_venue_list_multiple_strings() -> None:
    """A list entry is normalized and trimmed."""

    raw = ["  A  ", "B"]
    assert _home_venue_list_for_franchise("PBKS", raw) == ["A", "B"]


def test_home_venue_list_rejects_bad_types() -> None:
    """Invalid container types raise ``ValueError``."""

    with pytest.raises(ValueError, match="str or list"):
        _home_venue_list_for_franchise("T", 3)


def test_home_venue_list_rejects_empty() -> None:
    """Empty string or list raises ``ValueError``."""

    with pytest.raises(ValueError, match="Empty home venue string"):
        _home_venue_list_for_franchise("T", "   ")
    with pytest.raises(ValueError, match="Empty home venue list"):
        _home_venue_list_for_franchise("T", [])


def test_pair_report_jobs_multiplies_venues() -> None:
    """Each (home, away) pair expands to one job per home venue."""

    franchises = ["A", "B"]
    home_venues: dict[str, str | list[str]] = {
        "A": ["V1", "V2"],
        "B": "W",
    }
    jobs = _pair_report_jobs(franchises, home_venues)
    assert jobs == [
        ("A", "B", "V1"),
        ("A", "B", "V2"),
        ("B", "A", "W"),
    ]


def test_pair_report_jobs_missing_home_raises() -> None:
    """Missing ``home`` key in the venue map raises ``ValueError``."""

    with pytest.raises(ValueError, match="no entry"):
        _pair_report_jobs(["A", "B"], {"A": "V"})
