from __future__ import annotations

import logging

import pytest

from src.utils.logging_support import ensure_pipeline_logger, parse_log_level


def test_parse_log_level_defaults_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert parse_log_level() == int(logging.INFO)


def test_parse_log_level_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert parse_log_level() == int(logging.DEBUG)


def test_parse_log_level_invalid_falls_back_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_REAL_LEVEL")
    assert parse_log_level() == int(logging.INFO)


def test_ensure_pipeline_logger_returns_logger() -> None:
    log = ensure_pipeline_logger("src.utils.logging_support_test_logger")
    assert isinstance(log, logging.Logger)
    assert log.name == "src.utils.logging_support_test_logger"
