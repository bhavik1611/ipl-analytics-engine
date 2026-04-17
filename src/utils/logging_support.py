"""Shared logging setup for pipeline modules (aggregator, narrator).

Reads ``LOG_LEVEL`` from the environment (via ``python-dotenv`` / ``load_env``)
and configures module loggers for traceability without requiring a separate
application bootstrap.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Final

from src.config import load_env

_CONFIGURED_NAMES: Final[set[str]] = set()


def parse_log_level() -> int:
    """Resolve ``LOG_LEVEL`` to a ``logging`` numeric level.

    Returns:
        Logging level constant, defaulting to ``logging.INFO`` if unset or invalid.
    """

    load_env()
    raw = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, raw, logging.INFO)


def new_run_id() -> str:
    """Return a short unique id for correlating log lines within one pipeline run.

    Returns:
        Twelve-character hexadecimal string.
    """

    return uuid.uuid4().hex[:12]


def ensure_pipeline_logger(name: str) -> logging.Logger:
    """Ensure ``name`` logger honors ``LOG_LEVEL`` and can emit to stderr.

    If the root logger already has handlers (e.g. tests or host app), only the
    logger level is updated and messages propagate. Otherwise a single
    stderr ``StreamHandler`` is attached to this logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured ``logging.Logger`` instance.
    """

    level = parse_log_level()
    log = logging.getLogger(name)
    log.setLevel(level)
    for handler in log.handlers:
        handler.setLevel(level)
    if name in _CONFIGURED_NAMES:
        return log
    if logging.root.handlers:
        log.propagate = True
        _CONFIGURED_NAMES.add(name)
        return log
    if log.handlers:
        _CONFIGURED_NAMES.add(name)
        return log
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    log.addHandler(handler)
    log.propagate = False
    _CONFIGURED_NAMES.add(name)
    return log
