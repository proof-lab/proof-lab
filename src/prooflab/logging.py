"""Structured logging for Proof Lab.

Usage::

    from prooflab.config import get_settings
    from prooflab.logging import configure_logging, get_logger

    configure_logging(get_settings())
    log = get_logger(__name__)
    log.info("Ready.", extra={"component": "data"})

In *development* mode (log.format = "text") logs are rendered with Rich for
human readability.  In *production* mode (log.format = "json") each record
is emitted as a single-line JSON object suitable for log aggregation systems.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prooflab.config import ProofLabSettings

# Third-party loggers that tend to be very noisy at low levels.
_SUPPRESSED_LOGGERS: tuple[str, ...] = (
    "urllib3",
    "httpx",
    "httpcore",
    "uvicorn.access",
    "multipart",
)

# Attributes present on every LogRecord that should not be forwarded as extras.
_STANDARD_LOG_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record.

    Required fields in every object:
        timestamp   – ISO 8601, UTC
        level       – DEBUG / INFO / WARNING / ERROR / CRITICAL
        logger      – full logger name
        message     – formatted log message
    Optional fields:
        exception   – formatted traceback (only when exc_info is present)
        Any extra key-value pairs the caller attached via the ``extra`` kwarg.
    """

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        # Forward any caller-supplied extra fields.
        for key, val in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS:
                payload[key] = val

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(settings: ProofLabSettings) -> None:
    """Configure the root logger and the prooflab namespace logger.

    Must be called once at application start-up (e.g. in the CLI entry point
    or the FastAPI lifespan hook) before any other logging calls are made.

    Args:
        settings: The resolved ProofLabSettings instance.
    """
    level: int = getattr(logging, settings.log.level.upper(), logging.INFO)

    handler: logging.Handler
    if settings.log.format == "json":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
    else:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, show_path=False)
        except ImportError:  # pragma: no cover
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress noisy third-party output.
    for name in _SUPPRESSED_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger scoped to the prooflab namespace.

    If *name* already starts with ``"prooflab"`` it is used as-is; otherwise
    it is prefixed with ``"prooflab."`` to keep the logger hierarchy clean.

    Args:
        name: Module name (typically ``__name__``).

    Returns:
        A :class:`logging.Logger` instance.
    """
    if not name.startswith("prooflab"):
        name = f"prooflab.{name}"
    return logging.getLogger(name)
