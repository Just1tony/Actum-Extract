"""Structured JSON logging with request/correlation ID propagation.

Every log line emitted by this module is a single JSON object, suitable for
direct ingestion into log aggregation platforms (Datadog, ELK, CloudWatch
Logs Insights, etc.) without a separate parsing stage.

Correlation IDs are threaded through the application using ``contextvars``
rather than being passed explicitly as function arguments. This means any
code path — including code several layers deep inside the extraction engine
— can log with the correct request context automatically, as long as it runs
within the async task that the middleware initialized.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, MutableMapping

from core.config import get_settings

# ``request_id_ctx`` holds the correlation ID for the currently executing
# asynchronous task. It is set by the ``RequestContextMiddleware`` in
# ``main.py`` at the start of every inbound HTTP request and automatically
# reset once that request completes, thanks to context isolation between
# asyncio tasks.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="unbound")


def generate_request_id() -> str:
    """Generates a new, URL-safe correlation identifier.

    Returns:
        A UUIDv4 hex string prefixed for easy visual grep-ability in logs.
    """
    return f"req_{uuid.uuid4().hex}"


class JSONFormatter(logging.Formatter):
    """Renders log records as single-line JSON objects.

    The formatter enriches every record with the active request ID pulled
    from ``request_id_ctx``, a high-resolution UTC timestamp, and any
    ``extra`` fields passed to the logging call (e.g. ``logger.info("msg",
    extra={"document_id": "..."})``).
    """

    _RESERVED_ATTRS: frozenset[str] = frozenset(
        {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message", "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        """Formats a ``LogRecord`` into a JSON string.

        Args:
            record: The standard library log record to serialize.

        Returns:
            A JSON-encoded string representing the log event.
        """
        payload: MutableMapping[str, Any] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }

        for key, value in record.__dict__.items():
            if key not in self._RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging() -> None:
    """Configures the root logger to emit structured JSON to stdout.

    This is idempotent: calling it multiple times (e.g. once from ``main``
    and once from a test fixture) will not duplicate handlers.
    """
    settings = get_settings()
    root_logger = logging.getLogger()

    if any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.setLevel(settings.log_level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level)

    # Silence overly chatty third-party loggers at INFO while keeping our
    # own application namespace verbose.
    for noisy_logger in ("httpx", "httpcore", "openai._base_client"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Returns a namespaced logger configured for structured JSON output.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A standard ``logging.Logger`` instance; formatting/handlers are
        managed globally by ``configure_logging``.
    """
    return logging.getLogger(name)
