import json
import logging
import sys
import time
from typing import Final

# LoggingRecord attributes that we *don't* want to emit in structured logs.
# Everything else in `record.__dict__` is treated as contextual metadata
# (e.g. request_id, path, status, duration_ms, etc.).
EXCLUDED_LOG_FIELDS: Final[set[str]] = {
    "args",
    "msg",
    "levelname",
    "levelno",
    "msecs",
    "relativeCreated",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "name",
    "process",
    "processName",
    "thread",
    "threadName",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """
    Structured log formatter that emits JSON.

    This formatter:
      - normalizes common log fields (timestamp, level, logger name, message)
      - includes any extra attributes attached to the LogRecord (e.g., request_id)
      - serializes exceptions if present
      - writes timestamps in epoch milliseconds for easier correlation with metrics

    The output is a single-line JSON object per log event, suitable for
    ingestion by log processors (e.g. CloudWatch Logs, Loki, ELK, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a LogRecord into a structured JSON string.

        This method is called automatically by the logging framework for each
        log event. It extracts relevant metadata from the LogRecord, enriches it
        with contextual fields, and serializes the final structure as a compact
        JSON object suitable for log aggregation systems.

        The output includes:
        - Standard fields such as log level, logger name, message, and timestamp.
        - Serialized exception info (if present).
        - Any additional contextual fields passed via `extra={...}` in the
            logging call (e.g., request_id, path, duration_ms).
        - Excludes internal logging framework attributes (thread, filename, etc.)
            defined in `_EXCLUDED_LOG_FIELDS`.

        Parameters
        ----------
        record : logging.LogRecord
            The record object containing metadata about the emitted log event.

        Returns
        -------
        str
            A JSON-formatted string representing the structured log entry.
        """

        # Base structured fields always present.
        base = {
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
            "time_ms": int(time.time() * 1000),
        }

        # If an exception is attached to the record, include formatted traceback.
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)

        # Copy any additional contextual data that was passed via `extra={...}`
        # in the logging call. Filter out all the default LogRecord internals
        # we're not interested in.
        for key, value in record.__dict__.items():
            if key not in EXCLUDED_LOG_FIELDS:
                base[key] = value

        return json.dumps(base, ensure_ascii=False)


def configure_logging(json_logs: bool = True, level: str = "INFO"):
    """
    Configure root logger output for the SpamShield service.

    This:
      - Sets the global logging level.
      - Installs a single StreamHandler to stdout.
      - Uses JsonFormatter by default for structured logs, or a basic text formatter
        if `json_logs` is False.
      - Replaces any existing handlers on the root logger.

    Parameters
    ----------
    json_logs : bool
        If True, emit logs as structured JSON. If False, emit human-readable text.
    level : str
        Logging level to apply to the root logger (e.g. "DEBUG", "INFO", "WARNING").
    """

    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(message)s")
    )

    # Replace any existing handlers to avoid duplicate log lines.
    root.handlers[:] = [handler]
