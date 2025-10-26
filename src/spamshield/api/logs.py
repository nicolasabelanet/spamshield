import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
            "time": int(time.time() * 1000),
        }

        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)

        for k, v in record.__dict__.items():
            if k not in (
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
            ):
                base[k] = v
        return json.dumps(base, ensure_ascii=False)


def configure_logging(json_logs: bool = True, level: str = "INFO"):
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_logs else logging.Formatter("%(levelname)s %(message)s")
    )
    root.handlers[:] = [handler]
