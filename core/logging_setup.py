"""Structured logging with credential redaction.

Requirement 25 asks that logs make it possible to reconstruct what the system
believed happened. Requirement 9 forbids credentials appearing anywhere. Those
two pull in opposite directions, so redaction is enforced by a logging filter
rather than by remembering not to log secrets at each call site.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path

_SECRET_ENV_VARS = (
    "FYERS_ACCESS_TOKEN",
    "FYERS_SECRET_KEY",
    "FYERS_CLIENT_ID",
)

# Matches "Authorization: <id>:<token>" style header values and bare tokens
# that appear in URLs as query parameters.
_PATTERNS = (
    re.compile(r"(auth_code=)[^&\s\"']+", re.IGNORECASE),
    re.compile(r"(access_token=)[^&\s\"']+", re.IGNORECASE),
    re.compile(r"(secret_key=)[^&\s\"']+", re.IGNORECASE),
    re.compile(r"(appIdHash\"?\s*[:=]\s*\"?)[0-9a-f]{64}", re.IGNORECASE),
)

REDACTED = "***REDACTED***"


class RedactSecretsFilter(logging.Filter):
    """Removes known secret values and secret-shaped patterns from log records.

    Two mechanisms, because either alone is insufficient: exact-value matching
    catches secrets that are configured, and pattern matching catches secrets
    that arrive from elsewhere (an API response, a URL built at runtime).
    """

    def __init__(self) -> None:
        super().__init__()
        self._literals = [
            v for var in _SECRET_ENV_VARS if (v := os.getenv(var)) and len(v) >= 8
        ]

    def _scrub(self, text: str) -> str:
        for literal in self._literals:
            text = text.replace(literal, REDACTED)
        for pattern in _PATTERNS:
            text = pattern.sub(rf"\g<1>{REDACTED}", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._scrub(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._scrub(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            else:
                record.args = tuple(
                    self._scrub(a) if isinstance(a, str) else a for a in record.args
                )
        return True


def new_run_id() -> str:
    """Short unique id correlating every log line from one execution."""
    return uuid.uuid4().hex[:12]


def setup_logging(
    log_dir: Path,
    *,
    run_id: str | None = None,
    level: int = logging.INFO,
    console: bool = True,
) -> str:
    """Configure root logging to a per-run file, returning the run id."""
    run_id = run_id or new_run_id()
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt=f"%(asctime)s | {run_id} | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    redactor = RedactSecretsFilter()

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(log_dir / f"run_{run_id}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.addFilter(redactor)
    root.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        stream.addFilter(redactor)
        root.addHandler(stream)

    return run_id
