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


def _current_secret_literals() -> list[str]:
    """Read secret values from the environment AT CALL TIME.

    Deliberately not cached. Credentials are commonly loaded after logging is
    configured -- ``load_settings()`` calls ``load_dotenv()`` on first use --
    and a snapshot taken at construction would miss every secret introduced
    afterwards, silently logging it in clear text.
    """
    return [v for var in _SECRET_ENV_VARS if (v := os.getenv(var)) and len(v) >= 8]


def scrub(text: str) -> str:
    """Remove known secret values and secret-shaped patterns from a string."""
    for literal in _current_secret_literals():
        text = text.replace(literal, REDACTED)
    for pattern in _PATTERNS:
        text = pattern.sub(rf"\g<1>{REDACTED}", text)
    return text


class RedactingFormatter(logging.Formatter):
    """Formatter that scrubs the FULLY RENDERED log line.

    Redaction must happen here rather than in a Filter. Filters run during
    ``Logger.handle()``, before the formatter renders ``exc_info`` into
    ``record.exc_text`` -- so a filter can scrub the message and args but never
    the traceback. A secret inside an exception ("auth failed for <token>")
    would reach the log file untouched.

    Formatting last means every component is covered at once: message, args,
    exception message, traceback frames, and stack info.
    """

    def format(self, record: logging.LogRecord) -> str:
        return scrub(super().format(record))


class RedactSecretsFilter(logging.Filter):
    """Scrubs message and args on the record itself.

    Retained as defence in depth alongside RedactingFormatter, so that a
    handler configured without the formatter still gets partial protection.
    It cannot cover tracebacks -- see RedactingFormatter for why.
    """

    def _scrub(self, text: str) -> str:
        return scrub(text)

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

    fmt = RedactingFormatter(
        fmt=f"%(asctime)s | {run_id} | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    redactor = RedactSecretsFilter()

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        # Close as well as remove: a removed FileHandler still holds an open
        # descriptor until garbage collection, which leaks handles in a
        # long-running process and raises ResourceWarning under -W error.
        try:
            handler.close()
        except Exception:  # noqa: BLE001 - never let teardown break logging setup
            pass

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
