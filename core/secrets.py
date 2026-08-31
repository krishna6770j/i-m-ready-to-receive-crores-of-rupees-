"""Append-only, process-lifetime registry of secret literals.

This is the PRIMARY mechanism for knowing what a secret literal actually is.
Previously ``core.logging_setup`` rediscovered secrets by re-reading known
environment variables at scrub time; that only caught credentials that were
still current in the environment, and it excluded anything not shaped like an
env-var-sourced token (an auth code, an old rotated-out token still floating
around in a closure or a log buffer). Registering the literal the moment it is
known, and never forgetting it, closes both gaps.

Design choices, and why:

* Append-only, no eviction, no LRU: a secret that stops being the "current"
  one (say, after token rotation) can still appear in an in-flight log
  buffer, a retry, or a stack trace captured moments before rotation. Evicting
  the old value the instant a new one is registered would silently re-open
  the exact leak this registry exists to close.
* Any length, including one character, is accepted: a minimum-length rule is
  a guess about what secrets look like, and this project's guiding rule
  (established across Units 12/13A/13B) is that invented numeric thresholds
  standing in for real domain knowledge get removed, not added.
* NOT thread-safe by contract: ``register`` mutates a plain ``set`` with no
  lock. Every call site that registers a secret today (settings load, the
  FYERS auth flow) runs on a single thread. Adding locking here would be
  undocumented protection for a concurrency model this codebase does not
  have; if a concurrent registration path is ever introduced, add explicit
  locking then and update this docstring alongside it.
"""

from __future__ import annotations

REDACTED = "***REDACTED***"


class SecretRegistry:
    """Append-only store of secret literals that must always redact.

    NOT thread-safe: concurrent calls to ``register`` from multiple threads
    are not synchronised. See the module docstring for why.
    """

    def __init__(self) -> None:
        self._secrets: set[str] = set()

    def register(self, secret: str) -> None:
        """Record a secret literal. Idempotent -- re-registering is harmless.

        Rejects ``None`` and the empty string: neither is a secret value, and
        silently accepting them would mean ``scrub()`` does nothing useful
        for that "registration" while looking like it succeeded.
        """
        if secret is None:
            raise ValueError("cannot register None as a secret")
        if secret == "":
            raise ValueError("cannot register an empty string as a secret")
        self._secrets.add(secret)

    def scrub(self, text: str) -> str:
        """Replace every registered literal appearing in ``text``."""
        for secret in self._secrets:
            if secret in text:
                text = text.replace(secret, REDACTED)
        return text

    def __contains__(self, secret: str) -> bool:
        return secret in self._secrets

    def __len__(self) -> int:
        return len(self._secrets)


# Process-lifetime singleton. Deliberately module-level rather than
# constructed per call site: every part of the process (settings load, the
# auth flow, the logging formatter) must see the same accumulated set of
# secrets, and a secret registered by one must be redactable by all.
registry = SecretRegistry()


def register(secret: str) -> None:
    """Register a secret literal with the process-wide registry."""
    registry.register(secret)


def scrub(text: str) -> str:
    """Redact every literal registered with the process-wide registry."""
    return registry.scrub(text)
