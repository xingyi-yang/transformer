"""Secret redaction, identifier validation and read-only SQL guards."""

from __future__ import annotations

import logging
import os
import re

SECRET_KEY_PATTERN = re.compile(
    r"(?i)(password|passwd|pwd|secret|token|private[_-]?key|api[_-]?key|authenticator)(\s*[=:]\s*)([^\s,;'\"}]+)"
)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]{0,254}$")
_FORBIDDEN_SQL = re.compile(
    r"(?i)\b(insert|update|delete|merge|create|alter|drop|truncate|grant|revoke|copy|put|remove|call|"
    r"execute|use|set|unset|undrop|comment|begin|commit|rollback)\b"
)

_registered_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Register a runtime secret value (e.g. resolved from an env var) for redaction."""
    if value and len(value) >= 4:
        _registered_secrets.add(value)


def redact(text: str) -> str:
    out = SECRET_KEY_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}***REDACTED***", str(text))
    for s in _registered_secrets:
        out = out.replace(s, "***REDACTED***")
    return out


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def validate_identifier(name: str) -> str:
    """Accept only simple unquoted SQL identifiers; returned value is safe to double-quote."""
    if not isinstance(name, str) or not _IDENT.match(name):
        raise ValueError(f"Unsafe or invalid SQL identifier: {name!r}")
    return name


def assert_read_only_sql(sql: str) -> None:
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        raise PermissionError("Multiple SQL statements are not allowed")
    if not re.match(r"(?is)^\s*select\b", stripped):
        raise PermissionError("Only SELECT statements are allowed")
    if _FORBIDDEN_SQL.search(stripped):
        raise PermissionError("SQL contains a forbidden keyword for read-only access")


def resolve_env_reference(ref: str | None) -> str | None:
    """Resolve 'env:VAR' references. Literal values are rejected to keep secrets out of configs."""
    if ref is None:
        return None
    if not isinstance(ref, str) or not ref.startswith("env:"):
        raise ValueError("Credentials must be given as 'env:VARIABLE_NAME' references, never literal values")
    value = os.environ.get(ref[4:])
    register_secret(value)
    return value
