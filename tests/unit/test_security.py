from __future__ import annotations

import logging

import pytest

from send_bw.utils.security import (
    RedactingFilter,
    assert_read_only_sql,
    redact,
    register_secret,
    resolve_env_reference,
    validate_identifier,
)
from send_bw.utils.seeds import SeedStream


def test_redaction_patterns_and_registered_secrets():
    assert "hunter2" not in redact("password=hunter2 user=x")
    assert "abc123tok" not in redact("token: abc123tok")
    register_secret("S3cr3tValue")
    assert "S3cr3tValue" not in redact("connecting with S3cr3tValue now")


def test_logging_filter_redacts(caplog):
    logger = logging.getLogger("t_redact")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO, logger="t_redact"):
        logger.info("api_key=XYZ987654 ok")
    assert "XYZ987654" not in caplog.text


@pytest.mark.parametrize("sql", ["DELETE FROM t", "SELECT 1; DROP TABLE x", "select * from t where a=1 or 1=1; grant",
                                 "CREATE TABLE x AS SELECT 1", "WITH a AS (SELECT 1) INSERT INTO t SELECT * FROM a"])
def test_read_only_guard_rejects(sql):
    with pytest.raises(PermissionError):
        assert_read_only_sql(sql)


def test_read_only_guard_accepts_select():
    assert_read_only_sql('SELECT "A" FROM "DB"."S"."T" WHERE "STUDYID" IN (%(s0)s)')


@pytest.mark.parametrize("bad", ['x"; drop', "1abc", "a b", "", "a;b"])
def test_identifier_validation(bad):
    with pytest.raises(ValueError):
        validate_identifier(bad)


def test_literal_credentials_rejected(monkeypatch):
    with pytest.raises(ValueError):
        resolve_env_reference("plaintext-password")
    monkeypatch.setenv("SB_TEST_PW", "fromenvpw")
    assert resolve_env_reference("env:SB_TEST_PW") == "fromenvpw"
    assert "fromenvpw" not in redact("pw is fromenvpw")


def test_seed_streams_stable_and_independent():
    s = SeedStream(1)
    assert s.seed("a", 1) == SeedStream(1).seed("a", 1)
    assert s.seed("a", 1) != s.seed("a", 2)
    assert s.rng("x").normal() == SeedStream(1).rng("x").normal()
