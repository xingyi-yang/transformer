"""Mock-tested Snowflake adapter: no network, no credentials, no real service."""

from __future__ import annotations

import logging
import re

import pandas as pd
import pytest

from send_bw.data.harmonize import harmonize
from send_bw.data.sources import SnowflakeSendSource, build_source
from send_bw.utils.security import assert_read_only_sql

DOMAINS = ["DM", "BW", "EX", "DS", "SUPPBW"]
SELECT_RE = re.compile(r'^SELECT (?P<cols>.+?) FROM "(?P<db>\w+)"\."(?P<sch>\w+)"\."(?P<tbl>\w+)"'
                       r'(?: AT\(TIMESTAMP => TO_TIMESTAMP_LTZ\(%\(as_of\)s\)\))?'
                       r'(?: WHERE "STUDYID" IN \((?P<ids>.*)\))?$')


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self.description = None
        self._rows: list[tuple] = []
        self.sfqid = None

    def execute(self, sql, params):
        assert_read_only_sql(sql)  # the mock refuses anything that is not a single SELECT
        self.db.executed.append((sql, dict(params)))
        self.sfqid = f"qid-{len(self.db.executed)}"
        m = SELECT_RE.match(sql)
        assert m, f"unexpected SQL shape: {sql}"
        df = self.db.tables[m["tbl"]]
        if m["cols"] == "COUNT(*) AS N":
            self.description = [("N",)]
            self._rows = [(len(df),)]
            return
        cols = [c.strip().strip('"') for c in m["cols"].split(",")]
        if m["ids"]:
            names = re.findall(r"%\((s\d+)\)s", m["ids"])
            df = df[df["STUDYID"].astype(str).isin([params[n] for n in names])]
        self.description = [(c,) for c in cols]
        self._rows = [tuple(r) for r in df.reindex(columns=cols).itertuples(index=False, name=None)]

    def fetchmany(self, n):
        out, self._rows = self._rows[:n], self._rows[n:]
        return out

    def close(self):
        pass


class FakeConnection:
    def __init__(self, tables):
        self.tables = tables
        self.executed: list[tuple[str, dict]] = []
        self.conn_cfg = None

    def cursor(self):
        return FakeCursor(self)


def _mock_db(local_source):
    tables, columns = {}, {}
    for d in DOMAINS:
        df = local_source.read_domain(d).drop(columns=["_SRC_FILE", "_SRC_ROW"])
        tables[f"T_{d}"] = df.reset_index(drop=True)
        columns[d] = list(df.columns)
    return FakeConnection(tables), columns


def _cfg(columns, **kw):
    return {"authorized": True, "database": "SEND_DB", "schema": "SEND",
            "tables": {d: f"T_{d}" for d in DOMAINS}, "columns": columns, "batch_size": 37,
            "connection": {"account": "env:SB_ACCT", "user": "env:SB_USER", "password": "env:SB_PW"}, **kw}


@pytest.fixture(scope="module")
def mock(synth):
    conn, columns = _mock_db(synth["source"])
    return conn, columns


def test_local_and_mock_snowflake_yield_equivalent_canonical_tables(synth, mapping, mock):
    conn, columns = mock
    src = SnowflakeSendSource(_cfg(columns), connection_factory=lambda c: conn,
                              local_metadata_dir=synth["info"]["metadata_dir"])
    t_sf, _ = harmonize(src, mapping)
    t_loc = synth["tables"]
    lineage = {"SRC_FILE", "SRC_ROW", "SOURCE_ID"}
    for name in t_loc.tables:
        a = t_loc[name].drop(columns=[c for c in lineage if c in t_loc[name].columns])
        b = t_sf[name].drop(columns=[c for c in lineage if c in t_sf[name].columns])
        assert list(a.columns) == list(b.columns), name
        keys = list(a.columns)
        a = a.astype(str).sort_values(keys).reset_index(drop=True)
        b = b.astype(str).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False, obj=name)
    assert all(q.startswith("SELECT ") for q, _ in conn.executed)
    assert src.extraction_log and all(e["query_id"] for e in src.extraction_log)


def test_disabled_unless_authorized(mock):
    _, columns = mock
    with pytest.raises(PermissionError):
        SnowflakeSendSource({**_cfg(columns), "authorized": False}, connection_factory=lambda c: None)
    with pytest.raises(PermissionError):
        build_source({"type": "snowflake", "snowflake": {**_cfg(columns), "authorized": False}})


def test_literal_credentials_rejected(mock):
    _, columns = mock
    cfg = _cfg(columns)
    cfg["connection"] = {**cfg["connection"], "password": "hunter2"}
    with pytest.raises(ValueError):
        SnowflakeSendSource(cfg, connection_factory=lambda c: None)


def test_queries_are_parameterized_projected_and_allowlisted(mock):
    conn, columns = mock
    src = SnowflakeSendSource(_cfg(columns, as_of_timestamp="2026-01-01 00:00"), connection_factory=lambda c: conn)
    sql, params = src.build_query("BW", ["STUDYID", "BWSTRESN"], ["S1'; DROP TABLE X; --", "S2"])
    assert "DROP" not in sql and "S1" not in sql  # values only travel as bound parameters
    assert params["s0"].startswith("S1'") and params["as_of"] == "2026-01-01 00:00"
    assert "*" not in sql and "AT(TIMESTAMP" in sql
    with pytest.raises(PermissionError):
        src.build_query("LB", ["STUDYID"], None)  # domain not allowlisted
    with pytest.raises(PermissionError):
        src.build_query("BW", ["STUDYID", "SECRET_COL"], None)  # column not allowlisted
    with pytest.raises(ValueError):
        src.build_query("BW", ['STUDYID" FROM X; --'], None)  # identifier injection


def test_study_filter_batches_and_fetchmany(synth, mock):
    conn, columns = mock
    src = SnowflakeSendSource(_cfg(columns, max_studies_per_query=2), connection_factory=lambda c: conn)
    ids = sorted(synth["tables"]["study_context"]["STUDYID"])[:5]
    n_before = len(conn.executed)
    df = src.read_domain("BW", study_ids=ids)
    assert len(conn.executed) - n_before == 3  # ceil(5 / 2) parameterized queries
    assert set(df["STUDYID"]) <= set(ids)
    local = synth["source"].read_domain("BW", study_ids=ids)
    assert len(df) == len(local)


def test_select_star_never_issued(mock):
    conn, _ = mock
    src = SnowflakeSendSource(_cfg({}), connection_factory=lambda c: conn)
    with pytest.raises(PermissionError):
        src.read_domain("BW")


def test_logs_redact_secrets(mock, monkeypatch, caplog):
    from send_bw.utils.security import RedactingFilter, register_secret

    conn, columns = mock
    monkeypatch.setenv("SB_PW", "Sup3rSecretPw")
    register_secret("Sup3rSecretPw")
    log = logging.getLogger("send_bw")
    filt = RedactingFilter()
    log.addFilter(filt)
    try:
        with caplog.at_level(logging.INFO, logger="send_bw"):
            src = SnowflakeSendSource(_cfg(columns), connection_factory=lambda c: conn)
            src.inspect()
            log.info("connection password=Sup3rSecretPw")
    finally:
        log.removeFilter(filt)
    assert "Sup3rSecretPw" not in caplog.text
    assert "count:BW" in caplog.text or src.extraction_log
