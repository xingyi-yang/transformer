"""Source adapters: SendSource interface, LocalSendSource, SnowflakeSendSource.

Adapters only return raw domain tables (original values, upper-cased column
names, provenance columns _SRC_FILE/_SRC_ROW) and curated metadata. They never
harmonize; that is done once, identically for every source, in harmonize.py.
"""

from __future__ import annotations

import abc
import datetime as dt
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from send_bw.utils.artifacts import get_logger
from send_bw.utils.security import (
    assert_read_only_sql,
    redact,
    resolve_env_reference,
    validate_identifier,
)

SUPPORTED_EXT = {".csv": "csv", ".parquet": "parquet", ".xpt": "xpt"}
METADATA_TABLES = ("study_metadata", "arm_regimen", "compound_reference")


class SourceError(RuntimeError):
    pass


class AmbiguousSourceError(SourceError):
    pass


class OptionalDependencyError(ImportError):
    pass


def normalize_columns(df: pd.DataFrame, where: str) -> pd.DataFrame:
    """Explicit case/whitespace normalization with conflict detection."""
    new = [str(c).strip().upper() for c in df.columns]
    seen: dict[str, str] = {}
    for orig, n in zip(df.columns, new, strict=True):
        if n in seen:
            raise AmbiguousSourceError(f"{where}: columns {seen[n]!r} and {orig!r} collide after normalization")
        seen[n] = str(orig)
    out = df.copy()
    out.columns = new
    return out


class SendSource(abc.ABC):
    source_id: str = "abstract"

    @abc.abstractmethod
    def inspect(self) -> dict[str, Any]: ...

    @abc.abstractmethod
    def read_domain(self, name: str, columns: list[str] | None = None,
                    study_ids: Iterable[str] | None = None) -> pd.DataFrame: ...

    @abc.abstractmethod
    def read_metadata(self) -> dict[str, Any]: ...

    def available_domains(self) -> list[str]:
        return sorted(self.inspect()["domains"])


def _read_file(path: Path, fmt: str) -> tuple[pd.DataFrame, dict[str, str]]:
    labels: dict[str, str] = {}
    if fmt == "csv":
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "xpt":
        try:
            import pyreadstat  # type: ignore
        except ImportError as e:  # pragma: no cover - depends on env
            raise OptionalDependencyError(
                "Reading SAS XPT requires the optional 'xpt' extra: pip install 'send-bw[xpt]' (pyreadstat)"
            ) from e
        df, meta = pyreadstat.read_xport(str(path))
        labels = dict(meta.column_names_to_labels or {})
    else:  # pragma: no cover
        raise SourceError(f"unsupported format {fmt}")
    return df, {str(k).upper(): v for k, v in labels.items() if v}


class LocalSendSource(SendSource):
    """Local per-study folders (root/<STUDY>/<domain>.<ext>) or combined (root/<domain>.<ext>)."""

    source_id = "local"

    def __init__(self, root: str | Path, layout: str = "per_study", metadata_dir: str | Path | None = None,
                 domain_aliases: dict[str, str] | None = None):
        self.root = Path(root)
        if layout not in {"per_study", "combined"}:
            raise ValueError("layout must be per_study or combined")
        self.layout = layout
        self.metadata_dir = Path(metadata_dir) if metadata_dir else None
        self.domain_aliases = {k.upper(): v.upper() for k, v in (domain_aliases or {}).items()}
        if not self.root.exists():
            raise SourceError(f"source root does not exist: {self.root}")
        self._index = self._discover()
        self.labels: dict[str, dict[str, str]] = {}

    def _domain_of(self, stem: str) -> str:
        d = stem.strip().upper()
        return self.domain_aliases.get(d, d)

    def _discover(self) -> dict[str, list[tuple[str | None, Path, str]]]:
        index: dict[str, list[tuple[str | None, Path, str]]] = defaultdict(list)
        folders: list[tuple[str | None, Path]]
        if self.layout == "per_study":
            folders = [(p.name, p) for p in sorted(self.root.iterdir()) if p.is_dir()]
        else:
            folders = [(None, self.root)]
        for folder_name, folder in folders:
            per_folder: dict[str, list[Path]] = defaultdict(list)
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXT:
                    per_folder[self._domain_of(f.stem)].append(f)
            for dom, files in per_folder.items():
                if len(files) > 1:
                    raise AmbiguousSourceError(
                        f"{folder}: domain {dom} provided by multiple files {[x.name for x in files]}"
                    )
                index[dom].append((folder_name, files[0], SUPPORTED_EXT[files[0].suffix.lower()]))
        return dict(index)

    def inspect(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.source_id, "root": str(self.root), "layout": self.layout,
                               "domains": {}, "warnings": []}
        study_folders: dict[str, set[str]] = defaultdict(set)
        for dom, entries in self._index.items():
            info = {"files": len(entries), "rows": 0, "columns": set(), "formats": set()}
            for folder, path, fmt in entries:
                df, _ = _read_file(path, fmt)
                df = normalize_columns(df, str(path))
                info["rows"] += len(df)
                info["columns"].update(df.columns)
                info["formats"].add(fmt)
                if "STUDYID" in df.columns:
                    for s in df["STUDYID"].dropna().astype(str).unique():
                        study_folders[s].add(folder or path.name)
            info["columns"] = sorted(info["columns"])
            info["formats"] = sorted(info["formats"])
            out["domains"][dom] = info
        dups = {s: sorted(f) for s, f in study_folders.items() if len(f) > 1}
        if dups:
            out["warnings"].append({"type": "study_in_multiple_folders", "studies": dups})
        out["n_studies"] = len(study_folders)
        out["metadata_tables"] = sorted(self.read_metadata().keys() - {"labels"}) if self.metadata_dir else []
        return out

    def read_domain(self, name: str, columns: list[str] | None = None,
                    study_ids: Iterable[str] | None = None) -> pd.DataFrame:
        dom = self._domain_of(name)
        entries = self._index.get(dom, [])
        wanted = {str(s) for s in study_ids} if study_ids is not None else None
        frames = []
        for folder, path, fmt in entries:
            df, labels = _read_file(path, fmt)
            df = normalize_columns(df, str(path))
            if labels:
                self.labels.setdefault(dom, {}).update(labels)
            if df.empty and len(df.columns) == 0:
                continue
            if "STUDYID" not in df.columns:
                raise SourceError(f"{path}: STUDYID column missing")
            df["STUDYID"] = df["STUDYID"].astype(str)
            if folder is not None:
                bad = set(df["STUDYID"].unique()) - {folder}
                if bad and not all(folder.startswith(b) for b in bad):
                    raise AmbiguousSourceError(f"{path}: STUDYID {sorted(bad)} conflicts with folder {folder}")
            df["_SRC_FILE"] = str(path.relative_to(self.root)).replace("\\", "/")
            df["_SRC_ROW"] = range(len(df))
            if wanted is not None:
                df = df[df["STUDYID"].isin(wanted)]
            if columns is not None:
                cols = [c.upper() for c in columns]
                missing = [c for c in cols if c not in df.columns]
                for c in missing:
                    df[c] = pd.NA
                df = df[cols + ["_SRC_FILE", "_SRC_ROW"]]
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=(columns or []) + ["STUDYID", "_SRC_FILE", "_SRC_ROW"])
        return pd.concat(frames, ignore_index=True)

    def read_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {"labels": self.labels}
        if self.metadata_dir is None:
            return meta
        for t in METADATA_TABLES:
            for ext, fmt in SUPPORTED_EXT.items():
                p = self.metadata_dir / f"{t}{ext}"
                if p.exists():
                    df, _ = _read_file(p, fmt)
                    meta[t] = normalize_columns(df, str(p))
                    break
        return meta


# --------------------------------------------------------------------------- Snowflake

class SnowflakeConfig:
    """Read-only Snowflake access settings. Credentials only via 'env:VAR' references."""

    def __init__(self, cfg: dict[str, Any]):
        self.authorized = bool(cfg.get("authorized", False))
        self.database = validate_identifier(cfg["database"])
        self.schema = validate_identifier(cfg["schema"])
        self.tables: dict[str, str] = {k.upper(): validate_identifier(v) for k, v in cfg["tables"].items()}
        self.columns: dict[str, list[str]] = {
            k.upper(): [validate_identifier(c.upper()) for c in v] for k, v in cfg.get("columns", {}).items()
        }
        self.metadata_tables: dict[str, str] = {
            k: validate_identifier(v) for k, v in cfg.get("metadata_tables", {}).items()
        }
        self.batch_size = int(cfg.get("batch_size", 50_000))
        self.max_studies_per_query = int(cfg.get("max_studies_per_query", 200))
        self.as_of = cfg.get("as_of_timestamp")
        if self.as_of is not None and not re.match(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$", str(self.as_of)):
            raise ValueError("as_of_timestamp must be ISO-like YYYY-MM-DD[ HH:MM[:SS]]")
        self.connection = {k: cfg.get("connection", {}).get(k) for k in
                           ("account", "user", "password", "authenticator", "role", "warehouse",
                            "private_key_file")}
        self.export_dir = cfg.get("authorized_export_dir")


def _default_connection_factory(conn_cfg: dict[str, Any]):  # pragma: no cover - requires real service
    try:
        import snowflake.connector  # type: ignore
    except ImportError as e:
        raise OptionalDependencyError(
            "Snowflake access requires the optional 'snowflake' extra: pip install 'send-bw[snowflake]'"
        ) from e
    kwargs = {k: resolve_env_reference(v) for k, v in conn_cfg.items() if v is not None}
    return snowflake.connector.connect(**kwargs)


class SnowflakeSendSource(SendSource):
    """Read-only, allowlisted, parameterized extraction. Disabled unless explicitly authorized."""

    source_id = "snowflake"

    def __init__(self, cfg: dict[str, Any], connection_factory: Callable[[dict[str, Any]], Any] | None = None,
                 local_metadata_dir: str | Path | None = None):
        self.cfg = SnowflakeConfig(cfg)
        if not self.cfg.authorized:
            raise PermissionError(
                "Snowflake source is disabled: set authorized: true only with explicit written authorization"
            )
        for k, v in self.cfg.connection.items():
            if v is not None and k != "account" and not str(v).startswith("env:"):
                raise ValueError(f"connection.{k} must be an env: reference, not a literal")
        self._factory = connection_factory or _default_connection_factory
        self._conn = None
        self.extraction_log: list[dict[str, Any]] = []
        self.local_metadata_dir = Path(local_metadata_dir) if local_metadata_dir else None

    def _connection(self):
        if self._conn is None:
            self._conn = self._factory(self.cfg.connection)
        return self._conn

    def _table_ref(self, table: str) -> str:
        return f'"{self.cfg.database}"."{self.cfg.schema}"."{validate_identifier(table)}"'

    def build_query(self, domain: str, columns: list[str], study_ids: list[str] | None) -> tuple[str, dict]:
        dom = domain.upper()
        if dom not in self.cfg.tables:
            raise PermissionError(f"domain {dom} is not in the allowlist")
        cols = [validate_identifier(c.upper()) for c in columns]
        allowed = self.cfg.columns.get(dom)
        if allowed is not None:
            bad = set(cols) - set(allowed)
            if bad:
                raise PermissionError(f"columns {sorted(bad)} not allowlisted for {dom}")
        sel = ", ".join(f'"{c}"' for c in cols)
        sql = f"SELECT {sel} FROM {self._table_ref(self.cfg.tables[dom])}"
        params: dict[str, Any] = {}
        if self.cfg.as_of:
            sql += " AT(TIMESTAMP => TO_TIMESTAMP_LTZ(%(as_of)s))"
            params["as_of"] = str(self.cfg.as_of)
        if study_ids is not None:
            names = []
            for i, s in enumerate(study_ids):
                params[f"s{i}"] = str(s)
                names.append(f"%(s{i})s")
            sql += f' WHERE "STUDYID" IN ({", ".join(names)})'
        assert_read_only_sql(sql)
        return sql, params

    def _execute(self, sql: str, params: dict, label: str) -> pd.DataFrame:
        assert_read_only_sql(sql)
        cur = self._connection().cursor()
        try:
            cur.execute(sql, params)
            cols = [d[0].upper() for d in cur.description]
            rows: list[tuple] = []
            while True:
                batch = cur.fetchmany(self.cfg.batch_size)
                if not batch:
                    break
                rows.extend(batch)
            qid = getattr(cur, "sfqid", None)
        finally:
            cur.close()
        self.extraction_log.append({"label": label, "query_id": qid, "rows": len(rows),
                                    "extracted_utc": dt.datetime.now(dt.UTC).isoformat(),
                                    "as_of": self.cfg.as_of})
        get_logger().info(redact(f"snowflake extract {label}: {len(rows)} rows, query_id={qid}"))
        return pd.DataFrame(rows, columns=cols)

    def inspect(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.source_id, "domains": {}, "warnings": [
            "Separate queries are not a single consistent snapshot unless as_of_timestamp is set."]}
        for dom in self.cfg.tables:
            sql = f"SELECT COUNT(*) AS N FROM {self._table_ref(self.cfg.tables[dom])}"
            df = self._execute(sql, {}, f"count:{dom}")
            out["domains"][dom] = {"rows": int(df.iloc[0, 0]) if len(df) else 0,
                                   "columns": self.cfg.columns.get(dom)}
        return out

    def read_domain(self, name: str, columns: list[str] | None = None,
                    study_ids: Iterable[str] | None = None) -> pd.DataFrame:
        dom = name.upper()
        cols = columns or self.cfg.columns.get(dom)
        if not cols:
            raise PermissionError(f"projected columns must be configured for {dom}; SELECT * is not used")
        cols = [c.upper() for c in cols]
        if "STUDYID" not in cols:
            cols = ["STUDYID"] + cols
        ids = None if study_ids is None else [str(s) for s in study_ids]
        frames = []
        chunks = [None] if ids is None else [ids[i:i + self.cfg.max_studies_per_query]
                                              for i in range(0, len(ids), self.cfg.max_studies_per_query)]
        for chunk in chunks:
            sql, params = self.build_query(dom, cols, chunk)
            frames.append(self._execute(sql, params, f"domain:{dom}"))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
        df = normalize_columns(df, f"snowflake:{dom}")
        df["STUDYID"] = df["STUDYID"].astype(str)
        df["_SRC_FILE"] = f"snowflake:{self.cfg.database}.{self.cfg.schema}.{self.cfg.tables[dom]}"
        df["_SRC_ROW"] = range(len(df))
        return df

    def read_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {"labels": {}}
        if self.local_metadata_dir is not None:
            meta.update({k: v for k, v in LocalSendSource.read_metadata(self._local_meta_proxy()).items()
                         if k != "labels"})
        for t, table in self.cfg.metadata_tables.items():
            cols = self.cfg.columns.get(t.upper())
            if not cols:
                raise PermissionError(f"projected columns must be configured for metadata table {t}")
            sel = ", ".join(f'"{validate_identifier(c)}"' for c in cols)
            sql = f"SELECT {sel} FROM {self._table_ref(table)}"
            meta[t] = normalize_columns(self._execute(sql, {}, f"metadata:{t}"), t)
        return meta

    def _local_meta_proxy(self):
        class _P:
            pass

        p = _P()
        p.metadata_dir = self.local_metadata_dir
        p.labels = {}
        return p


def build_source(cfg: dict[str, Any], connection_factory: Callable | None = None,
                 domain_aliases: dict[str, str] | None = None) -> SendSource:
    kind = cfg.get("type", "local")
    if kind == "local":
        return LocalSendSource(cfg["root"], cfg.get("layout", "per_study"), cfg.get("metadata_dir"),
                               domain_aliases=domain_aliases)
    if kind == "snowflake":
        return SnowflakeSendSource(cfg["snowflake"], connection_factory=connection_factory,
                                   local_metadata_dir=cfg.get("metadata_dir"))
    raise ValueError(f"unknown source type {kind}")
