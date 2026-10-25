"""Checksums of frozen PR #12/#13/#14 artifacts. Do not rewrite those dirs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

PROTECTED_PATHS = (
    "configs/v3_eval_correct.frozen.yaml",
    "configs/v3_cov_chronos.frozen.yaml",
    "configs/v3_share_repl.frozen.yaml",
    "docs/v3/eval_correct_RESULTS.md",
    "docs/v3/cov_chronos_RESULTS.md",
    "docs/v3/share_repl_RESULTS.md",
    "docs/v3/share_repl_protocol.md",
    "runs/v3_eval_correct/RESULTS.md",
    "runs/v3_cov_chronos/RESULTS.md",
    "runs/v3_cov_chronos/SUMMARY.json",
    "runs/v3_share_repl/RESULTS.md",
    "runs/v3_share_repl/SUMMARY.json",
    "runs/v3_share_repl/freeze_lock.json",
    "configs/v3_share_abstain.frozen.yaml",
    "docs/v3/share_abstain_protocol.md",
    "docs/v3/share_abstain_RESULTS.md",
    "docs/v3/share_repl_ERRATUM.md",
    "docs/v3/selective_sharing_audit.md",
    "runs/v3_share_abstain/RESULTS.md",
    "runs/v3_share_abstain/SUMMARY.json",
    "runs/v3_share_abstain/freeze_lock.json",
    "configs/v3_share_abstain.approved.yaml",
    "docs/v3/share_abstain_run_RESULTS.md",
    "runs/v3_share_abstain_run/RESULTS.md",
    "runs/v3_share_abstain_run/SUMMARY.json",
    "runs/v3_share_abstain_run/freeze_lock.json",
)

PROTECTED_DIRS = (
    "runs/v3_eval_correct",
    "runs/v3_cov_chronos",
    "runs/v3_share_repl",
    "runs/v3_share_abstain",
    "runs/v3_share_abstain_run",
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_protected(root: Path | None = None) -> dict[str, Any]:
    base = Path(root or ".")
    files: dict[str, Any] = {}
    missing: list[str] = []
    for rel in PROTECTED_PATHS:
        p = base / rel
        if not p.is_file():
            missing.append(rel)
            files[rel] = None
            continue
        files[rel] = {"sha256": file_sha256(p), "bytes": p.stat().st_size}
    dirs: dict[str, Any] = {}
    for rel in PROTECTED_DIRS:
        p = base / rel
        if not p.is_dir():
            dirs[rel] = None
            missing.append(rel)
            continue
        digest = hashlib.sha256()
        n = 0
        for fp in sorted(p.rglob("*")):
            if not fp.is_file():
                continue
            rel_f = str(fp.relative_to(p))
            if "/hf_cache/" in f"/{rel_f}/" or rel_f.endswith("/pipeline/model.safetensors"):
                continue
            digest.update(rel_f.encode())
            digest.update(file_sha256(fp).encode())
            n += 1
        dirs[rel] = {"sha256": digest.hexdigest(), "n_files": n}
    return {"files": files, "directories": dirs, "missing": missing}


def compare_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    changed: list[str] = []
    for key in ("files", "directories"):
        a, b = before.get(key) or {}, after.get(key) or {}
        for name in sorted(set(a) | set(b)):
            if a.get(name) != b.get(name):
                changed.append(f"{key}:{name}")
    return {"unchanged": not changed, "changed": changed}
