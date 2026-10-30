"""Provenance helpers — record which code, config and inputs produced a result.

Every metrics/report JSON written by a script should go through `write_json` so that numbers in the
README and docs can be traced back to a git commit and a config hash.
"""
import hashlib
import json
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parents[2]


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks (safe for large rasters)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(cfg: dict) -> str:
    """Short, order-independent hash of a config dict."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def git_state(repo: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Current commit SHA and whether the tree has uncommitted changes (None outside a git repo)."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    sha = run("rev-parse", "--short", "HEAD")
    status = run("status", "--porcelain")
    return {"git_sha": sha, "git_dirty": None if status is None else bool(status)}


def provenance(
    cfg: dict | None = None, inputs: Iterable[str | Path] = (), seed: int | None = None
) -> dict[str, Any]:
    """Provenance record: UTC timestamp, git state, config hash, seed and input-file hashes."""
    info: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_state(),
        "seed": seed,
    }
    if cfg is not None:
        info["config_hash"] = config_hash(cfg)
    hashes = {str(p): file_sha256(p) for p in inputs}
    if hashes:
        info["inputs_sha256"] = hashes
    return info


def write_json(path: str | Path, payload: dict, info: dict[str, Any] | None = None) -> Path:
    """Write `payload` as pretty JSON, adding a "provenance" key (computed if `info` is None)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**payload, "provenance": info if info is not None else provenance()}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return path
