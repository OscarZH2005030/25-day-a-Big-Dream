"""Content addressed Task 2 cache. Paths never participate in identity.

Every payload is published atomically and has a SHA256 sidecar. A broken or
interrupted entry is a cache miss, never a source of silently changed labels.
No Task 1 cache is opened or written by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

CACHE_SCHEMA = "task2q-cache-v1"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def content_key(kind: str, sources: dict[str, str], config: dict[str, Any]) -> str:
    """Source keys are portable logical names, values are file content hashes."""
    doc = {"schema": CACHE_SCHEMA, "kind": kind,
           "sources": sources, "config": config}
    return hashlib.sha256(json.dumps(doc, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
            json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_arrays(directory: str | Path, key: str, arrays: dict[str, np.ndarray],
                 metadata: dict[str, Any]) -> dict[str, Any]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = directory / f"{key}.npz"
    fd, temporary = tempfile.mkstemp(prefix=".tmp-", suffix=".npz", dir=directory)
    try:
        with os.fdopen(fd, "wb") as out:
            np.savez_compressed(out, **arrays)
            out.flush()
            os.fsync(out.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, payload)
        record = {"schema": CACHE_SCHEMA, "key": key, "sha256": digest,
                  "file": payload.name, "metadata": metadata}
        atomic_json(directory / f"{key}.json", record)
        return record
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_arrays(directory: str | Path, key: str) -> tuple[dict[str, np.ndarray], dict] | None:
    directory = Path(directory)
    payload = directory / f"{key}.npz"
    try:
        record = json.loads((directory / f"{key}.json").read_text(encoding="utf-8"))
        if record["schema"] != CACHE_SCHEMA or record["key"] != key:
            return None
        if sha256_file(payload) != record["sha256"]:
            return None
        with np.load(payload, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        return arrays, record
    except (OSError, ValueError, KeyError, EOFError, json.JSONDecodeError):
        return None
