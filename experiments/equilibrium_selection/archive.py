# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Raw attempt I/O, source recording and resumable checkpoints for article V-A.

Archive writes new experiment outputs. Bundled reference records are read
through tools/results.py without resuming or modifying those inputs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

import numpy as np

from .protocol import PROTOCOL

ROOT = Path(__file__).resolve().parents[2]




def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(serializable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_paths():
    return sorted([*(ROOT / 'src/game_theory_mpc').glob('*.py'),
                   *(ROOT / 'experiments').rglob('*.py')])




@contextmanager
def archive_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".study.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using this study archive") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Archive:
    def __init__(self, directory, config=PROTOCOL, verify_sources=True, verify_environment=False):
        self.directory = Path(directory)
        self.config = config
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / "manifest.json"
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths()}
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            if self.manifest["protocol"] != config.as_dict():
                raise ValueError("Protocol changed; use a new output archive")
            if verify_sources and self.manifest["source_sha256"] != hashes:
                raise ValueError("Sources changed; preserve this archive and choose a new --output")
            if verify_environment:
                current = {name: importlib.metadata.version(name) for name in
                           ("numpy", "scipy", "jax", "jaxlib", "nashopt")}
                if any(self.manifest["library_versions"].get(name) != version for name, version in current.items()):
                    raise ValueError("Numerical library versions changed; use a new output archive")
                if self.manifest["python"] != sys.version:
                    raise ValueError("Python version changed; use a new output archive")
            for name, sha in self.manifest["source_sha256"].items():
                if digest(self.directory / "source_snapshots" / name) != sha:
                    raise ValueError(f"Corrupted source snapshot: {name}")
        else:
            versions = {dist.metadata["Name"].lower(): dist.version for dist in importlib.metadata.distributions()}
            cpu_info = Path("/proc/cpuinfo")
            cpu = next((line.partition(":")[2].strip() for line in cpu_info.read_text().splitlines()
                        if line.startswith("model name")), platform.processor()) if cpu_info.exists() else platform.processor()
            self.manifest = {"schema": 1, "created_utc": utc_now(), "protocol": config.as_dict(),
                "source_commit": None,
                "source_sha256": hashes, "python": sys.version, "executable": sys.executable,
                "platform": platform.platform(), "processor": cpu,
                "library_versions": versions, "thread_environment": {
                    name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "XLA_FLAGS", "JAX_ENABLE_X64", "JAX_PLATFORM_NAME")},
                "timing": "Solve only; construction, compilation, independent validation and deviations excluded",
                "nonfinite_encoding": "Nonfinite floats become JSON null; such candidates are rejected"}
            for source in source_paths():
                destination = self.directory / "source_snapshots" / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
            write_json(manifest_path, self.manifest)
        checkpoint_path = self.directory / "checkpoint.json"
        self.checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {"records": {}}
        # A raw attempt may have been committed just before a process stopped.
        for path in sorted((self.directory / "raw").glob("*.json")):
            key = path.stem
            sha = digest(path)
            if key in self.checkpoint["records"] and self.checkpoint["records"][key] != sha:
                raise ValueError(f"Raw attempt changed: {key}")
            self.checkpoint["records"][key] = sha
        for key in self.checkpoint["records"]:
            if not (self.directory / "raw" / f"{key}.json").exists():
                raise ValueError(f"Missing checkpointed attempt: {key}")

    def get(self, key):
        if key not in self.checkpoint["records"]:
            return None
        path = self.directory / "raw" / f"{key}.json"
        if digest(path) != self.checkpoint["records"][key]:
            raise ValueError(f"Raw attempt changed: {key}")
        return json.loads(path.read_text())

    def save(self, key, record):
        if self.get(key) is not None:
            raise ValueError(f"Refusing to overwrite an existing raw attempt: {key}")
        record = dict(record, record_id=key, recorded_utc=utc_now())
        path = self.directory / "raw" / f"{key}.json"
        write_json(path, record)
        self.checkpoint["records"][key] = digest(path)
        write_json(self.directory / "checkpoint.json", self.checkpoint)
        return self.get(key)

    def event(self, kind, **details):
        key = str(uuid.uuid4())
        write_json(self.directory / "events" / f"{key}.json",
                   dict(event=kind, utc=utc_now(), **details))
        return key

    def records(self, kind=None):
        for key in sorted(self.checkpoint["records"]):
            record = self.get(key)
            if kind is None or record.get("kind") == kind:
                yield record
