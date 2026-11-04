# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Extract local data archives without trusting archive member paths or links."""
from pathlib import Path
import shutil
import stat
import tarfile
import zipfile


def extract(archive, destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    seen = set()

    def target(name):
        path = (root / name).resolve()
        if (
            Path(name).is_absolute()
            or not path.is_relative_to(root)
            or path == root
            or name in seen
        ):
            raise ValueError("Unsafe or duplicate archive member: " + name)
        seen.add(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                if entry.is_dir():
                    continue
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Archive links are unsupported")
                path = target(entry.filename)
                with source.open(entry) as stream, path.open("xb") as output:
                    shutil.copyfileobj(stream, output)
    else:
        with tarfile.open(archive, "r:*") as source:
            for entry in source:
                if entry.isdir():
                    continue
                if not entry.isfile():
                    raise ValueError("Archive links/special files are unsupported")
                path = target(entry.name)
                with source.extractfile(entry) as stream, path.open("xb") as output:
                    shutil.copyfileobj(stream, output)
    return root
