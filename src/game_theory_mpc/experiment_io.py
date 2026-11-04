# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Result persistence and provenance; no dependence on the reference archive."""
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def write_json(path, value):
    from experiments.equilibrium_selection.archive import serializable

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(value), indent=2, allow_nan=False) + "\n")


def provenance(config):
    import experiments, tools

    hashes = {}
    for package, name in (
        (Path(__file__).parent, "game_theory_mpc"),
        (Path(experiments.__file__).parent, "experiments"),
        (Path(tools.__file__).parent, "tools"),
    ):
        for file in sorted(package.rglob("*.py")):
            hashes[name + "/" + file.relative_to(package).as_posix()] = sha256(
                file.read_bytes()
            ).hexdigest()
    versions = {}
    for name in (
        "numpy",
        "scipy",
        "jax",
        "jaxlib",
        "nashopt",
        "highspy",
        "matplotlib",
        "pillow",
        "pybullet",
        "gym-pybullet-drones",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(
        config=asdict(config),
        seed=config.seed,
        python=sys.version,
        platform=platform.platform(),
        dependencies=versions,
        code_sha256=hashes,
    )
