# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
import json
from pathlib import Path
import re
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_reference_manifest_covers_the_selected_archive_and_media():
    from tools.reproduce import verified_manifest

    manifest = verified_manifest(ROOT)
    assert manifest["profile"] == "article"
    assert manifest["audit_dataset"]["doi"] is None
    assert len(manifest["audit_dataset"]["manifest_sha256"]) == 64
    for folder in ("backend_choice", "weights"):
        assert not (ROOT / "data" / folder).exists()
    assert not any(check["key"].startswith("weights.") for check in manifest["checks"])
    assert not (ROOT / "configs/weights.toml").exists()
    assert not (ROOT / "src/game_theory_mpc/adversarial_controller.py").exists()
    media = json.loads((ROOT / "media/cooperative.json").read_text())
    import zipfile

    with zipfile.ZipFile(ROOT / "data/reference.zip") as bundle:
        provenance = json.loads(bundle.read("provenance.json"))
        assert "tuning.json" not in bundle.namelist()
        assert "weights" not in json.loads(bundle.read("metrics.json"))
        assert "nominal_weights" not in json.loads(bundle.read("metrics.json"))["adversarial"]
        assert not any("/weights/" in name for name in bundle.namelist())
    assert manifest["audit_dataset"]["records"] == provenance["source_files"]
    assert provenance["original_cooperative_sha256"] == media["source_sha256"]


def test_preview_dimensions_frames_palette_and_timing():
    from PIL import Image

    with Image.open(ROOT / "media/cooperative.gif") as image:
        assert image.width == 960 and image.n_frames == 80 and image.info["loop"] == 0
        palette = image.getpalette()
        assert (
            len(set(tuple(palette[i : i + 3]) for i in range(0, len(palette), 3)))
            <= 128
        )
        durations = []
        for k in range(image.n_frames):
            image.seek(k)
            durations.append(image.info["duration"])
        assert durations == [100] * 80


def test_relative_documentation_links():
    failures = []
    for path in [
        ROOT / "README.md",
        ROOT / "REPRODUCIBILITY.md",
        *(ROOT / "docs").glob("*.md"),
    ]:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if "://" in link or link.startswith("#"):
                continue
            target = link.split("#")[0]
            if not (path.parent / target).exists():
                failures.append(f"{path.name}: {link}")
    assert not failures, failures


def test_verify_requires_full_dataset_and_never_substitutes_summaries(tmp_path):
    from tools.reproduce import reference

    # A private root with the same software assets but no sibling dataset.
    import shutil

    root = tmp_path / "software"
    shutil.copytree(ROOT / "data", root / "data")
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / row["path"], path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="separate audit dataset"):
        reference("verify", output, root)
    bad = tmp_path / "bad.tar.xz"
    bad.write_bytes(b"wrong archive")
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        reference("verify", output, root, bad)


def test_archive_rejects_path_traversal_and_links(tmp_path):
    import zipfile, tarfile
    from tools.data_archives import extract

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract(archive, tmp_path / "out")
    tar = tmp_path / "bad.tar"
    with tarfile.open(tar, "w") as output:
        entry = tarfile.TarInfo("link")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "/tmp"
        output.addfile(entry)
    with pytest.raises(ValueError, match="links"):
        extract(tar, tmp_path / "tar-out")
    assert not (tmp_path / "escape").exists()


def test_tables_use_compact_inputs_without_running_the_solver_audit(
    tmp_path, monkeypatch
):
    from tools import results
    from tools.reproduce import reference

    def unexpected_audit(*args, **kwargs):
        raise AssertionError("Rendering must not run the complete solver audit")

    monkeypatch.setattr(results, "analyze", unexpected_audit)
    output = tmp_path / "tables"
    output.mkdir()
    reference("tables", output, ROOT)
    assert (output / "tables/equilibrium_selection.tex").is_file()
    scope = json.loads((output / "scope.json").read_text())
    assert scope["independent_solver_audit"] is False
