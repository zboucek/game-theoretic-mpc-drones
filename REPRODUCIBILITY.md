# Reproduction and independent verification

Use Python 3.11 and the CPU dependencies pinned in `requirements/core.txt`. Add `requirements/reproduction.txt` for PyBullet.

The small software package supports figure/table regeneration and custom experiments on its own:

```bash
python -m pytest -q
python -m game_theory_mpc figures
python -m game_theory_mpc tables
python -m game_theory_mpc quick
python -m build --no-isolation
python -m twine check dist/*
```

`figures` creates four composed main PDFs and three supporting PDFs from compact numerical inputs. `tables` creates the controlled-study table as CSV and LaTeX using archived aggregate statistics. These commands check their input hashes and expected published values, but **do not perform the full independent solver audit**. The generated `scope.json` records this distinction. New experiments always plot their own trajectories.

The complete retained evidence, including failed calls, approximate inputs and unilateral-response searches, is distributed separately as `release-game-theory-mpc-article-data-1.0.0.tar.xz`. Keep it compressed:

```bash
python -m game_theory_mpc verify --audit-data /path/to/release-game-theory-mpc-article-data-1.0.0.tar.xz
```

`verify` first checks the archive's SHA-256 and version, extracts it into temporary storage, checks every record against its manifest and independently reconstructs the numerical results. It separates accepted candidates from applied inputs, verifies all twelve manual/automatic weight pairs, every design candidate, warmup/timed scaling call and the flight/laboratory evidence. Without the dataset, `verify` explains what is missing and exits; a summary-only check is never presented as a full audit. An extracted dataset directory is also supported, with its manifest checked against the expected manifest hash. The archive is automatically found when placed beside the extracted software folder.

The software's `data/manifest.json` pins the dataset filename, byte count, SHA-256 and version. The Dataset record is identified by DOI [10.5281/zenodo.23947592](https://doi.org/10.5281/zenodo.23947592), under CC BY 4.0; see [dataset details](zenodo/README.md). Files are available after the Zenodo record is published. Use the DOI identifying the matching dataset version for exact reproduction. No download is performed automatically.

All commands create fresh output directories; `--output` selects one explicitly. `--root` selects another extracted software package. Custom runs do not inspect reference checksums or compare against article numbers. `quick --skip-pybullet` uses nominal dynamics. Preparation and warmup are excluded from benchmark timing; new wall-clock measurements depend on hardware.

Rebuild the preview without FFmpeg: `python -m tools.animate --source data/reference.zip --output outputs/cooperative.gif`. It uses 80 frames, 10 fps, width 960 pixels, one 128-color palette and infinite looping. The view bundle preserves the original trajectory samples; its provenance identifies the complete source trajectories.

The checked compact plotting inputs and complete audit dataset are versioned
release artifacts. To change the model or perform new studies, follow
[DEVELOPMENT.md](docs/DEVELOPMENT.md). Keep the archived reference inputs intact
and write new results to a fresh output directory.
