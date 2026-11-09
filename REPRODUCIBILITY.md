# Reproduction and independent verification

Use Python 3.11 and the CPU dependencies pinned in `requirements/core.txt`. Add `requirements/reproduction.txt` for PyBullet.

Weight-comparison and automatic weight-calibration studies are excluded from both deliverables. Fixed objective weights remain in the retained experiment configurations. The pursuit–evasion flight and matched damping comparison remain included.

The small software package supports figure/table regeneration and custom experiments on its own:

```bash
python -m pytest -q
python -m game_theory_mpc figures
python -m game_theory_mpc tables
python -m game_theory_mpc quick
python -m build --no-isolation
python -m twine check dist/*
```

`figures` creates four composed main PDFs and two supporting laboratory PDFs from compact numerical inputs. `tables` creates the controlled-study table as CSV and LaTeX using archived aggregate statistics. These commands check their input hashes and expected published values, but **do not perform the full independent solver audit**. The generated `scope.json` records this distinction. New experiments always plot their own trajectories.

The complete retained evidence, including failed calls, approximate inputs and unilateral-response searches, is distributed separately as `release-game-theory-mpc-article-data-1.0.0.tar.xz`. Keep it compressed:

```bash
python -m game_theory_mpc verify --audit-data /path/to/release-game-theory-mpc-article-data-1.0.0.tar.xz
```

`verify` first checks the archive's SHA-256 and version, extracts it into temporary storage, checks every record against its manifest and independently reconstructs the numerical results. It separates accepted candidates from applied inputs and verifies the controlled-game candidates and unilateral-response searches, warmup/timed scaling calls and flight/laboratory evidence. Without the dataset, `verify` explains what is missing and exits; a summary-only check is never presented as a full audit. An extracted dataset directory is also supported, with its manifest checked against the expected manifest hash. The archive is automatically found when placed beside the extracted software folder.

The software's `data/manifest.json` pins the dataset filename, byte count, SHA-256 and version. [Zenodo metadata](zenodo/README.md) are prepared for a separate Dataset record under CC BY 4.0. They are not a published deposit. Once a DOI is assigned, use the **version DOI** for exact reproduction and link the dataset and software through their related identifiers. No download is performed automatically.

All commands create fresh output directories; `--output` selects one explicitly. `--root` selects another extracted software package. Custom runs do not inspect reference checksums or compare against article numbers. `quick --skip-pybullet` uses nominal dynamics. Preparation and warmup are excluded from benchmark timing; new wall-clock measurements depend on hardware.

Rebuild the preview without FFmpeg: `python -m tools.animate --source data/reference.zip --output outputs/cooperative.gif`. It uses 80 frames, 10 fps, width 960 pixels, one 128-color palette and infinite looping. The view bundle preserves the original trajectory samples; its provenance identifies the complete source trajectories.

Maintainers build both deliverables in the research repository:

```bash
python scripts/build_release.py --profile article --archive
python scripts/build_release.py --profile article --check
```

The builder pins every raw input and refuses stale compact views. To update those views, run `release_support/build_article_views.py` against a fully verified dataset and its metrics. The full release profile remains independent. Neither extracted deliverable requires the research repository or Git.

The [response-relaxation comparison](docs/RESPONSE_RELAXATION.md) is included in the pursuit–evasion audit: both flight arms, their matched settings, selected fractions and the rejected full-update altitude prediction are recomputed from raw records.
