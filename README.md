# Game-theoretic MPC — article edition 1.0.0

![Three moving drones, GNE and vGNE](media/cooperative.gif)

Animation of the article's cooperative **PyBullet trajectories**: the first eight seconds of **simulated time**, at 10 frames/s. [Full MP4](media/cooperative.mp4). Physics pauses while a plan is solved.

The main package contains the planners, five editable study configurations, examples, and compact numerical inputs for the article figures and table. Complete solver attempts and audit evidence for these five studies are a **separate dataset**, prepared for Zenodo. Code is MIT; data and visualizations are CC BY 4.0.

Use Python 3.11 on CPU, from this extracted directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/core.txt
python -m game_theory_mpc figures
python -m game_theory_mpc tables
```

Run your own experiment:

```bash
cp configs/cooperative.toml my-flight.toml
# Edit targets, weights, bounds, blocks or execution.updates.
python -m game_theory_mpc run --config my-flight.toml --output outputs/my-flight
```

For PyBullet flights, install `pip install -r requirements/reproduction.txt`; the pinned simulator wheel is included. Set `execution.plant = "nominal"` to run without a simulator. `python -m game_theory_mpc quick` runs short smoke checks. Installation is editable, so Python changes take effect on the next run. Existing output directories are rejected.

Full independent numerical verification uses the optional dataset **without unpacking it yourself**:

```bash
python -m game_theory_mpc verify --audit-data ../release-game-theory-mpc-article-data-1.0.0.tar.xz
```

See [dataset availability and proposed Zenodo metadata](zenodo/README.md). The dataset has been prepared locally; a Zenodo DOI has not yet been assigned. Figures/tables use compact numeric inputs and archived summaries; they do not claim to repeat the complete solver audit. Custom experiments do not require the reference dataset.

Continue with [development and configuration](docs/DEVELOPMENT.md), [article-to-code map](docs/CODE_MAP.md), [controller APIs](docs/API.md), [data formats](docs/DATA.md), and [verification](REPRODUCIBILITY.md). Four short [examples](examples/) use the article methods. Regenerate the GIF without FFmpeg or another solve: `python -m tools.animate --source data/reference.zip --output outputs/cooperative.gif`.
