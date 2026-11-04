# Data, diagnostics and provenance

The default software distribution contains `data/reference.zip`, `data/hardware/run.csv` and a checksum/availability manifest. The compact ZIP stores unrounded trajectory samples needed for plotting, controlled-conflict paths, twelve weight-transfer pairs, scaling rows, and archived table/metric summaries. It contains no full per-attempt solver archive. Figures are rendered from these numerical inputs rather than copied reference PDFs.

Complete evidence is in the separate `release-game-theory-mpc-article-data-1.0.0.tar.xz` dataset, identified by DOI [10.5281/zenodo.23947592](https://doi.org/10.5281/zenodo.23947592). Its internal manifest checks every retained record. The software pins the archive and manifest SHA-256 values. The DOI and archive identity are recorded in `zenodo/`. Files become available after the dataset record is published.

| Path inside the optional dataset | Evidence |
|---|---|
| `data/equilibrium_selection` | 270 static calls, 90 episodes, 7,399 raw KKT/candidate/deviation records including failures and duplicate-profile associations |
| `data/weights/calibration` | Every coordinate-search candidate, frozen trace, all twelve manual/automatic test pairs and two physics transfer cases |
| `data/scalability` | 42 cases, preparation, discarded warmups, 210 timed calls and rejected candidates |
| `data/cooperative` | GNE/vGNE states, references, motor/PID telemetry and all attempted/applied updates |
| `data/adversarial` | Pursuit–evasion flight and four nominal weight profiles |
| `data/hardware` | Accepted five-drone CSV with commands, targets, measurements and timings |

Use `python -m game_theory_mpc verify --audit-data DATASET.tar.xz` for the complete independent audit. The archive is extracted temporarily and reference records are never modified. `figures` and `tables` work without it and explicitly record their more limited scope. Custom experiments require neither the audit archive nor its checksums.

Trajectories use `states (T,N,6)`, `targets (N-P,3)` and `times (T,)`. Nominal controls have shape `(T-1,N,3)`. PyBullet trajectories sample at PID frequency; the audit dataset additionally retains references, quaternions, RPMs, forces and accelerations. Planner predictions are agent-major `(N,H,6)`; see [API.md](API.md). All quantities use SI units.

Attempt records store states, targets, actual previous inputs, full/blocked controls, predictions, costs, timings, constraints, multipliers and acceptance checks. `feasible`, `validation.accepted` and `applied` are distinct. Feasible approximations remain counted separately from accepted equilibria. Failed calls and repair candidates remain evidence in the dataset.

New runs retain `effective_config.json`, `provenance.json`, `status.json`, metrics, attempts and trajectories. Provenance includes the seed, interpreter/platform, dependency versions and code hashes. Failed runs retain their error and available diagnostics. Existing directories are rejected. No new-run output is sent to Zenodo automatically.
