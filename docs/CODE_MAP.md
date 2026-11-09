# Article to code

| Study | Planner / workflow | Independent evidence |
|---|---|---|
| Controlled GNE/vGNE/potential | `experiments/equilibrium_selection/{protocol,solvers,runner}.py`, `NashOptController`, `reusable_gne.py` | `model.py`, `reporting.py`, raw attempts and unilateral deviations |
| Computational scaling | `benchmark_current_scalability.py`; configurable cases in `game_theory_mpc/experiments.py` | `audit_current_scalability.py`, full/blocked matched raw samples |
| Cooperative flight | `CooperativeEquilibriumMPC`, `experiment_blocked_terminal_mpc.py` | `audit_cooperative_equilibria.py`, `audit_blocked_terminal_mpc.py`, PID trajectories |
| Pursuit–evasion | `BlockedGameMPC`, finite damped player responses | `audit_blocked_terminal_mpc.py`, full flight and matched damping ablation (`tools/response_relaxation.py`) |
| Laboratory tracking | `LinearNashMPCController` | `tools/hardware.py`, recorded five-drone CSV |

The four composed main figures are cooperative flight, pursuit–evasion, computational scaling and numerical conflict. The controlled-method table is regenerated as CSV and LaTeX. Supporting figures show the laboratory XY paths/timing. The reference renderers regenerate these from numeric inputs; new-run plots come only from that run's saved trajectories.

`load_config` and `run_experiment` are the public workflow API. Controllers remain importable directly. The CLI in `game_theory_mpc/__main__.py` is a thin dispatcher. `tools/reproduce.py` exclusively handles reference verification/figures/tables and smoke checks. It is independent of the custom-run dispatcher.

The raw evidence paths above are inside the optional audit dataset, prepared as a separate Zenodo deposit. The default software ZIP carries compact plotting inputs. `tools/reference_views.py` renders them; `tools/data_archives.py` safely opens local archives. Full independent verification requires `verify --audit-data DATASET.tar.xz`.

The [response-relaxation protocol](RESPONSE_RELAXATION.md) gives the matched flight settings and reproduction commands.
