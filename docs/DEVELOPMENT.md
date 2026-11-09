# From a reference case to your own model

Start in the unpacked archive and install `pip install -r requirements/core.txt`. This installs an editable package and the tested CPU versions. `requirements/reproduction.txt` adds simulation. `pip install -e .[reproduction,test,dev]` is the shorter editable installation when exact transitive versions are unnecessary. Build distributions with `python -m build --no-isolation`; the source distribution retains compact plotting data, media and configurations, while the wheel supplies the numerical API and experiment modules. Run figure/table commands from the extracted source distribution or ZIP, where `data/manifest.json` exists. Full `verify` additionally requires the separate audit dataset; see `zenodo/README.md`.

## Run a supplied case

The five fully populated TOML files in `configs/` retain article defaults. They are parsed with Python 3.11 `tomllib` into dataclasses. No third-party configuration framework is used.

| Config | Study and output |
|---|---|
| `cooperative.toml` | GNE and vGNE, all three agents moving, nominal or PyBullet flight |
| `pursuit-evasion.toml` | Blocked, damped best responses; three pursuers, two evaders |
| `controlled.toml` | Three-agent static/closed-loop comparison, all common starts, unilateral searches |
| `scalability.toml` | 42 matched cases, full/blocked controls, discarded warmup and five timed calls |
| `laboratory.toml` | Offline analysis of the retained log plus a configurable linear-controller demonstration |

`python -m game_theory_mpc run --config configs/cooperative.toml --output outputs/new-flight` executes the full supplied case. These can be long computations. For exploration reduce `execution.updates`, the controlled starts/cases, or the benchmark cases/repeats.

The same code is available from Python:

```python
from dataclasses import replace
from game_theory_mpc import load_config, run_experiment
config = load_config('configs/cooperative.toml')
config = replace(config, execution=replace(config.execution, plant='nominal', updates=2))
results = run_experiment(config, 'outputs/two-steps')
```

## Change parameters

Copy a TOML file before editing. `model.states` and `model.targets` set a flight's initial condition and goals. State rows list pursuers first in an adversarial game; only evaders have targets. Controlled and scaling studies store geometry in `[[cases]]`. Case IDs become directory names. Benchmark block schedules belong in each case rather than the common model. An adversarial benchmark case explicitly overrides the common velocity coefficient to 5. Structural case fields are validated for each study. The controlled game remains a three-agent study. Benchmark cases explicitly contain each geometry and block schedule; change those rows to extend the benchmark.

`model.blocks` defines the horizon by its sum. `model.dt` is the planning period and is used by prediction, DARE terminal weights, jerk limits, first-interval checks, plant integration and time-based evaluation. PyBullet additionally requires `physics_hz / pid_hz` and `dt * pid_hz` to be integers. Physics and PID settings affect only `plant = "pybullet"`; nominal dynamics are exact double-integrator steps. The physical envelope must contain planned limits.

Position/velocity/repulsion/effort coefficients affect cooperative costs. The six named `adversarial_weights` coefficients replace position/repulsion/effort in pursuit–evasion; velocity and terminal weights remain active. `max_sweeps` selects the cooperative response budget, `response_sweeps` the adversarial response budget, and `max_iterations` the local NLP budget. The default initialization, solver tolerances, acceptance and retry policies match the retained planners. Controlled-study tolerances and budgets are explicitly in `[controlled]` and shared by its numerical and independent models.

Laboratory CSV paths are relative to the working directory. This release analyzes an existing flight and exposes the linear planner; it does not issue hardware commands. Linear examples use unconstrained tracking with an input-increment penalty, so bounds and obstacle avoidance require Python model work.

## Change a cost or constraint

`src/game_theory_mpc/experiment_model.py` builds planners. `control_step.py` separates candidate admission from solving; `experiments.py` handles workflows; `experiment_io.py` records provenance; `plotting.py` renders new trajectories. The PyBullet loop is in `experiments/experiment_blocked_terminal_mpc.py`, where each planner update spans the configured number of PID ticks.

For blocked games, change both the JAX model (`blocked_game_mpc.py`, `_terms`, `_constraints`) and the independent NumPy evaluation (`evaluate_numpy`). Cooperative optimization uses analytic NumPy derivatives in `cooperative_equilibrium_mpc.py`; update its gradient, constraint Jacobian and Hessians as well. Add a check at nonzero controls with nondefault parameters comparing values and derivatives against JAX AD. Preserve the meaning and ordering of inequality rows and multiplier columns.

For the controlled game, update `experiments/equilibrium_selection/model.py` and `NashOptController` together. `Solvers.derivative_check` compares costs, potential, own-player derivatives and constraint Jacobians. A small residual alone is not a global equilibrium certificate; retain feasible unilateral-improvement searches and their rejected/accepted diagnostics.

A controller solve does not execute a blocked input. Check its admission policy, apply the first acceleration and call `commit_controls` with the actual applied value so the next jerk constraint has the correct history. Preserve rejected attempts. Extending the state dimension, plant, player topology of the controlled study or acceptance policy requires Python changes and renewed independent checks.

For the pursuit–evasion damping ablation, `model.response_fractions` lists the update fractions in trial order. Its default is `[0.7, 0.35, 1.0]`; use `[1.0]` for full responses. See the [matched flight protocol](RESPONSE_RELAXATION.md).
