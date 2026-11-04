# Controller interfaces and data conventions

Start with [DEVELOPMENT.md](DEVELOPMENT.md) to select the appropriate model,
then [CODE_MAP.md](CODE_MAP.md) to find its implementation. All paths and commands
below assume the extracted companion package is the current directory.

## Arrays and units

`N` is the total number of agents, `H` the prediction horizon and `B` the number
of control blocks. State rows are `[px, py, pz, vx, vy, vz]`, with positions in
metres and velocities in metres per second. Controls are desired translational
accelerations in metres per second squared; they are not motor commands.

| Quantity | Shape | Meaning |
|---|---|---|
| Measured states | `(N, 6)` | Current position and velocity of every agent |
| Cooperative targets | `(N, 3)` | One fixed position target per agent |
| Adversarial targets | `(num_evaders, 3)` | Evader targets only; state rows list pursuers first |
| Expanded controls | `(N, H, 3)` | Acceleration at every prediction interval |
| Blocked controls | `(N, B, 3)` | Constant acceleration per block; `sum(block_lengths) == H` |
| Full predictions | `(N, H, 6)` | States after each interval; index zero is the next state |

The older nonlinear and linear interfaces also accept six-column target rows,
using their first three columns. The blocked planners require three columns.
Use finite floating-point arrays, CPU JAX and double precision for the reported
numerical tolerances. Enable it before constructing a planner:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Return values differ between controller families

| Controller | `solve(...)` returns | Prediction layout |
|---|---|---|
| `CooperativeEquilibriumMPC`, `BlockedGameMPC` | `(controls, predictions, info)` | Controls `(N,H,3)`; predictions array `(N,H,6)` |
| `NashOptController` | `(controls, predictions, info)` | Controls `(N,H,3)`; list of `N` prediction arrays `(H,6)` |
| `LinearNashMPCController` | `(controls, predictions, info)` | First input only, `(N,1,3)`; list of `N` predictions `(1,6)` |

Use `np.asarray(predictions)` to inspect a list as a joint array. Take
`controls[:, 0]` for the first acceleration. The linear planner optimizes over
`H` intervals but returns only the first one. 

## One blocked cooperative update

Run `python examples/cooperative.py` for an executable example.

1. Construct `CooperativeEquilibriumMPC` with a fixed block schedule, bounds and
   weights. `method="gne"` uses sequential player minimizations; `"vgne"` uses
   the exact potential. See the protocol for polishing and repair checks.
2. Call `prepare(states, targets)` to prepare derivative graphs before timing
   solves. Its return value is elapsed preparation time in seconds.
3. Call `solve(states, targets)`. It computes and shifts a warm start, even if
   the returned candidate is rejected. This call does not advance a plant.
4. Inspect the plan and apply an execution policy. The article's flight driver
   also checks the first interval, recovery bounds and physical telemetry.
5. After an acceleration is admitted and applied, call
   `commit_controls(applied_accelerations)`, with shape `(N,3)`. The next jerk
   bound must use the actual applied input, including any external modification.

The library does not automatically undo a warm-start shift if the caller
rejects execution. The linear and older nonlinear controllers manage their
input histories inside `solve`; they do not use this commit interface.

## Reading diagnostics

For the cooperative blocked solvers:

- `info['feasible']` concerns the returned nominal plan, including input bounds.
- `info['validation']['accepted']` additionally requires the configured KKT,
  multiplier-sign and tested unilateral-response checks. It may be false for a
  feasible approximation. It does not prove global equilibrium.
- `info['multipliers']` has one row per player. Columns are shared model
  inequalities, all lower input bounds, then all upper input bounds. vGNE uses
  common multipliers on the shared rows.
- `info['repair_history']`, when present, retains rejected points and improving
  unilateral witnesses used for restarts; `info['deviations']` contains the final
  candidate's response checks.
- `compute_time`, `projection_seconds`, `solver_seconds` and `checks_seconds`
  are seconds. Preparation is separate from the complete solve time.

`BlockedGameMPC` reports feasibility and response attempts without the
cooperative `validation` dictionary. Its adversarial route performs finite
damped player responses. The older NashOpt interfaces instead expose
`kkt_residual`; `return_diagnostics=True` adds solver status/evaluation counts
and shared inequalities/multipliers when present. A small residual alone does
not establish conflict resolution or absence of improving responses.

The linear and reactive interfaces report computation time without a coupled
equilibrium check. The linear model has no internal input, separation or
obstacle constraints. Its `reset_input_history()` clears only the previous
input, for example after a synthetic warmup.

## Navigating saved data and code

[data/README.md](../data/README.md) identifies datasets, record families and
weight-profile aliases. Stored numerical records use experiment-specific
schemas; a saved record's `validation` fields need not equal a public
controller's entire `info` dictionary. [DEVELOPMENT.md](DEVELOPMENT.md)
defines thresholds and reporting denominators.

To change a scenario, start with an example and explicit constructor settings.
To change a model, follow `_prediction`, `_terms` and `_constraints` in
`blocked_game_mpc.py`. Cooperative analytic derivatives are implemented
separately in `cooperative_equilibrium_mpc.py`; a model change must update both
routes and their independent checks. `reusable_gne.py` contains the joint KKT
residual wrapper. Drivers own execution and logging; `tools/results.py` owns
reference metrics. Keep reference data intact and write new runs below
`outputs/`.
