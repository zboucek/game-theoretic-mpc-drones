# Pursuit–evasion response relaxation

The article compares the existing update fractions `(0.7, 0.35, 1.0)` with
full updates `(1.0,)` in the same PyBullet CF2X scenario. Both arms request
300 updates (30 simulated seconds), with five response sweeps, horizon 20,
blocks `(1,1,2,4,6,6)`, goal/evasion weights 100/10, and the same PID,
initial states, constraints, warm starts and altitude reserve retry.
Physics/PID/planning rates are 240/120/10 Hz. Physics pauses during optimization.

The damped run reaches simultaneous arrival at 16.35 s and runs for 30 s.
The full-update run stops before applying a rejected plan at 2.5 s.
Pursuer 2 then has height 0.301842 m and vertical velocity −0.178365 m/s.
Even maximum upward model acceleration of 2 m/s² gives a next-node height
of 0.294006 m, below the 0.30 m altitude limit. Neither run crosses the
0.30 m capture threshold before its endpoint. The 35% option is never
selected in the damped run, including altitude retries.

This comparison supports retaining damping in this finite-sweep planner.
It does not establish convergence or optimal update fractions. Whole-run
effort and tracking errors must not rank trajectories of unequal duration.

The article edition’s optional audit dataset (and the full edition’s bundled data) retains both flights under
`data/adversarial/relaxation/{damped,full}/`: configurations, full trajectories,
summaries and every attempted plan, including the final rejection and retries.
`verify` runs the independent dynamics/cost/constraint/telemetry auditor on
both arms, checks their matched configurations and selected fractions, and
recomputes arrival, stop time and the first-node altitude witness with
`tools/response_relaxation.py`.

In the article edition, reproduce this pair using two copies of
`configs/pursuit-evasion.toml`. Set `execution.updates = 300` in both;
retain `model.response_fractions = [0.7, 0.35, 1.0]` in the first and set
`model.response_fractions = [1.0]` in the second. Keep all other fields equal:

```bash
python -m game_theory_mpc run --config damped.toml --output outputs/damped
python -m game_theory_mpc run --config full.toml --output outputs/full
```

In the full edition, the corresponding original flight driver is
`python -m experiments.experiment_blocked_terminal_mpc`; use the declared
pursuit–evasion settings in `EXPERIMENT_PROTOCOLS.md`, `--updates 300` and
`--response-fractions .7 .35 1` or `--response-fractions 1` respectively.
