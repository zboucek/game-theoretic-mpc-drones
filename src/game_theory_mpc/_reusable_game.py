# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček

"""Connect UAV controller costs to reusable_gne (article III-B/D and IV-A).

The parameter vector holds states, targets and fixed-capacity obstacles.
This mixin builds the shared residual graph and shifts MPC warm starts."""

from time import perf_counter

import jax.numpy as jnp
import numpy as np

from .reusable_gne import ReusableGNEP


class ReusableGameMixin:
    """Bind controller-specific costs to one fixed residual graph.

    Runtime parameters pack states, position targets and padded obstacles.
    The mixin owns the shifted warm start used by the older nonlinear APIs."""
    def _initialize_game(self, target_count):
        self._target_count = target_count
        self._state_parameter_count = self.num_agents*self.nx
        self._target_parameter_count = target_count*3
        npar = self._state_parameter_count + self._target_parameter_count + self.max_obstacles*3
        self._hard_obstacle_count = (self.num_agents*self.max_obstacles*self.horizon
                                     if self.use_hard_constraints else 0)
        self._hard_pair_count = (self.num_agents*(self.num_agents-1)//2*self.horizon
                                 if getattr(self, "use_hard_collision", False) else 0)
        ng = self._hard_obstacle_count + self._hard_pair_count

        def cost(controls, parameters, player):
            states, targets, obstacles = self._unpack_parameters(parameters)
            return self._jit_cost(controls, states, targets, obstacles)[player]

        def inequalities(controls, parameters):
            states, _, obstacles = self._unpack_parameters(parameters)
            return self._shared_inequalities(controls, states, obstacles)

        self._game = ReusableGNEP(
            [self.horizon*self.nu]*self.num_agents,
            [lambda u, p, i=i: cost(u, p, i) for i in range(self.num_agents)],
            npar=npar, g=inequalities if ng else None, ng=ng,
            lb=np.full(self.num_agents*self.horizon*self.nu, -getattr(self, "input_bound", 5.)),
            ub=np.full(self.num_agents*self.horizon*self.nu, getattr(self, "input_bound", 5.)),
            variational=self.variational,
        )

    def _unpack_parameters(self, parameters):
        a = self._state_parameter_count
        b = a + self._target_parameter_count
        return (parameters[:a].reshape(self.num_agents, self.nx),
                parameters[a:b].reshape(self._target_count, 3),
                parameters[b:].reshape(self.max_obstacles, 3))

    def _shared_inequalities(self, controls, states, obstacles):
        inputs = controls.reshape(self.num_agents, self.horizon*self.nu)
        predicted = (states @ self.Sx.T + inputs @ self.Su.T).reshape(
            self.num_agents, self.horizon, self.nx)
        parts = []
        if self._hard_obstacle_count:
            # Reserve every slot even when no obstacle is currently present.
            # Inactive rows have g=-1 and never change the compiled topology.
            valid = ~jnp.isnan(obstacles[:, 0])
            safe_obstacles = jnp.where(valid[:, None], obstacles, 0.)
            # Keep legacy ordering: agent, obstacle, prediction step.
            for i in range(self.num_agents):
                for k in range(self.max_obstacles):
                    squared = jnp.sum((predicted[i, :, :3] - safe_obstacles[k])**2, axis=1)
                    parts.append(jnp.where(valid[k], 0.5**2-squared, -jnp.ones(self.horizon)))
        if self._hard_pair_count:
            for i in range(self.num_agents):
                for j in range(i+1, self.num_agents):
                    squared = jnp.sum((predicted[i, :, :3] - predicted[j, :, :3])**2, axis=1)
                    parts.append(self.d_min**2-squared)
        return jnp.concatenate(parts) if parts else jnp.empty(0)

    def _solve_game(self, current_states, targets, obstacles, return_diagnostics, max_nfev):
        if obstacles is not None:
            self._obs_padded = self._pad_obstacles(obstacles)
        states = np.asarray(current_states, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if states.shape != (self.num_agents, self.nx):
            raise ValueError(f"Expected states with shape {(self.num_agents, self.nx)}.")
        if targets.ndim != 2 or targets.shape[0] != self._target_count or targets.shape[1] < 3:
            raise ValueError(f"Expected {self._target_count} targets with at least three coordinates.")
        parameters = np.concatenate([states.ravel(), targets[:, :3].ravel(), np.asarray(self._obs_padded).ravel()])
        guess = self._u_prev if self._u_prev is not None else np.zeros(self._game.nvar)
        started = perf_counter()
        solution = self._game.solve(parameters, x0=guess, max_nfev=max_nfev)
        elapsed = perf_counter()-started
        controls = solution.x.reshape(self.num_agents, self.horizon, self.nu)
        self._u_prev = np.concatenate([
            np.concatenate([controls[i, 1:].ravel(), np.zeros(self.nu)])
            for i in range(self.num_agents)
        ])
        predictions = (states @ np.asarray(self.Sx).T
                       + controls.reshape(self.num_agents, -1) @ np.asarray(self.Su).T)
        predictions = predictions.reshape(self.num_agents, self.horizon, self.nx)
        info = {"compute_time": elapsed, "kkt_residual": solution.norm_residual}
        if return_diagnostics:
            info["solver_evaluations"] = solution.stats.kkt_evals
            info["solver_status"] = solution.stats.status
            if self._game.ng:
                info["shared_inequalities"] = np.asarray(self._game.problem.g(solution.x, jnp.asarray(parameters)))
                info["shared_multipliers"] = np.asarray([v[:self._game.ng] for v in solution.lam])
        return controls, list(predictions), info
