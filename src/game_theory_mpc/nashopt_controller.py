# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček

"""NashOpt residual backend for the article's KKT method (III-B/D) and controlled study (IV-A).

_reusable_game binds UAV costs and constraints to reusable_gne; the blocked
quadrotor controller is implemented in cooperative_equilibrium_mpc."""
import numpy as np
import jax.numpy as jnp
import jax
from ._reusable_game import ReusableGameMixin
from ._validation import positive_integer, nonnegative_integer, positive_number, states_and_targets, obstacles_array

class NashOptController(ReusableGameMixin):
    """
    NashOpt-based MPC controller for multi-agent swarm control.
    Uses JAX JIT and precomputed matrices for 3D double integrator dynamics.
    """
    def __init__(self, num_agents, horizon=10, dt=0.1, obstacles=None,
                 variational=False, use_hard_constraints=False,
                 use_hard_collision=False, d_min=0.5, max_obstacles=10, *, effort_weight=.1,
                 repulsion_offset=.01, input_bound=5.):
        positive_integer(num_agents, "num_agents")
        positive_integer(horizon, "horizon")
        nonnegative_integer(max_obstacles, "max_obstacles")
        positive_number(dt, "dt")
        positive_number(d_min, "d_min")
        positive_number(effort_weight, "effort_weight")
        positive_number(repulsion_offset, "repulsion_offset")
        positive_number(input_bound, "input_bound")
        self.repulsion_offset = repulsion_offset
        self.input_bound = input_bound
        self.num_agents = num_agents
        self.horizon = horizon
        self.dt = dt
        self.variational = variational
        self.use_hard_constraints = use_hard_constraints
        self.use_hard_collision = use_hard_collision
        self.d_min = d_min
        self.max_obstacles = max_obstacles
        self.nx = 6  # [px, py, pz, vx, vy, vz]
        self.nu = 3  # [ax, ay, az]

        # Precompute dynamics matrices: X = Sx*x0 + Su*U
        A = jnp.eye(6) + jnp.diag(jnp.array([dt]*3), k=3)
        B = jnp.vstack([0.5 * dt**2 * jnp.eye(3), dt * jnp.eye(3)])

        Sx_list = []
        curr_A = A
        for k in range(horizon):
            Sx_list.append(curr_A)
            curr_A = curr_A @ A

        Su = jnp.zeros((horizon * self.nx, horizon * self.nu))
        for i in range(horizon):
            for j in range(i + 1):
                block = jnp.linalg.matrix_power(A, i-j) @ B
                Su = Su.at[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu].set(block)

        self.Sx = jnp.vstack(Sx_list)
        self.Su = Su

        self.Q = jnp.eye(self.nx)
        self.R = jnp.eye(self.nu) * effort_weight

        # Pad obstacles to fixed size to avoid JAX retracing on shape changes.
        # NaN rows mark unused slots.
        self._obs_padded = self._pad_obstacles(obstacles)

        # Warm-start state: None until first solve
        self._u_prev = None

        self._jit_cost = jax.jit(self._compute_joint_cost)
        self._initialize_game(self.num_agents)

    def _pad_obstacles(self, obstacles):
        obstacles = obstacles_array(obstacles, self.max_obstacles)
        padded = np.full((self.max_obstacles, 3), np.nan)
        if obstacles is not None and np.asarray(obstacles).size:
            obs = np.atleast_2d(np.asarray(obstacles))
            n = min(len(obs), self.max_obstacles)
            padded[:n] = obs[:n, :3]
        return jnp.asarray(padded)

    def _compute_joint_cost(self, u_full, current_states, targets, obs_padded):
        """
        Cost for each agent. obs_padded has fixed shape (max_obstacles, 3)
        with NaN rows for unused slots — avoids JAX retracing on obstacle count changes.
        """
        u_all = u_full.reshape((self.num_agents, self.horizon * self.nu))
        valid_obs = ~jnp.isnan(obs_padded[:, 0])  # (max_obstacles,)

        trajs = []
        for i in range(self.num_agents):
            traj_flat = self.Sx @ current_states[i] + self.Su @ u_all[i]
            trajs.append(traj_flat.reshape((self.horizon, self.nx)))

        costs = []
        for i in range(self.num_agents):
            traj_i = trajs[i]
            u_i = u_all[i]

            pos_err = traj_i[:, :3] - targets[i][:3]
            cost = jnp.sum(pos_err**2)
            cost += self.R[0, 0] * jnp.sum(u_i**2)

            # The soft pair penalty is retained alongside hard separation.
            for j in range(self.num_agents):
                if i == j:
                    continue
                dist_sq = jnp.sum((traj_i[:, :3] - trajs[j][:, :3])**2, axis=1)
                cost += jnp.sum(1.0 / (dist_sq + self.repulsion_offset))

            # Soft obstacle barrier (used when use_hard_constraints=False)
            if not self.use_hard_constraints:
                for k in range(self.max_obstacles):
                    obs_p = jnp.where(valid_obs[k], obs_padded[k], jnp.zeros(3))
                    dist_sq = jnp.sum((traj_i[:, :3] - obs_p)**2, axis=1)
                    penalty = jnp.sum(5.0 / (dist_sq + 1e-2))
                    cost += jnp.where(valid_obs[k], penalty, 0.0)

            costs.append(cost)
        return jnp.array(costs)

    def solve(self, current_states, target_positions, obstacles=None,
              return_diagnostics=False, max_nfev=200):
        """Solve the joint KKT system at current states and position targets.

        Accept states (N,6) and target rows of length 3 or 6 (position used).
        Return controls (N,H,3), a list of N predictions (H,6), and an info
        dictionary with compute_time in seconds and kkt_residual. Optional
        diagnostics add solver status and shared multipliers/inequalities.
        The shifted warm start updates here; a residual is not a global Nash test."""
        positive_integer(max_nfev, "max_nfev")
        current_states, target_positions = states_and_targets(
            current_states, target_positions, self.num_agents)
        return self._solve_game(current_states, target_positions, obstacles,
                                return_diagnostics, max_nfev)
