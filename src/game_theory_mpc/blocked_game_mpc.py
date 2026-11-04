# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Move-blocked UAV model and primal solvers: article IV-C and V-A/B.

MoveBlocking expands decisions to every prediction node. BlockedGameMPC
provides dynamics, costs and constraints; cooperative_equilibrium_mpc adds
GNE/vGNE selection. Adversarial solves use damped player responses."""
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize


@dataclass(frozen=True)
class AdversarialWeights:
    """Stage coefficients for goal tracking, evasion, pursuit and soft repulsion.

    All coefficients are nonnegative; goal, pursuit and effort must be positive.
    These defaults are not the article flight preset, which uses goal/evasion 100/10."""
    goal: float = 2.
    evasion: float = 50.
    pursuit: float = 10.
    repulsion: float = 2.
    obstacle: float = 5.
    effort: float = .1

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if not np.isfinite(values).all() or np.any(values < 0) or min(self.goal, self.pursuit, self.effort) <= 0:
            raise ValueError("Game weights must be finite and nonnegative, with positive tracking/effort")


@dataclass(frozen=True)
class CooperativeWeights:
    """Position, symmetric pair-repulsion and acceleration-effort coefficients.

    Velocity and terminal coefficients are separate planner constructor arguments."""
    position: float = 1.
    repulsion: float = 1.
    effort: float = .1

    def __post_init__(self):
        if (not np.isfinite([self.position, self.repulsion, self.effort]).all()
                or self.position <= 0 or self.repulsion < 0 or self.effort <= 0):
            raise ValueError("Cooperative position/effort must be positive and repulsion nonnegative")


@dataclass(frozen=True)
class MoveBlocking:
    """Piecewise-constant acceleration schedule in prediction intervals.

    Positive block lengths sum to H; B is their count. Array methods preserve
    leading agent dimensions and operate on the final time/block and xyz axes."""
    lengths: tuple[int, ...]

    def __post_init__(self):
        if not self.lengths or any(isinstance(n, bool) or not np.isfinite(n) or int(n) != n or n <= 0 for n in self.lengths):
            raise ValueError("Block lengths must be positive integers")
        object.__setattr__(self, "lengths", tuple(int(n) for n in self.lengths))

    @property
    def horizon(self):
        return sum(self.lengths)

    @property
    def blocks(self):
        return len(self.lengths)

    @property
    def indices(self):
        return np.repeat(np.arange(self.blocks), self.lengths)

    def expand(self, blocked):
        """Repeat each blocked acceleration over its prediction intervals.

        Map (..., B, 3) to (..., H, 3), where H is the sum of block lengths."""
        value = np.asarray(blocked)
        if value.shape[-2:] != (self.blocks, 3):
            raise ValueError("Expected (..., blocks, 3) controls")
        return value[..., self.indices, :]

    def compress(self, full):
        """Average full-node accelerations in each block: (..., H, 3) to (..., B, 3).

        This is a least-squares projection; it does not enforce model constraints."""
        value = np.asarray(full)
        if value.shape[-2:] != (self.horizon, 3):
            raise ValueError("Expected (..., horizon, 3) controls")
        boundaries = np.cumsum((0, *self.lengths))
        return np.stack([value[..., a:b, :].mean(axis=-2)
                         for a, b in zip(boundaries[:-1], boundaries[1:])], axis=-2)

    def shift(self, blocked):
        """Least-squares projection of the shifted sequence; feasibility is rechecked."""
        full = self.expand(blocked)
        shifted = np.concatenate([full[..., 1:, :], np.zeros_like(full[..., :1, :])], axis=-2)
        return self.compress(shifted)


def terminal_matrix(dt, position_weight, velocity_weight, effort_weight=.1):
    """Extra terminal cost for stages evaluated at the predicted next state.

    The last node already contains Q. Adding P-Q therefore gives the usual
    Riccati tail P without counting that terminal stage twice.
    """
    a = np.eye(6)
    a[:3, 3:] = dt*np.eye(3)
    b = np.vstack([.5*dt*dt*np.eye(3), dt*np.eye(3)])
    q = np.diag([position_weight]*3+[velocity_weight]*3)
    return solve_discrete_are(a, b, q, effort_weight*np.eye(3))-q


class BlockedGameMPC:
    """Condensed full-information game, with explicitly reported solve semantics.

    Cooperative games minimize their exact potential by SLSQP. Adversarial
    games use damped sequential best responses, also solving primal NLPs.
    Neither local numerical method is advertised as a global NE certificate.
    ``make_kkt_game`` exposes exactly the same blocked model for GNE/vGNE work.
    """
    def __init__(self, num_agents, block_lengths, *, num_pursuers=0, dt=.1,
                 velocity_weight=0., terminal_scale=0., obstacles=(),
                 hard_pairs=None, separation=.55, altitude_bounds=(.35, 1.95),
                 input_bound=5., max_iterations=200, response_sweeps=3,
                 adversarial_weights=None, speed_bound=None, jerk_bound=None,
                 cooperative_weights=None):
        """Build a fixed-size condensed model for cooperative or adversarial planning.

        block_lengths determines H and the number of decision blocks B. States
        are (N,6); controls are accelerations in m/s². Positive num_pursuers
        selects the adversarial model, with pursuers listed before evaders and
        only evaders receiving targets. Otherwise every agent receives a target.

        input_bound and jerk_bound apply per axis; speed_bound is a Euclidean
        velocity limit. All-node pair constraints are enabled by default for
        cooperation. velocity_weight and terminal_scale add the flight terms.
        Use explicit experiment settings to reproduce the article; constructor
        defaults are generic. See docs/API.md for output and execution semantics."""
        if not 0 <= num_pursuers < num_agents:
            raise ValueError("Need at least one non-pursuer agent")
        if (not np.isfinite([dt, velocity_weight, terminal_scale, input_bound]).all()
                or dt <= 0 or velocity_weight < 0 or terminal_scale < 0 or input_bound <= 0):
            raise ValueError("Invalid dynamics, weights or input bound")
        if altitude_bounds is not None and altitude_bounds[0] >= altitude_bounds[1]:
            raise ValueError("Invalid altitude bounds")
        if any(value is not None and (not np.isfinite(value) or value <= 0) for value in (speed_bound, jerk_bound)):
            raise ValueError("Speed and jerk bounds must be finite and positive")
        self.blocking = MoveBlocking(tuple(block_lengths))
        self.num_agents, self.num_pursuers = num_agents, num_pursuers
        self.num_evaders = num_agents-num_pursuers if num_pursuers else 0
        self.adversarial = num_pursuers > 0
        self.weights = adversarial_weights or AdversarialWeights()
        self.cooperative_weights = cooperative_weights or CooperativeWeights()
        self.effort_weight = self.weights.effort if self.adversarial else self.cooperative_weights.effort
        self.position_weight = self.weights.goal if self.adversarial else self.cooperative_weights.position
        self.speed_bound, self.jerk_bound = speed_bound, jerk_bound
        self._previous_controls = np.zeros((num_agents, 3))
        self.target_count = self.num_evaders if self.adversarial else num_agents
        self.horizon, self.dt = self.blocking.horizon, dt
        self.width = 3*self.blocking.blocks
        self.nvar = num_agents*self.width
        self.velocity_weight, self.terminal_scale = velocity_weight, terminal_scale
        self.hard_pairs = not self.adversarial if hard_pairs is None else hard_pairs
        self.separation, self.altitude_bounds, self.input_bound = separation, altitude_bounds, input_bound
        self.max_iterations, self.response_sweeps = max_iterations, response_sweeps
        self.obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 3)
        self.pairs = tuple(combinations(range(num_agents), 2))
        self.ng = (len(self.pairs)*self.horizon if self.hard_pairs else 0)
        self.ng += 2*num_agents*self.horizon if altitude_bounds is not None else 0
        self.ng += num_agents*self.horizon if speed_bound is not None else 0
        self.ng += 2*self.nvar if jerk_bound is not None else 0
        self.npar = 6*num_agents+3*self.target_count+3*len(self.obstacles)
        self.npar += 3*num_agents if jerk_bound is not None else 0
        # Eliminate states, then compose the full-input map with the blocking map.
        a = np.eye(6)
        a[:3, 3:] = dt*np.eye(3)
        b = np.vstack([.5*dt*dt*np.eye(3), dt*np.eye(3)])
        self.Sx = np.vstack([np.linalg.matrix_power(a, k+1) for k in range(self.horizon)])
        full_su = np.zeros((6*self.horizon, 3*self.horizon))
        for k in range(self.horizon):
            for t in range(k+1):
                full_su[6*k:6*k+6, 3*t:3*t+3] = np.linalg.matrix_power(a, k-t)@b
        expansion = np.kron(np.eye(self.blocking.blocks)[self.blocking.indices], np.eye(3))
        self.Su = full_su@expansion
        self.P = terminal_matrix(dt, self.position_weight, velocity_weight, self.effort_weight)
        self._warm = np.zeros((num_agents, self.blocking.blocks, 3))
        self.costs = jax.jit(self._costs)
        self.cost_jacobian = jax.jit(jax.jacrev(self._costs, argnums=0))
        self.constraints = jax.jit(self._constraints)
        self.constraint_jacobian = jax.jit(jax.jacfwd(self._constraints, argnums=0))
        self.potential = jax.jit(self._potential)
        self.potential_gradient = jax.jit(jax.grad(self._potential, argnums=0))

    def parameters(self, states, targets, obstacles=None):
        """Pack finite states, targets, obstacles and previous applied inputs.

        States have shape (N,6), targets (target_count,3). Obstacle count must
        match construction so compiled shapes stay fixed. Previous inputs are
        appended only when jerk bounds are enabled; update them via commit_controls."""
        states, targets = np.asarray(states, dtype=float), np.asarray(targets, dtype=float)
        obs = self.obstacles if obstacles is None else np.asarray(obstacles, dtype=float).reshape(-1, 3)
        if states.shape != (self.num_agents, 6) or targets.shape != (self.target_count, 3):
            raise ValueError("Incorrect state/target shape")
        if obs.shape != self.obstacles.shape or not all(np.isfinite(x).all() for x in (states, targets, obs)):
            raise ValueError("Nonfinite parameters or changed obstacle topology")
        values = [states.ravel(), targets.ravel(), obs.ravel()]
        if self.jerk_bound is not None:
            values.append(self._previous_controls.ravel())
        return np.concatenate(values)

    def commit_controls(self, controls):
        """Remember the actually applied acceleration, only after execution admission."""
        value = np.asarray(controls, dtype=float)
        if value.shape != (self.num_agents, 3) or not np.isfinite(value).all():
            raise ValueError("Expected finite applied accelerations of shape (agents, 3)")
        self._previous_controls = value.copy()

    def _unpack(self, parameters):
        a, b = 6*self.num_agents, 6*self.num_agents+3*self.target_count
        return (parameters[:a].reshape(self.num_agents, 6), parameters[a:b].reshape(self.target_count, 3),
                parameters[b:b+3*len(self.obstacles)].reshape(-1, 3))

    def _prediction(self, controls, parameters):
        states, _, _ = self._unpack(parameters)
        blocked = controls.reshape(self.num_agents, self.width)
        return (states@jnp.asarray(self.Sx).T+blocked@jnp.asarray(self.Su).T).reshape(self.num_agents, self.horizon, 6)

    def _terms(self, controls, parameters):
        _, targets, obstacles = self._unpack(parameters)
        prediction = self._prediction(controls, parameters)
        blocked = controls.reshape(self.num_agents, self.blocking.blocks, 3)
        full = blocked[:, self.blocking.indices, :]
        effort = self.effort_weight*jnp.sum(full**2, axis=(1, 2))
        position = prediction[:, :, :3]
        pair_sq = jnp.sum((position[:, None]-position[None, :])**2, axis=-1)
        diagonal = jnp.eye(self.num_agents, dtype=bool)[:, :, None]
        repulsion = jnp.sum(jnp.where(diagonal, 0., 1/(pair_sq+.01)), axis=(1, 2))
        tracking, evasion, velocity, terminal = [jnp.zeros(self.num_agents) for _ in range(4)]
        if self.adversarial:
            for i in range(self.num_agents):
                if i < self.num_pursuers:
                    nearest = jnp.min(pair_sq[i, self.num_pursuers:], axis=0)
                    tracking = tracking.at[i].set(self.weights.pursuit*jnp.sum(nearest))
                else:
                    own_target = targets[i-self.num_pursuers]
                    tracking = tracking.at[i].set(self.weights.goal*jnp.sum((position[i]-own_target)**2))
                    nearest = jnp.min(pair_sq[i, :self.num_pursuers], axis=0)
                    evasion = evasion.at[i].set(jnp.sum(self.weights.evasion/(nearest+.1)))
                    velocity = velocity.at[i].set(self.velocity_weight*jnp.sum(prediction[i, :, 3:]**2))
                    error = prediction[i, -1]-jnp.concatenate([own_target, jnp.zeros(3)])
                    terminal = terminal.at[i].set(self.terminal_scale*(error@jnp.asarray(self.P)@error))
            repulsion = self.weights.repulsion*repulsion
        else:
            tracking = self.position_weight*jnp.sum((position-targets[:, None])**2, axis=(1, 2))
            repulsion = self.cooperative_weights.repulsion*repulsion
            velocity = self.velocity_weight*jnp.sum(prediction[:, :, 3:]**2, axis=(1, 2))
            error = prediction[:, -1]-jnp.concatenate([targets, jnp.zeros_like(targets)], axis=1)
            terminal = self.terminal_scale*jnp.einsum("ni,ij,nj->n", error, jnp.asarray(self.P), error)
        obstacle = jnp.zeros(self.num_agents)
        if len(self.obstacles):
            squared = jnp.sum((position[:, :, None]-obstacles[None, None])**2, axis=-1)
            obstacle = (self.weights.obstacle if self.adversarial else 5.)*jnp.sum(1/(squared+.01), axis=(1, 2))
        return dict(tracking=tracking, effort=effort, repulsion=repulsion,
                    evasion=evasion, velocity=velocity, terminal=terminal, obstacle=obstacle)

    def _costs(self, controls, parameters):
        return sum(self._terms(controls, parameters).values())

    def _potential(self, controls, parameters):
        terms = self._terms(controls, parameters)
        return jnp.sum(self._costs(controls, parameters)-.5*terms["repulsion"])

    def _constraints(self, controls, parameters):
        prediction = self._prediction(controls, parameters)
        parts = []
        if self.hard_pairs:
            for i, j in self.pairs:
                parts.append(self.separation**2-jnp.sum((prediction[i, :, :3]-prediction[j, :, :3])**2, axis=1))
        if self.altitude_bounds is not None:
            height = prediction[:, :, 2].ravel()
            parts.extend([self.altitude_bounds[0]-height, height-self.altitude_bounds[1]])
        if self.speed_bound is not None:
            parts.append((jnp.sum(prediction[:, :, 3:]**2, axis=2)-self.speed_bound**2).ravel())
        if self.jerk_bound is not None:
            blocked = controls.reshape(self.num_agents, self.blocking.blocks, 3)
            previous = parameters[-3*self.num_agents:].reshape(self.num_agents, 1, 3)
            change = jnp.diff(jnp.concatenate([previous, blocked], axis=1), axis=1).ravel()
            parts.extend([change-self.dt*self.jerk_bound, -change-self.dt*self.jerk_bound])
        return jnp.concatenate(parts) if parts else jnp.empty(0)

    def constraint_rows(self, player):
        """Rows affected by a player's decision, in the full constraint ordering."""
        rows, offset = [], 0
        if self.hard_pairs:
            for index, pair in enumerate(self.pairs):
                if player in pair:
                    rows.extend(range(index*self.horizon, (index+1)*self.horizon))
            offset += len(self.pairs)*self.horizon
        for enabled, count, repetitions in ((self.altitude_bounds is not None, self.horizon, 2),
                                            (self.speed_bound is not None, self.horizon, 1),
                                            (self.jerk_bound is not None, self.width, 2)):
            if enabled:
                for _ in range(repetitions):
                    rows.extend(range(offset+player*count, offset+(player+1)*count))
                    offset += self.num_agents*count
        return np.asarray(rows, dtype=int)

    def evaluate_numpy(self, controls, states, targets):
        """Independent dynamics/cost values, with every prediction node retained."""
        blocked = np.asarray(controls).reshape(self.num_agents, self.blocking.blocks, 3)
        full, current = self.blocking.expand(blocked), np.asarray(states, dtype=float).copy()
        history = []
        for k in range(self.horizon):
            current[:, :3] += self.dt*current[:, 3:]+.5*self.dt**2*full[:, k]
            current[:, 3:] += self.dt*full[:, k]
            history.append(current.copy())
        prediction = np.stack(history, axis=1)
        position = prediction[:, :, :3]
        terms = {name: np.zeros(self.num_agents) for name in ("tracking", "effort", "repulsion", "evasion", "velocity", "terminal", "obstacle")}
        terms["effort"] = self.effort_weight*np.sum(full**2, axis=(1, 2))
        constraints = []
        for i, j in self.pairs:
            squared = np.sum((position[i]-position[j])**2, axis=1)
            penalty = np.sum((self.weights.repulsion if self.adversarial else self.cooperative_weights.repulsion)/(squared+.01))
            terms["repulsion"][[i, j]] += penalty
            if self.hard_pairs:
                constraints.extend(self.separation**2-squared)
        for i in range(self.num_agents):
            if self.adversarial and i < self.num_pursuers:
                squared = np.sum((position[i]-position[self.num_pursuers:])**2, axis=2)
                terms["tracking"][i] = self.weights.pursuit*np.min(squared, axis=0).sum()
            else:
                target = np.asarray(targets)[i-self.num_pursuers if self.adversarial else i]
                terms["tracking"][i] = self.position_weight*np.sum((position[i]-target)**2)
                terms["velocity"][i] = self.velocity_weight*np.sum(prediction[i, :, 3:]**2)
                error = prediction[i, -1]-np.r_[target, np.zeros(3)]
                terms["terminal"][i] = self.terminal_scale*(error@self.P@error)
                if self.adversarial:
                    squared = np.sum((position[i]-position[:self.num_pursuers])**2, axis=2)
                    terms["evasion"][i] = np.sum(self.weights.evasion/(np.min(squared, axis=0)+.1))
            if len(self.obstacles):
                squared = np.sum((position[i, :, None]-self.obstacles[None])**2, axis=2)
                terms["obstacle"][i] = (self.weights.obstacle if self.adversarial else 5.)*np.sum(1/(squared+.01))
        if self.altitude_bounds is not None:
            height = prediction[:, :, 2].ravel()
            constraints.extend(self.altitude_bounds[0]-height)
            constraints.extend(height-self.altitude_bounds[1])
        if self.speed_bound is not None:
            constraints.extend((np.sum(prediction[:, :, 3:]**2, axis=2)-self.speed_bound**2).ravel())
        if self.jerk_bound is not None:
            previous = self._previous_controls[:, None]
            change = np.diff(np.concatenate([previous, blocked], axis=1), axis=1).ravel()
            constraints.extend(change-self.dt*self.jerk_bound)
            constraints.extend(-change-self.dt*self.jerk_bound)
        costs = sum(terms.values())
        return dict(prediction=prediction, full_controls=full, costs=costs, terms=terms,
                    potential=float(np.sum(costs-.5*terms["repulsion"])), g=np.asarray(constraints))

    def prepare(self, states, targets):
        """Prepare the residual/derivative graphs for these shapes before timing solves."""
        started = perf_counter()
        parameters = self.parameters(states, targets)
        zero = np.zeros(self.nvar)
        for function in (self.costs, self.cost_jacobian, self.constraints, self.constraint_jacobian):
            function(zero, parameters).block_until_ready()
        if not self.adversarial:
            self.potential(zero, parameters).block_until_ready()
            self.potential_gradient(zero, parameters).block_until_ready()
        return perf_counter()-started

    def make_kkt_game(self, variational=False):
        """Expose the same blocked costs and constraints as a reusable KKT game.

        variational=True shares multipliers on shared inequalities. This builds
        the game structure without running a solve or changing the primal planner."""
        from game_theory_mpc.reusable_gne import ReusableGNEP
        return ReusableGNEP([self.width]*self.num_agents,
            [lambda u, p, i=i: self.costs(u, p)[i] for i in range(self.num_agents)],
            npar=self.npar, g=self.constraints if self.ng else None, ng=self.ng,
            lb=np.full(self.nvar, -self.input_bound), ub=np.full(self.nvar, self.input_bound),
            variational=variational)

    def solve(self, states, targets):
        """Solve the current game and shift its warm start.

        Return expanded controls (N,H,3), predictions (N,H,6) and diagnostics.
        Call commit_controls only for the acceleration actually applied."""
        parameters = self.parameters(states, targets)
        started = perf_counter()
        attempts = []
        joint = self._warm.ravel().copy()
        constraints = ({"type": "ineq", "fun": lambda x: -np.asarray(self.constraints(x, parameters)),
                        "jac": lambda x: -np.asarray(self.constraint_jacobian(x, parameters))} if self.ng else ())

        if not self.adversarial:
            starts = [joint]
            if not getattr(self, "_solved", False) and self.num_agents > 1:
                for sign in (-1, 1):
                    seed = self._warm.copy()
                    seed[:2, :, 1] += sign*np.array([1., -1.])[:, None]
                    starts.append(seed.ravel())
            answers = []
            for guess in starts:
                result = minimize(lambda x: float(self.potential(x, parameters)), guess,
                    jac=lambda x: np.asarray(self.potential_gradient(x, parameters)), method="SLSQP",
                    bounds=[(-self.input_bound, self.input_bound)]*self.nvar, constraints=constraints,
                    options=dict(ftol=1e-9, maxiter=self.max_iterations))
                data = self.evaluate_numpy(result.x, states, targets)
                feasible = bool(np.isfinite(result.x).all() and (not len(data["g"]) or data["g"].max() <= 1e-7))
                answers.append((feasible, data["potential"], result.x))
                attempts.append(dict(success=bool(result.success), status=int(result.status), iterations=int(result.nit), feasible=feasible))
            feasible = [a for a in answers if a[0]]
            joint = min(feasible or answers, key=lambda a: a[1])[2]
            change = None
        else:
            # Each player minimizes its own primal objective. Damping applies
            # to the response, not to the stage or terminal cost definitions.
            change = 0.
            for sweep in range(self.response_sweeps):
                before_sweep = joint.copy()
                for player in range(self.num_agents):
                    own = slice(player*self.width, (player+1)*self.width)
                    fixed = joint.copy()
                    rows = self.constraint_rows(player)
                    def insert(x):
                        candidate = fixed.copy()
                        candidate[own] = x
                        return candidate
                    cons = ({"type": "ineq", "fun": lambda x: -np.asarray(self.constraints(insert(x), parameters))[rows],
                             "jac": lambda x: -np.asarray(self.constraint_jacobian(insert(x), parameters))[rows, own]} if len(rows) else ())
                    result = minimize(lambda x: float(self.costs(insert(x), parameters)[player]), fixed[own],
                        jac=lambda x: np.asarray(self.cost_jacobian(insert(x), parameters))[player, own],
                        method="SLSQP", bounds=[(-self.input_bound, self.input_bound)]*self.width,
                        constraints=cons, options=dict(ftol=1e-8, maxiter=self.max_iterations))
                    candidate = insert(result.x)
                    data = self.evaluate_numpy(candidate, states, targets)
                    feasible = bool(np.isfinite(result.x).all() and (not len(rows) or data["g"][rows].max() <= 1e-7))
                    old_cost = float(self.costs(fixed, parameters)[player])
                    old_g = np.asarray(self.constraints(fixed, parameters))[rows]
                    old_feasible = not len(rows) or old_g.max() <= 1e-7
                    # An infeasible warm start has no admissible reference cost.
                    # Restore feasibility even when that raises the objective.
                    improved = feasible and (not old_feasible or data["costs"][player] <= old_cost+1e-8)
                    used_fraction = 0.
                    if improved:
                        for fraction in (.7, .35, 1.):
                            trial = insert(fraction*result.x+(1-fraction)*fixed[own])
                            trial_data = self.evaluate_numpy(trial, states, targets)
                            if ((not old_feasible or trial_data["costs"][player] <= old_cost+1e-8)
                                    and (not len(rows) or trial_data["g"][rows].max() <= 1e-7)):
                                joint = trial
                                used_fraction = fraction
                                break
                    attempts.append(dict(sweep=sweep, player=player, success=bool(result.success),
                        status=int(result.status), iterations=int(result.nit), feasible=feasible,
                        restored_feasibility=bool(not old_feasible and used_fraction > 0),
                        improvement=float(old_cost-data["costs"][player]), used=used_fraction > 0,
                        fraction=used_fraction))
                change = float(np.max(np.abs(joint-before_sweep)))
                if change < 1e-3:
                    break
        data = self.evaluate_numpy(joint, states, targets)
        feasible = bool(np.isfinite(joint).all() and np.abs(joint).max() <= self.input_bound+1e-8
                        and (not len(data["g"]) or data["g"].max() <= 1e-7))
        self._warm = self.blocking.shift(joint.reshape(self.num_agents, self.blocking.blocks, 3))
        self._solved = True
        info = dict(compute_time=perf_counter()-started, feasible=feasible, costs=data["costs"],
                    cost_terms=data["terms"], blocked_controls=joint.reshape(self.num_agents, self.blocking.blocks, 3),
                    solver="damped_primal_best_response" if self.adversarial else "exact_potential_slsqp",
                    response_change=change, attempts=attempts,
                    interpretation="Local numerical candidate; global equilibrium is not certified")
        return data["full_controls"], data["prediction"], info
