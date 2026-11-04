# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Cooperative flight solver: article IV-B/C and V-A.

The model is inherited from blocked_game_mpc. CooperativeDerivatives supplies
analytic NumPy derivatives; CooperativeEquilibriumMPC selects GNE or vGNE,
polishes active constraints and checks unilateral responses."""
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares, minimize, nnls

from game_theory_mpc.blocked_game_mpc import BlockedGameMPC


class CooperativeDerivatives:
    """Explicit NumPy recurrence and analytic derivatives, independent of JAX."""
    def __init__(self, model):
        self.model = model
        n, h, w, nv = model.num_agents, model.horizon, model.width, model.nvar
        position, velocity = np.zeros((3, w)), np.zeros((3, w))
        ps, vs = [], []
        for block in model.blocking.indices:
            acceleration = np.zeros((3, w))
            acceleration[:, 3*block:3*block+3] = np.eye(3)
            position = position+model.dt*velocity+.5*model.dt**2*acceleration
            velocity = velocity+model.dt*acceleration
            ps.append(position.copy())
            vs.append(velocity.copy())
        self.ps, self.vs = np.asarray(ps), np.asarray(vs)
        self.end = np.vstack([self.ps[-1], self.vs[-1]])
        local = (2*model.position_weight*np.einsum("hdi,hdj->ij", self.ps, self.ps)
                 +2*model.velocity_weight*np.einsum("hdi,hdj->ij", self.vs, self.vs)
                 +2*model.effort_weight*np.diag(np.repeat(model.blocking.lengths, 3))
                 +2*model.terminal_scale*self.end.T@model.P@self.end)
        self.quadratic = np.kron(np.eye(n), local)
        self.constraint_hessians = np.zeros((model.ng+2*nv, nv, nv))
        offset = 0
        if model.hard_pairs:
            for i, j in model.pairs:
                ii, jj = slice(i*w, (i+1)*w), slice(j*w, (j+1)*w)
                for k in range(h):
                    value = -2*self.ps[k].T@self.ps[k]
                    self.constraint_hessians[offset, ii, ii] = value
                    self.constraint_hessians[offset, jj, jj] = value
                    self.constraint_hessians[offset, ii, jj] = -value
                    self.constraint_hessians[offset, jj, ii] = -value
                    offset += 1
        offset += 2*n*h if model.altitude_bounds is not None else 0
        if model.speed_bound is not None:
            for i in range(n):
                own = slice(i*w, (i+1)*w)
                for k in range(h):
                    self.constraint_hessians[offset, own, own] = 2*self.vs[k].T@self.vs[k]
                    offset += 1

    def evaluate(self, controls, states, targets, hessian=False):
        """Evaluate the cooperative model using an explicit NumPy recurrence.

        controls is the flattened (N,B,3) array. Returned gradient stacks each
        player's own-control derivatives; for this exact-potential game it also
        equals the potential gradient. Optional hessian is the potential Hessian.
        Constraint rows follow the blocked model: pairs, altitude, speed, jerk.
        Input bounds are appended by augmented, not by this method."""
        m, ps, vs = self.model, self.ps, self.vs
        n, h, w, nv = m.num_agents, m.horizon, m.width, m.nvar
        blocked = np.asarray(controls).reshape(n, m.blocking.blocks, 3)
        full = np.repeat(blocked, m.blocking.lengths, axis=1)
        current, history = np.asarray(states).copy(), []
        for k in range(h):
            current[:, :3] += m.dt*current[:, 3:]+.5*m.dt**2*full[:, k]
            current[:, 3:] += m.dt*full[:, k]
            history.append(current.copy())
        predicted = np.stack(history, axis=1)
        error, speed = predicted[:, :, :3]-np.asarray(targets)[:, None], predicted[:, :, 3:]
        end_error = predicted[:, -1]-np.c_[targets, np.zeros((n, 3))]
        terms = dict(tracking=m.position_weight*np.sum(error**2, axis=(1, 2)), effort=m.effort_weight*np.sum(full**2, axis=(1, 2)),
            velocity=m.velocity_weight*np.sum(speed**2, axis=(1, 2)),
            terminal=m.terminal_scale*np.einsum("ni,ij,nj->n", end_error, m.P, end_error),
            repulsion=np.zeros(n), evasion=np.zeros(n), obstacle=np.zeros(n))
        gradient = (2*m.position_weight*np.einsum("nhd,hdw->nw", error, ps)
            +2*m.velocity_weight*np.einsum("nhd,hdw->nw", speed, vs)
            +2*m.effort_weight*blocked.reshape(n, w)*np.repeat(m.blocking.lengths, 3)[None]
            +2*m.terminal_scale*end_error@m.P@self.end)
        matrix = self.quadratic.copy() if hessian else None
        inequalities, rows = [], []
        for i, j in m.pairs:
            delta = predicted[i, :, :3]-predicted[j, :, :3]
            squared = np.sum(delta**2, axis=1)
            denominator = squared+.01
            terms["repulsion"][[i, j]] += m.cooperative_weights.repulsion*np.sum(1/denominator)
            value = -2*m.cooperative_weights.repulsion*np.einsum("hd,hdw->hw", delta, ps)/denominator[:, None]**2
            gradient[i] += value.sum(axis=0)
            gradient[j] -= value.sum(axis=0)
            ii, jj = slice(i*w, (i+1)*w), slice(j*w, (j+1)*w)
            if matrix is not None:
                pair_hessian = np.zeros((w, w))
                for k in range(h):
                    local = -2*np.eye(3)/denominator[k]**2+8*np.outer(delta[k], delta[k])/denominator[k]**3
                    pair_hessian += m.cooperative_weights.repulsion*(ps[k].T@local@ps[k])
                matrix[ii, ii] += pair_hessian
                matrix[jj, jj] += pair_hessian
                matrix[ii, jj] -= pair_hessian
                matrix[jj, ii] -= pair_hessian
            if m.hard_pairs:
                inequalities.extend(m.separation**2-squared)
                derivative = np.zeros((h, nv))
                derivative[:, ii] = -2*np.einsum("hd,hdw->hw", delta, ps)
                derivative[:, jj] = -derivative[:, ii]
                rows.extend(derivative)
        if m.altitude_bounds is not None:
            for sign, bound in ((-1., m.altitude_bounds[0]), (1., m.altitude_bounds[1])):
                for i in range(n):
                    inequalities.extend(sign*(predicted[i, :, 2]-bound))
                    derivative = np.zeros((h, nv))
                    derivative[:, i*w:(i+1)*w] = sign*ps[:, 2]
                    rows.extend(derivative)
        if m.speed_bound is not None:
            for i in range(n):
                inequalities.extend(np.sum(speed[i]**2, axis=1)-m.speed_bound**2)
                derivative = np.zeros((h, nv))
                derivative[:, i*w:(i+1)*w] = 2*np.einsum("hd,hdw->hw", speed[i], vs)
                rows.extend(derivative)
        if m.jerk_bound is not None:
            difference = np.diff(np.concatenate([m._previous_controls[:, None], blocked], axis=1), axis=1).ravel()
            local = np.eye(w)
            local[3:, :-3] -= np.eye(w-3)
            derivative = np.kron(np.eye(n), local)
            for sign in (1., -1.):
                inequalities.extend(sign*difference-m.dt*m.jerk_bound)
                rows.extend(sign*derivative)
        costs = sum(terms.values())
        return dict(prediction=predicted, full_controls=full, costs=costs, terms=terms,
            potential=float(np.sum(costs-.5*terms["repulsion"])), gradient=gradient.ravel(),
            g=np.asarray(inequalities), jac=np.asarray(rows).reshape(m.ng, nv), hessian=matrix)

    def augmented(self, controls, states, targets, hessian=False):
        """Append all lower, then all upper input bounds to model inequalities.

        The g_all/jac_all ordering is also the multiplier column ordering.
        Every inequality uses g <= 0; lower bounds have derivative -I."""
        data = self.evaluate(controls, states, targets, hessian)
        m = self.model
        data["g_all"] = np.r_[data["g"], -m.input_bound-controls, controls-m.input_bound]
        data["jac_all"] = np.vstack([data["jac"], -np.eye(m.nvar), np.eye(m.nvar)])
        return data


class CooperativeEquilibriumMPC(BlockedGameMPC):
    """Select a cooperative candidate on the shared blocked model.

    method="gne" uses player responses; "vgne" uses the exact potential.
    info["feasible"] and info["validation"]["accepted"] are separate flags;
    the flight driver decides whether a feasible approximation may be applied."""
    def __init__(self, *args, method="gne", max_sweeps=30, repair_deviations=True, **kwargs):
        """Configure local GNE/vGNE selection on an obstacle-free cooperative model.

        All remaining settings are forwarded to BlockedGameMPC. max_sweeps
        limits sequential GNE responses; repair_deviations enables restarts from
        feasible improving unilateral witnesses, with at most four nested repairs."""
        super().__init__(*args, **kwargs)
        if self.adversarial or len(self.obstacles) or method not in ("gne", "vgne"):
            raise ValueError("This solver compares the cooperative obstacle-free GNE/vGNE game")
        self.method, self.max_sweeps = method, max_sweeps
        self.repair_deviations = repair_deviations
        self._repair_depth = 0
        self.derivatives = CooperativeDerivatives(self)

    def _multipliers(self, data):
        active = data["g_all"] >= -1e-6
        lam = np.zeros((self.num_agents, len(active)))
        if self.method == "vgne":
            if active.any():
                value, _ = nnls(data["jac_all"][active].T, -data["gradient"], maxiter=10000)
                lam[:, active] = value
        else:
            for i in range(self.num_agents):
                own = slice(i*self.width, (i+1)*self.width)
                relevant = active & (np.abs(data["jac_all"][:, own]).max(axis=1) > 1e-12)
                if relevant.any():
                    value, _ = nnls(data["jac_all"][relevant, own].T, -data["gradient"][own], maxiter=10000)
                    lam[i, relevant] = value
        return lam

    def _quality(self, data, lam):
        stats, fb = [], []
        for i in range(self.num_agents):
            own = slice(i*self.width, (i+1)*self.width)
            stats.extend(data["gradient"][own]+data["jac_all"][:, own].T@lam[i])
            # Omit constraints independent of this player from their KKT test.
            relevant = np.abs(data["jac_all"][:, own]).max(axis=1) > 1e-12
            g, value = data["g_all"][relevant], lam[i, relevant]
            fb.extend(np.hypot(value, g)-value+g)
        pair_difference = 0.
        if self.hard_pairs:
            for index, (i, j) in enumerate(self.pairs):
                rows = slice(index*self.horizon, (index+1)*self.horizon)
                active = (data["g"][rows] >= -1e-6) & (np.maximum(lam[i, rows], lam[j, rows]) > 1e-6)
                pair_difference = max(pair_difference, float(np.max(np.abs(lam[i, rows]-lam[j, rows])[active], initial=0.)))
        return dict(kkt_norm=float(np.linalg.norm(np.r_[stats, fb])),
            stationarity_norm=float(np.linalg.norm(stats)),
            maximum_constraint_violation=float(max(0., data["g_all"].max())),
            minimum_multiplier=float(lam.min()),
            collision_participant_multiplier_difference=pair_difference,
            shared_multiplier_difference=float(np.abs(lam[:, :self.ng]-lam[0, :self.ng]).max()))

    def _polish(self, controls, states, targets, data, lam):
        """Small active-set KKT system, initialized by primal minimizations."""
        active = list(np.flatnonzero(data["g_all"] >= -1e-6))
        original = (controls.copy(), data, lam.copy())
        for _ in range(5):
            labels = []
            for row in active:
                if self.method == "vgne":
                    labels.append((row, -1))
                else:
                    for player in range(self.num_agents):
                        own = slice(player*self.width, (player+1)*self.width)
                        if np.abs(data["jac_all"][row, own]).max() > 1e-12:
                            labels.append((row, player))
            cache = {}
            def values(z):
                if "z" not in cache or not np.array_equal(cache["z"], z):
                    d = self.derivatives.augmented(z[:self.nvar], states, targets, hessian=True)
                    multipliers = np.zeros_like(lam)
                    for value, (row, player) in zip(z[self.nvar:], labels):
                        multipliers[:, row] = value if player == -1 else multipliers[:, row]
                        if player >= 0:
                            multipliers[player, row] = value
                    stat, jac = d["gradient"].copy(), np.zeros((self.nvar+len(active), len(z)))
                    for player in range(self.num_agents):
                        own = slice(player*self.width, (player+1)*self.width)
                        stat[own] += d["jac_all"][:, own].T@multipliers[player]
                        jac[own, :self.nvar] = d["hessian"][own]+np.einsum("r,rij->ij", multipliers[player], self.derivatives.constraint_hessians[:, own])
                    for column, (row, player) in enumerate(labels):
                        if player == -1:
                            jac[:self.nvar, self.nvar+column] = d["jac_all"][row]
                        else:
                            own = slice(player*self.width, (player+1)*self.width)
                            jac[own, self.nvar+column] = d["jac_all"][row, own]
                    jac[self.nvar:, :self.nvar] = d["jac_all"][active]
                    cache.update(z=z.copy(), data=d, lam=multipliers, residual=np.r_[stat, d["g_all"][active]], jac=jac)
                return cache
            z0 = np.r_[controls, [lam[0 if player == -1 else player, row] for row, player in labels]]
            result = least_squares(lambda z: values(z)["residual"], z0, jac=lambda z: values(z)["jac"],
                                   ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=80)
            final = values(result.x)
            controls, data, lam = result.x[:self.nvar], final["data"], final["lam"]
            negative = [row for row in active if np.min(lam[:, row]) < -1e-7]
            violated = list(np.flatnonzero(data["g_all"] > 1e-7))
            if negative:
                active = [row for row in active if row not in negative]
                continue
            if any(row not in active for row in violated):
                active = sorted(set(active+violated))
                continue
            if data["g_all"].max() <= 1e-7 and self._quality(data, lam)["kkt_norm"] < self._quality(original[1], original[2])["kkt_norm"]:
                return controls, data, lam
            break
        return original

    def _response(self, player, controls, states, targets, guess=None):
        own = slice(player*self.width, (player+1)*self.width)
        fixed = controls.copy()
        def insert(value):
            joint = fixed.copy()
            joint[own] = value
            return joint
        rows = self.constraint_rows(player)
        cached = {}
        def evaluate(value):
            if "x" not in cached or not np.array_equal(value, cached["x"]):
                cached.update(x=value.copy(), data=self.derivatives.evaluate(insert(value), states, targets))
            return cached["data"]
        result = minimize(lambda x: evaluate(x)["costs"][player], fixed[own] if guess is None else guess,
            jac=lambda x: evaluate(x)["gradient"][own], method="SLSQP",
            bounds=[(-self.input_bound, self.input_bound)]*self.width,
            constraints=dict(type="ineq", fun=lambda x: -evaluate(x)["g"][rows],
                             jac=lambda x: -evaluate(x)["jac"][rows, own]),
            options=dict(ftol=1e-12, maxiter=self.max_iterations))
        candidate = insert(result.x)
        data = self.derivatives.evaluate(candidate, states, targets)
        feasible = bool(data["g"].max() <= 1e-7 and np.abs(candidate).max() <= self.input_bound+1e-8)
        return candidate, data, feasible

    @property
    def solver_name(self):
        return f"cooperative_{self.method}_primal_active_set"

    def _select(self, u, states, targets):
        """Produce a candidate and its multipliers before the common audit."""
        attempts = []
        if self.method == "vgne":
            result = minimize(lambda x: self.derivatives.evaluate(x, states, targets)["potential"], u,
                jac=lambda x: self.derivatives.evaluate(x, states, targets)["gradient"], method="SLSQP",
                bounds=[(-self.input_bound, self.input_bound)]*self.nvar,
                constraints=dict(type="ineq", fun=lambda x: -self.derivatives.evaluate(x, states, targets)["g"],
                                 jac=lambda x: -self.derivatives.evaluate(x, states, targets)["jac"]),
                options=dict(ftol=1e-12, maxiter=self.max_iterations))
            u = result.x
            attempts.append(dict(success=bool(result.success), iterations=int(result.nit), status=int(result.status)))
        else:
            for sweep in range(self.max_sweeps):
                before = u.copy()
                for player in range(self.num_agents):
                    baseline = self.derivatives.evaluate(u, states, targets)
                    candidate, candidate_data, feasible = self._response(player, u, states, targets)
                    if feasible and (baseline["g"].max() > 1e-7 or candidate_data["costs"][player] <= baseline["costs"][player]+1e-9):
                        u = candidate
                if np.max(np.abs(u-before)) < 1e-6:
                    break
            attempts.append(dict(sweeps=sweep+1, last_control_change=float(np.max(np.abs(u-before)))))
        data = self.derivatives.augmented(u, states, targets)
        lam = self._multipliers(data)
        if data["g_all"].max() <= 1e-7 and self._quality(data, lam)["kkt_norm"] > 1e-6:
            u, data, lam = self._polish(u, states, targets, data, lam)
        return u, data, lam, attempts, {}

    def solve(self, states, targets):
        """Run selection and unilateral checks with rejected restart candidates.

        Return full-node controls, predictions and diagnostics. Warm starts
        shift after this solve; execution admission is handled by the runner."""
        started = perf_counter()
        u = self._warm.ravel().copy()
        if not getattr(self, "_solved", False):
            seed = np.zeros_like(self._warm)
            if self.num_agents > 1:
                seed[0, :, 0], seed[1, :, 0] = -.6, .6
                seed[0, :, 1], seed[1, :, 1] = .3, -.3
            u = seed.ravel()
        initial = u.copy()
        data = self.derivatives.evaluate(u, states, targets)
        if data["g"].max() > 1e-7:
            origin = u.copy()
            result = minimize(lambda x: .5*np.sum((x-origin)**2), u, jac=lambda x: x-origin,
                method="SLSQP", bounds=[(-self.input_bound, self.input_bound)]*self.nvar,
                constraints=dict(type="ineq", fun=lambda x: -self.derivatives.evaluate(x, states, targets)["g"],
                                 jac=lambda x: -self.derivatives.evaluate(x, states, targets)["jac"]),
                options=dict(ftol=1e-12, maxiter=self.max_iterations))
            u = result.x
        projection_seconds = perf_counter()-started
        selected_at = perf_counter()
        u, data, lam, attempts, native = self._select(u, states, targets)
        solver_seconds = perf_counter()-selected_at
        checked_at = perf_counter()
        quality = self._quality(data, lam)
        feasible = bool(np.isfinite(u).all() and np.isfinite(data["g_all"]).all()
                        and data["g_all"].max() <= 1e-7)
        deviations = []
        if feasible:
            for player in range(self.num_agents):
                guesses = [("candidate", None)]
                if not getattr(self, "_solved", False) or self._repair_depth > 0:
                    for sign in (-1, 1):
                        guess = u.reshape(self.num_agents, self.blocking.blocks, 3)[player].copy()
                        guess[:, 1] += sign*.3
                        guesses.append((f"lateral_{sign:+}", np.clip(guess.ravel(), -self.input_bound, self.input_bound)))
                for start_name, guess in guesses:
                    changed, dev, valid = self._response(player, u, states, targets, guess)
                    improvement = float(data["costs"][player]-dev["costs"][player])
                    threshold = max(1e-6, 1e-7*abs(data["costs"][player]))
                    deviations.append(dict(player=player, start=start_name, feasible=valid, improvement=improvement,
                                           baseline_cost=float(data["costs"][player]), deviation_cost=float(dev["costs"][player]),
                                           own_blocked_controls=changed.reshape(self.num_agents, self.blocking.blocks, 3)[player],
                                           threshold=threshold,
                                           refutes_local_candidate=bool(valid and improvement > threshold)))
        improvements = [d for d in deviations if d["refutes_local_candidate"]]
        if improvements and self.repair_deviations and self._repair_depth < 4:
            # A feasible unilateral improvement is also a strict decrease of
            # the cooperative exact potential. Restart the same selection
            # method there rather than executing the already refuted point.
            witness = max(improvements, key=lambda d: d["improvement"])
            restart = u.reshape(self.num_agents, self.blocking.blocks, 3).copy()
            restart[witness["player"]] = witness["own_blocked_controls"]
            rejected = dict(**native, multipliers=lam.copy(), blocked_controls=u.reshape(self.num_agents, self.blocking.blocks, 3).copy(),
                            costs=data["costs"].copy(), validation=quality, witness=witness)
            own_checks = perf_counter()-checked_at
            self._warm, self._solved = restart, True
            self._repair_depth += 1
            try:
                controls, prediction, info = self.solve(states, targets)
            finally:
                self._repair_depth -= 1
            info["repair_history"] = [rejected, *info.get("repair_history", [])]
            info["initial_blocked_controls"] = initial.reshape(self.num_agents, self.blocking.blocks, 3)
            for key, value in (("projection_seconds", projection_seconds), ("solver_seconds", solver_seconds),
                               ("checks_seconds", own_checks)):
                info[key] += value
            info["compute_time"] = perf_counter()-started
            return controls, prediction, info
        accepted = bool(feasible and quality["kkt_norm"] <= 1e-5
                        and quality["minimum_multiplier"] >= -1e-7
                        and not any(d["refutes_local_candidate"] for d in deviations))
        self._warm = self.blocking.shift(u.reshape(self.num_agents, self.blocking.blocks, 3))
        self._solved = True
        info = dict(**native, projection_seconds=projection_seconds, solver_seconds=solver_seconds,
            checks_seconds=perf_counter()-checked_at, compute_time=perf_counter()-started, feasible=feasible, costs=data["costs"], cost_terms=data["terms"],
            blocked_controls=u.reshape(self.num_agents, self.blocking.blocks, 3), solver=self.solver_name,
            initial_blocked_controls=initial.reshape(self.num_agents, self.blocking.blocks, 3),
            attempts=attempts, response_change=None, multipliers=lam, validation=dict(accepted=accepted, **quality),
            deviations=deviations, interpretation="Numerically checked local candidate; global Nash optimality is not certified")
        return data["full_controls"], data["prediction"], info
