# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Independent NumPy dynamics, costs and numerical checks for article V-A.

NumpyModel evaluates stored controls and reconstructs multiplier conditions.
The JAX/NashOpt optimization path is in solvers.py, keeping checks independent."""
import numpy as np
from scipy.optimize import nnls

from .protocol import PAIRS, PROTOCOL


def step(states, controls, config=PROTOCOL):
    result = np.asarray(states, dtype=float).copy()
    controls = np.asarray(controls).reshape(3, 3)
    result[:, :3] += config.dt * result[:, 3:] + .5 * config.dt**2 * controls
    result[:, 3:] += config.dt * controls
    return result


class NumpyModel:
    def __init__(self, config=PROTOCOL):
        self.config = config
        self.width = config.horizon * 3
        self.nvar = 3 * self.width
        self.ng = len(PAIRS) * config.horizon
        self.sensitivity = np.zeros((config.horizon, 3, self.width))
        for k in range(config.horizon):
            for t in range(k + 1):
                self.sensitivity[k, :, 3*t:3*t+3] = (k-t+.5)*config.dt**2*np.eye(3)

    def predict(self, controls, states):
        controls = np.asarray(controls).reshape(3, self.config.horizon, 3)
        current = np.asarray(states, dtype=float)
        history = []
        for k in range(self.config.horizon):
            current = step(current, controls[:, k], self.config)
            history.append(current)
        return np.stack(history, axis=1)

    def evaluate(self, controls, states, targets):
        u = np.asarray(controls).reshape(3, self.config.horizon, 3)
        prediction = self.predict(u, states)
        errors = prediction[:, :, :3] - np.asarray(targets)[:, None, :3]
        tracking = np.sum(errors**2, axis=(1, 2))
        effort = np.sum(u**2, axis=(1, 2))
        stage = np.sum(errors**2, axis=2) + self.config.effort_weight*np.sum(u**2, axis=2)
        costs = tracking + self.config.effort_weight * effort
        pseudo = 2*np.einsum("ikd,kdw->iw", errors, self.sensitivity)
        pseudo += 2*self.config.effort_weight*u.reshape(3, -1)
        jac = np.zeros((self.ng, self.nvar))
        inequalities, pair_penalties = [], []
        for pair_index, (i, j) in enumerate(PAIRS):
            delta = prediction[i, :, :3] - prediction[j, :, :3]
            squared = np.sum(delta**2, axis=1)
            penalty = 1/(squared + self.config.repulsion_offset)
            pair_penalties.append(penalty)
            costs[i] += penalty.sum()
            costs[j] += penalty.sum()
            stage[i] += penalty
            stage[j] += penalty
            derivative = -2*np.einsum("kd,kdw->kw", delta, self.sensitivity)
            pseudo[i] += np.sum(derivative/(squared[:, None]+self.config.repulsion_offset)**2, axis=0)
            pseudo[j] -= np.sum(derivative/(squared[:, None]+self.config.repulsion_offset)**2, axis=0)
            rows = slice(pair_index*self.config.horizon, (pair_index+1)*self.config.horizon)
            jac[rows, i*self.width:(i+1)*self.width] = derivative
            jac[rows, j*self.width:(j+1)*self.width] = -derivative
            inequalities.extend(self.config.separation**2-squared)
        return {"prediction": prediction, "costs": costs, "tracking": tracking,
                "effort": effort, "stage_costs": stage,
                "potential": float(tracking.sum()+self.config.effort_weight*effort.sum()
                                   +np.sum(pair_penalties)),
                "gradient": pseudo.ravel(), "g": np.asarray(inequalities), "jac": jac,
                "minimum_predicted_distance_m": float(np.sqrt(self.config.separation**2-max(inequalities)))}

    def reconstruct_multipliers(self, data, controls):
        """NNLS on active geometric AND box normals, never signed LS clipping."""
        u = np.asarray(controls).ravel()
        g = np.concatenate([data["g"], -self.config.bound-u, u-self.config.bound])
        jac = np.vstack([data["jac"], -np.eye(self.nvar), np.eye(self.nvar)])
        active = g >= -self.config.activity_tolerance
        multipliers = np.zeros(len(g))
        if active.any():
            multipliers[active], _ = nnls(jac[active].T, -data["gradient"],
                                          maxiter=20*len(g))
        shared = multipliers[:self.ng]
        lower, upper = np.split(multipliers[self.ng:], 2)
        return np.stack([np.concatenate([shared, lower[i*self.width:(i+1)*self.width],
                                        upper[i*self.width:(i+1)*self.width]]) for i in range(3)])

    def kkt(self, data, controls, multipliers, variational):
        multipliers = np.asarray(multipliers)
        if multipliers.shape != (3, self.ng+2*self.width):
            raise ValueError("Incorrect player multiplier layout")
        u = np.asarray(controls).reshape(3, self.width)
        residual = []
        stationarity = []
        for i in range(3):
            lam = multipliers[i]
            own = slice(i*self.width, (i+1)*self.width)
            stat = (data["gradient"][own] + data["jac"][:, own].T@lam[:self.ng]
                    - lam[self.ng:self.ng+self.width] + lam[self.ng+self.width:])
            g = np.concatenate([data["g"], -self.config.bound-u[i], u[i]-self.config.bound])
            fb = np.hypot(lam, g) - lam + g
            stationarity.extend(stat)
            residual.extend(np.concatenate([stat, fb]))
        common_error = float(np.max(np.abs(multipliers[:, :self.ng]-multipliers[0, :self.ng])))
        if variational:
            residual.extend((multipliers[1:, :self.ng]-multipliers[0, :self.ng]).ravel())
        return {"kkt_fb_norm": float(np.linalg.norm(residual)),
                "stationarity_norm": float(np.linalg.norm(stationarity)),
                "shared_multiplier_difference": common_error,
                "minimum_multiplier": float(multipliers.min())}

    def validate(self, controls, states, targets, multipliers=None, variational=True):
        data = self.evaluate(controls, states, targets)
        if multipliers is None:
            multipliers = self.reconstruct_multipliers(data, controls)
        multipliers = np.asarray(multipliers, dtype=float)
        quality = self.kkt(data, controls, multipliers, variational)
        quality.update(geometric_violation_m2=float(max(0, data["g"].max())),
                       bound_violation=float(max(0, np.max(np.abs(controls))-self.config.bound)),
                       minimum_predicted_distance_m=data["minimum_predicted_distance_m"],
                       active_constraints=int(np.sum(np.abs(data["g"]) <= self.config.activity_tolerance)))
        participant_differences = []
        positive_active = 0
        for pair_index, (i, j) in enumerate(PAIRS):
            rows = slice(pair_index*self.config.horizon, (pair_index+1)*self.config.horizon)
            geometric_active = np.abs(data["g"][rows]) <= self.config.activity_tolerance
            participant_active = geometric_active & (np.maximum(multipliers[i, rows], multipliers[j, rows]) > 1e-6)
            positive_active += int(np.sum(participant_active))
            participant_differences.extend(np.abs(multipliers[i, rows]-multipliers[j, rows])[participant_active])
        quality["positive_active_constraints"] = positive_active
        quality["max_participant_multiplier_difference_on_active"] = float(max(participant_differences, default=0.))
        finite = all(np.isfinite(v).all() for v in (controls, multipliers, data["prediction"],
                                                  data["costs"], data["gradient"]))
        quality["finite"] = bool(finite)
        quality["accepted"] = bool(finite and quality["kkt_fb_norm"] <= self.config.kkt_tolerance
            and quality["geometric_violation_m2"] <= self.config.feasibility_tolerance
            and quality["bound_violation"] <= self.config.feasibility_tolerance
            and quality["minimum_multiplier"] >= -self.config.feasibility_tolerance)
        return quality, data, multipliers
