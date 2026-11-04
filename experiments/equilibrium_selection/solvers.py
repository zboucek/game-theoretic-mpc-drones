# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček
"""Controlled-study solver policies: article III-B/D and IV-A.

GNE/vGNE use reusable NashOpt residuals with TRF least squares; the exact
potential uses SLSQP. NumpyModel checks outputs, and deviation_attempt searches
for feasible unilateral improvements with fixed opponent controls."""
import time
import traceback

import numpy as np
from scipy.optimize import least_squares, minimize

from .model import NumpyModel
from .protocol import INITIALIZATIONS, PROTOCOL, initial_controls


class TimeLimit(RuntimeError):
    pass


class Deadline:
    def __init__(self, limit, x0, clock=None):
        self.clock = clock or time.perf_counter
        self.started = self.clock()
        self.limit = limit
        self.last = np.asarray(x0).copy()
        self.calls = 0

    def __call__(self, x):
        self.last = np.asarray(x).copy()
        self.calls += 1
        if self.clock()-self.started > self.limit:
            raise TimeLimit("Soft wall-time limit exceeded at iteration callback")


def timed_optimizer(optimize, x0, limit):
    """Return the last callback iterate even after timeouts and exceptions."""
    deadline = Deadline(limit, x0)
    try:
        result = optimize(deadline)
        x = np.asarray(result.x)
        info = {"status": int(result.status), "success": bool(result.success),
                "message": str(result.message), "nfev": int(result.nfev),
                "iterations": int(getattr(result, "nit", deadline.calls)),
                "exception": None}
    except Exception as error:
        x = deadline.last
        info = {"status": "time_limit" if isinstance(error, TimeLimit) else "exception",
                "success": False, "message": str(error), "nfev": None,
                "iterations": deadline.calls, "exception": type(error).__name__,
                "traceback": traceback.format_exc()}
    elapsed = deadline.clock()-deadline.started
    info.update(solve_seconds=elapsed, limit_seconds=limit,
                limit_exceeded=bool(elapsed > limit), overrun_seconds=max(0., elapsed-limit),
                callback_calls=deadline.calls)
    return x, info


class Solvers:
    def __init__(self, config=PROTOCOL):
        started = time.perf_counter()
        self.config = config
        self.numpy = NumpyModel(config)
        self.controllers = {}
        self.preparations = []
        self.numpy_preparation_seconds = time.perf_counter()-started

    def prepare(self, method, states, targets):
        if method in self.controllers:
            return None
        import_started = time.perf_counter()
        import jax
        import jax.numpy as jnp
        from game_theory_mpc.nashopt_controller import NashOptController

        if not jax.config.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)
        started = time.perf_counter()
        controller = NashOptController(3, horizon=self.config.horizon, dt=self.config.dt,
            variational=method != "gne", use_hard_collision=True,
            d_min=self.config.separation, max_obstacles=0,
            effort_weight=self.config.effort_weight, repulsion_offset=self.config.repulsion_offset,
            input_bound=self.config.bound)
        constructed = time.perf_counter()
        parameters = jnp.asarray(np.concatenate([np.asarray(states).ravel(), np.asarray(targets).ravel()]))
        if method == "potential":
            def objective(u, p):
                states = p[:18].reshape(3, 6)
                targets = p[18:].reshape(3, 3)
                inputs = u.reshape(3, -1)
                predicted = (states@controller.Sx.T + inputs@controller.Su.T).reshape(3, self.config.horizon, 6)
                own = jnp.sum((predicted[:, :, :3]-targets[:, None, :])**2)
                own += self.config.effort_weight*jnp.sum(u**2)
                all_costs = controller._jit_cost(u, states, targets, jnp.empty((0, 3)))
                return .5*(jnp.sum(all_costs)+own)

            def constraints(u, p):
                return controller._shared_inequalities(u, p[:18].reshape(3, 6), jnp.empty((0, 3)))

            controller.potential = jax.jit(objective)
            controller.gradient = jax.jit(jax.grad(objective, argnums=0))
            controller.constraints = jax.jit(constraints)
            controller.constraint_jacobian = jax.jit(jax.jacfwd(constraints, argnums=0))
            # Match SciPy's NumPy callback inputs. JAX caches device-array and
            # NumPy-array calls separately, even at identical shape and dtype.
            warmup_controls = np.zeros(self.numpy.nvar)
            for function in (controller.potential, controller.gradient,
                             controller.constraints, controller.constraint_jacobian):
                function(warmup_controls, parameters).block_until_ready()
        else:
            game = controller._game
            z = np.concatenate([np.zeros(game.nvar), np.full(game.nlam_sum, .1)])
            game.residual(z, parameters).block_until_ready()
            game.jacobian(z, parameters).block_until_ready()
        finished = time.perf_counter()
        self.controllers[method] = controller
        record = {"method": method, "import_seconds": started-import_started,
                  "numpy_model_preparation_seconds": self.numpy_preparation_seconds,
                  "construction_seconds": constructed-started,
                  "compilation_seconds": finished-constructed, "measurement_excluded": True}
        self.preparations.append(record)
        return record

    def solve(self, method, states, targets, guess):
        import jax.numpy as jnp

        self.prepare(method, states, targets)
        controller = self.controllers[method]
        parameters = jnp.asarray(np.concatenate([np.asarray(states).ravel(), np.asarray(targets).ravel()]))
        guess = np.asarray(guess)
        initial = self.numpy.evaluate(guess, states, targets)
        if method == "potential":
            def optimize(callback):
                return minimize(lambda u: float(controller.potential(u, parameters)), guess,
                    jac=lambda u: np.asarray(controller.gradient(u, parameters)), method="SLSQP",
                    bounds=[(-self.config.bound, self.config.bound)]*len(guess),
                    constraints={"type": "ineq",
                        "fun": lambda u: -np.asarray(controller.constraints(u, parameters)),
                        "jac": lambda u: -np.asarray(controller.constraint_jacobian(u, parameters))},
                    callback=callback, options={"ftol": self.config.solver_tolerance,
                                               "maxiter": self.config.max_evaluations})
            controls, solver = timed_optimizer(optimize, guess, self.config.solve_limit_s)
            multipliers, solver_norm = None, None
        else:
            game = controller._game
            z0 = np.concatenate([guess, np.full(game.nlam_sum, .1)])
            def optimize(callback):
                return least_squares(lambda z: np.asarray(game.residual(z, parameters)), z0,
                    jac=lambda z: np.asarray(game.jacobian(z, parameters)), method="trf",
                    ftol=self.config.solver_tolerance, xtol=self.config.solver_tolerance,
                    gtol=self.config.solver_tolerance, max_nfev=self.config.max_evaluations,
                    callback=callback)
            decisions, solver = timed_optimizer(optimize, z0, self.config.solve_limit_s)
            controls = decisions[:game.nvar]
            multipliers = np.stack([decisions[game.nvar:][indices] for indices in game.problem.ii_lam])
            solver_norm = float(np.linalg.norm(game.residual(decisions, parameters)))
        started = time.perf_counter()
        try:
            quality, data, multipliers = self.numpy.validate(controls, states, targets, multipliers,
                                                           variational=method != "gne")
            validation_error = None
        except Exception as error:
            data = self.numpy.evaluate(controls, states, targets)
            quality = {"accepted": False}
            validation_error = {"type": type(error).__name__, "message": str(error)}
        return {"method": method, "states": np.asarray(states), "targets": np.asarray(targets),
                "initial_controls": guess, "initial_feasible": bool(
                    initial["g"].max() <= self.config.feasibility_tolerance
                    and np.max(np.abs(guess)) <= self.config.bound+self.config.feasibility_tolerance),
                "controls": controls, "multipliers": multipliers, "solver": solver,
                "solver_reported_kkt_norm": solver_norm, "validation": quality,
                "validation_error": validation_error, "validation_seconds": time.perf_counter()-started,
                "prediction": data["prediction"], "player_costs": data["costs"],
                "player_tracking": data["tracking"], "player_effort": data["effort"],
                "stage_costs": data["stage_costs"], "potential": data["potential"],
                "shared_inequalities": data["g"], "compiled_cache_entries": (
                    {"objective": controller.potential._cache_size(), "gradient": controller.gradient._cache_size(),
                     "constraints": controller.constraints._cache_size(), "jacobian": controller.constraint_jacobian._cache_size()}
                    if method == "potential" else
                    {"residual": controller._game.residual._cache_size(), "jacobian": controller._game.jacobian._cache_size()})}

    def derivative_check(self, states, targets):
        """Compare AD derivatives and prices to the independent NumPy model."""
        import jax
        import jax.numpy as jnp

        self.prepare("potential", states, targets)
        controller = self.controllers["potential"]
        p = jnp.asarray(np.concatenate([np.asarray(states).ravel(), np.asarray(targets).ravel()]))
        u = np.linspace(-.4, .4, self.numpy.nvar)
        data = self.numpy.evaluate(u, states, targets)
        pseudo = np.concatenate([np.asarray(controller._game.problem.df[i](
            jnp.asarray(u[i*self.numpy.width:(i+1)*self.numpy.width]), jnp.asarray(u), p)) for i in range(3)])
        grad = np.asarray(controller.gradient(u, p))
        errors = {"potential_pseudogradient_max_error": float(np.max(np.abs(pseudo-grad))),
                  "numpy_gradient_max_error": float(np.max(np.abs(data["gradient"]-grad))),
                  "numpy_jacobian_max_error": float(np.max(np.abs(data["jac"]-controller.constraint_jacobian(u, p)))),
                  "numpy_potential_error": float(abs(data["potential"]-controller.potential(u, p))),
                  "numpy_player_cost_max_error": float(np.max(np.abs(data["costs"]-controller._jit_cost(
                      u, jnp.asarray(states), jnp.asarray(targets), jnp.empty((0, 3))))))}
        errors["passed"] = all(v <= 1e-10 for v in errors.values())
        errors["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
        return errors

    def parameter_reuse_check(self, method, states, targets):
        """Exercise changed states AND targets before measurement, without a solve."""
        import jax.numpy as jnp

        self.prepare(method, states, targets)
        controller = self.controllers[method]
        u = np.linspace(-.2, .3, self.numpy.nvar)
        errors = []
        cache_sizes = []
        for offset in (0., .003):
            changed_states = np.asarray(states)+offset
            changed_targets = np.asarray(targets)-offset
            p = jnp.asarray(np.concatenate([changed_states.ravel(), changed_targets.ravel()]))
            data = self.numpy.evaluate(u, changed_states, changed_targets)
            predicted = (changed_states@np.asarray(controller.Sx).T
                         +u.reshape(3, -1)@np.asarray(controller.Su).T).reshape(3, self.config.horizon, 6)
            errors.append(float(np.max(np.abs(predicted-data["prediction"]))))
            if method == "potential":
                errors.append(float(np.max(np.abs(controller.gradient(u, p)-data["gradient"]))))
                errors.append(float(np.max(np.abs(controller.constraints(u, p)-data["g"]))))
                controller.potential(u, p).block_until_ready()
                controller.constraint_jacobian(u, p).block_until_ready()
                cache_sizes.extend(f._cache_size() for f in (controller.potential, controller.gradient,
                    controller.constraints, controller.constraint_jacobian))
            else:
                game = controller._game
                z = np.concatenate([u, np.full(game.nlam_sum, .1)])
                lam = np.stack([z[game.nvar:][idx] for idx in game.problem.ii_lam])
                expected = self.numpy.kkt(data, u, lam, method == "vgne")["kkt_fb_norm"]
                errors.append(float(abs(np.linalg.norm(game.residual(z, p))-expected)))
                game.jacobian(z, p).block_until_ready()
                cache_sizes.extend([game.residual._cache_size(), game.jacobian._cache_size()])
        return {"maximum_independent_error": max(errors), "cache_entries": cache_sizes,
                "passed": bool(max(errors) <= 1e-10 and all(n == 1 for n in cache_sizes))}


def deviation_jobs(record, geometry, config=PROTOCOL):
    """Seven starts/player, preserving duplicate starts as prescribed attempts."""
    for player in range(3):
        controls = np.asarray(record["controls"]).reshape(3, config.horizon*3)
        yield player, "candidate", controls[player].copy()
        for name in INITIALIZATIONS:
            guess = initial_controls(record["states"], geometry, name, config).reshape(3, -1)
            yield player, name, guess[player].copy()


def deviation_attempt(record, player, start_name, guess, config=PROTOCOL):
    """A feasible improvement disproves GNE; local search is never a certificate."""
    model = NumpyModel(config)
    fixed = np.asarray(record["controls"]).reshape(3, model.width).copy()
    states, targets = record["states"], record["targets"]
    def evaluate(own):
        joint = fixed.copy()
        joint[player] = own
        return model.evaluate(joint.ravel(), states, targets)
    own_slice = slice(player*model.width, (player+1)*model.width)
    def optimize(callback):
        return minimize(lambda own: evaluate(own)["costs"][player], guess, method="SLSQP",
            jac=lambda own: evaluate(own)["gradient"][own_slice],
            bounds=[(-config.bound, config.bound)]*model.width,
            constraints={"type": "ineq", "fun": lambda own: -evaluate(own)["g"],
                         "jac": lambda own: -evaluate(own)["jac"][:, own_slice]},
            callback=callback, options={"ftol": config.solver_tolerance,
                                        "maxiter": config.deviation_iterations})
    own, solver = timed_optimizer(optimize, guess, config.deviation_limit_s)
    data = evaluate(own)
    baseline = model.evaluate(fixed.ravel(), states, targets)["costs"][player]
    improvement = float(baseline-data["costs"][player])
    feasible = bool(np.isfinite(own).all() and np.isfinite(data["g"]).all()
        and data["g"].max() <= config.feasibility_tolerance
        and np.max(np.abs(own)) <= config.bound+config.feasibility_tolerance)
    return {"player": player, "start": start_name, "own_controls": own,
            "solver": solver, "baseline_cost": float(baseline),
            "deviation_cost": float(data["costs"][player]), "improvement": improvement,
            "feasible": feasible, "threshold": max(1e-5, 1e-6*abs(baseline)),
            "refutes_gne": bool(feasible and improvement > max(1e-5, 1e-6*abs(baseline))),
            "geometric_violation_m2": float(max(0, data["g"].max())),
            "minimum_predicted_distance_m": data["minimum_predicted_distance_m"]}
