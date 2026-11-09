# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček

"""Reusable NashOpt KKT residuals and Jacobians: article sections III-B, IV-B and V-A.

ParametricGNEP provides player derivatives, bound complementarity and the
GNE/vGNE multiplier layout. Runtime parameters carry states and targets;
equilibrium_selection/solvers.py supplies the controlled-study solve policy."""

from time import perf_counter
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from nashopt import ParametricGNEP
from scipy.optimize import least_squares


class ReusableGNEP:
    """Compile one fixed game structure and supply changing data as parameters.

    Costs have signature f_i(controls, parameters), and shared inequalities
    have signature g(controls, parameters). Sizes and constraint topology are
    fixed for the lifetime of the object. Changing numeric parameters does
    not create a new compiled residual or Jacobian.
    """

    def __init__(self, sizes, f, *, npar, g=None, ng=0, lb=None, ub=None,
                 variational=False):
        if npar < 1:
            raise ValueError("At least one runtime parameter is required.")
        self.problem = ParametricGNEP(sizes, f, npar=npar, g=g, ng=ng,
                                     lb=lb, ub=ub, variational=variational)
        self.npar = int(npar)
        self.nvar = self.problem.nvar
        self.nlam_sum = self.problem.nlam_sum
        self.ng = self.problem.ng

        def residual(decisions, parameters):
            return self.problem.kkt_residual(jnp.concatenate([decisions, parameters]))

        self.residual = jax.jit(residual)
        # Differentiate only controls/multipliers: parameters stay fixed in
        # the inner game. Keep NashOpt's reverse-mode Jacobian construction.
        self.jacobian = jax.jit(jax.jacobian(residual, argnums=0))

    def solve(self, parameters, *, x0=None, max_nfev=200, tol=1e-12,
              solver="trf", verbose=0):
        """Solve at fixed parameters with the original multiplier initialization."""
        started = perf_counter()
        parameters = jnp.asarray(parameters, dtype=jnp.float64)
        if parameters.shape != (self.npar,):
            raise ValueError(f"Expected {self.npar} parameters, got {parameters.shape}.")
        inputs = np.zeros(self.nvar) if x0 is None else np.asarray(x0, dtype=float)
        if inputs.shape != (self.nvar,):
            raise ValueError(f"Expected {self.nvar} initial controls, got {inputs.shape}.")
        decisions = np.concatenate([inputs, np.full(self.nlam_sum, 0.1)])
        # These lightweight Python callbacks change per solve; the JAX
        # functions they call remain the same and take parameters explicitly.
        result = least_squares(
            lambda z: self.residual(z, parameters), decisions,
            jac=lambda z: self.jacobian(z, parameters), method=solver,
            ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev, verbose=verbose,
        )
        multipliers = result.x[self.nvar:]
        stats = SimpleNamespace(
            solver=solver, kkt_evals=int(result.nfev),
            elapsed_time=perf_counter()-started, status=int(result.status),
            message=str(result.message), success=bool(result.success),
        )
        return SimpleNamespace(
            x=np.asarray(result.x[:self.nvar]), res=np.asarray(result.fun),
            lam=[np.asarray(multipliers[indices]) for indices in self.problem.ii_lam],
            stats=stats, norm_residual=float(np.linalg.norm(result.fun)),
        )
