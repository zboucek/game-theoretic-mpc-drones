# SPDX-License-Identifier: MIT
# Copyright 2026 Zdeněk Bouček

"""Separable tracking backend for the five-drone integration (article V-C).

The fast and NashOpt/HiGHS paths solve the same unconstrained tracking model.
The supplied flight log is analyzed offline by tools/hardware.py."""
import os
import tempfile
import contextlib
import numpy as np
import time
from ._validation import positive_integer, positive_number, states_and_targets

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))


class LinearNashMPCController:
    """
    Linear MPC controller for 3D drone formation (linear baseline).

    Parameters
    ----------
    num_agents : int
    horizon    : int   — prediction horizon H
    dt         : float — sampling time [s]
    q_track    : float — position tracking weight Q_y (diagonal; NashOpt delta-input model)
    q_du       : float — input rate penalty R (diagonal; NashOpt delta-input model)
    backend    : str   — "fast" for precomputed unconstrained MPC,
                         "nashopt" for NashLinearMPC/HiGHS
    """

    def __init__(self, num_agents, horizon=10, dt=0.1,
                 q_track=10.0, q_du=0.1, backend=None, variational=False):
        positive_integer(num_agents, "num_agents")
        positive_integer(horizon, "horizon")
        positive_number(dt, "dt")
        positive_number(q_track, "q_track")
        positive_number(q_du, "q_du")
        self.num_agents = num_agents
        self.horizon = horizon
        self.dt = dt
        self.q_track = q_track
        self.q_du = q_du
        self.backend = backend or os.environ.get("LINEAR_NASH_MPC_BACKEND", "fast")
        if self.backend not in {"fast", "nashopt"}:
            raise ValueError(f"Unsupported LinearNashMPC backend '{self.backend}'")
        self.variational = bool(variational)
        self.nx_i = 6  # [px, py, pz, vx, vy, vz]
        self.nu_i = 3  # [ax, ay, az]
        self.ny_i = 3  # [px, py, pz]  — position output C_i = [I | 0]

        nx = num_agents * self.nx_i
        nu = num_agents * self.nu_i
        ny = num_agents * self.ny_i

        # --- 3D double-integrator matrices (exact zero-order hold) ---
        A_i = np.eye(6)
        A_i[:3, 3:] = np.eye(3) * dt        # velocity → position coupling
        # Velocity rows: A_i[3:, 3:] = I (no drag)

        B_i = np.zeros((6, 3))
        B_i[:3] = 0.5 * dt**2 * np.eye(3)   # pos term from acceleration
        B_i[3:] = dt * np.eye(3)             # vel term from acceleration

        C_i = np.zeros((3, 6))
        C_i[:, :3] = np.eye(3)              # extract only xyz position

        # Block-diagonal stacked system (joint state/input)
        A = np.kron(np.eye(num_agents), A_i)   # (N·nx_i, N·nx_i)
        B = np.kron(np.eye(num_agents), B_i)   # (N·nx_i, N·nu_i)
        C = np.kron(np.eye(num_agents), C_i)   # (N·ny_i, N·nx_i)

        # --- Cost matrices ---
        # Qy_i: (N·ny_i, N·ny_i) — agent i penalises only its own output y_i
        # (decoupled tracking; formation coupling would require off-diagonal blocks)
        Qy = []
        for i in range(num_agents):
            Qi = np.zeros((ny, ny))
            Qi[i*self.ny_i:(i+1)*self.ny_i,
               i*self.ny_i:(i+1)*self.ny_i] = np.eye(self.ny_i) * q_track
            Qy.append(Qi)

        # Qdu_i: (nu_i, nu_i) — input rate penalty for agent i (input-increment penalty)
        Qdu = [np.eye(self.nu_i) * q_du for _ in range(num_agents)]

        # NashLinearMPC: sizes = [nu_i]*N (input sizes, not horizon-stacked).
        # Built lazily because the lab path uses the faster precomputed solver.
        self._mpc = None
        self._nashopt_args = ([self.nu_i] * num_agents, A, B, C, Qy, Qdu, horizon)

        # Previous input u_{k-1} for Δu = u_k − u_{k-1} (stored input history)
        self._u_prev = np.zeros(nu)

        # Precompute single-step dynamics for output trajectory prediction
        self._A_i = A_i
        self._B_i = B_i
        self._C_i = C_i

        self._build_fast_solver()

    def reset_input_history(self) -> None:
        """Reset only the delta-input warm start without rebuilding the solver.

        Hardware runners use this after a ground-only NashLinearMPC warm-up so
        no synthetic command from initialization can affect the first flight
        control step.
        """
        self._u_prev.fill(0.0)

    def _build_fast_solver(self) -> None:
        """Precompute the unconstrained finite-horizon MPC linear solve."""
        h = self.horizon
        nx = self.nx_i
        nu = self.nu_i
        ny = self.ny_i
        A = self._A_i
        B = self._B_i
        C = self._C_i

        sx = np.zeros((h * nx, nx))
        su = np.zeros((h * nx, h * nu))
        powers = [np.eye(nx)]
        for _ in range(h):
            powers.append(powers[-1] @ A)

        for k in range(h):
            sx[k * nx:(k + 1) * nx, :] = powers[k + 1]
            for j in range(k + 1):
                block = powers[k - j] @ B
                su[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = block

        cy = np.kron(np.eye(h), C)
        yx = cy @ sx
        yu = cy @ su

        dmat = np.zeros((h * nu, h * nu))
        for k in range(h):
            rows = slice(k * nu, (k + 1) * nu)
            cols = slice(k * nu, (k + 1) * nu)
            dmat[rows, cols] = np.eye(nu)
            if k > 0:
                prev_cols = slice((k - 1) * nu, k * nu)
                dmat[rows, prev_cols] = -np.eye(nu)

        emat = np.zeros((h * nu, nu))
        emat[:nu, :] = np.eye(nu)

        qbar = self.q_track * np.eye(h * ny)
        rbar = self.q_du * np.eye(h * nu)
        hmat = yu.T @ qbar @ yu + dmat.T @ rbar @ dmat

        self._fast_yx = yx
        self._fast_yu_t_q = yu.T @ qbar
        self._fast_du_t_r_e = dmat.T @ rbar @ emat
        self._fast_h_inv = np.linalg.inv(hmat)

    def solve(self, current_states, target_positions):
        """
        Compute one MPC step.

        Parameters
        ----------
        current_states    : list of N (6,) arrays — current [pos, vel] states
        target_positions  : list of N (3,) or (6,) arrays — position targets

        Returns
        -------
        u_star      : ndarray (N, 1, nu_i) — first optimal input per agent
        x_stars     : list of N (1, nx_i) — one-step predicted states
        info        : dict with 'compute_time' and 'kkt_residual' (None for MILP)
        """
        current_states, target_positions = states_and_targets(
            current_states, target_positions, self.num_agents)
        t0 = time.time()
        if self.backend == "nashopt":
            u_first, solver_time = self._solve_nashopt(current_states, target_positions)
            backend = "nashopt"
        elif self.backend == "fast":
            u_first, solver_time = self._solve_fast(current_states, target_positions)
            backend = "fast"
        else:
            raise ValueError(f"Unsupported LinearNashMPC backend '{self.backend}'")
        compute_time = time.time() - t0

        # Update warm-start: pass current applied input as u_prev for next step
        self._u_prev = u_first.flatten()

        # One-step predicted states: x_{k+1} = A_i*x_k + B_i*u_k
        u_star = u_first.reshape((self.num_agents, 1, self.nu_i))
        x_stars = []
        for i in range(self.num_agents):
            x_i = np.asarray(current_states[i], dtype=float)
            x_next = self._A_i @ x_i + self._B_i @ u_first[i]
            x_stars.append(x_next.reshape((1, self.nx_i)))

        info = {"compute_time": compute_time,
                "kkt_residual": None,
                "solver_time": solver_time,
                "backend": backend}
        return u_star, x_stars, info

    def _solve_fast(self, current_states, target_positions):
        """Solve the separable unconstrained MPC problem with precomputed matrices."""
        h = self.horizon
        nu = self.nu_i
        ny = self.ny_i
        u_first = np.zeros((self.num_agents, nu))

        for i in range(self.num_agents):
            x_i = np.asarray(current_states[i], dtype=float)
            ref_i = np.asarray(target_positions[i], dtype=float)[:ny]
            u_prev_i = self._u_prev[i * nu:(i + 1) * nu]
            ref_stack = np.tile(ref_i, h)
            rhs = (
                self._fast_yu_t_q @ (ref_stack - self._fast_yx @ x_i)
                + self._fast_du_t_r_e @ u_prev_i
            )
            u_seq = self._fast_h_inv @ rhs
            u_first[i] = u_seq[:nu]

        return u_first, 0.0

    def _solve_nashopt(self, current_states, target_positions):
        """Compatibility path using NashLinearMPC/HiGHS."""
        if self._mpc is None:
            from nashopt import NashLinearMPC

            self._mpc = NashLinearMPC(*self._nashopt_args)

        x0 = np.concatenate([np.asarray(s, dtype=float) for s in current_states])
        ref = np.concatenate([np.asarray(t, dtype=float)[:3] for t in target_positions])

        if os.environ.get("LINEAR_NASH_MPC_SOLVER_OUTPUT", "").lower() in {"1", "true", "yes"}:
            sol = self._mpc.solve(
                x0, self._u_prev, ref,
                variational=self.variational, solver="highs",
            )
        else:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    sol = self._mpc.solve(
                        x0, self._u_prev, ref,
                        variational=self.variational, solver="highs",
                    )

        u_first = np.array(sol.u).reshape((self.num_agents, self.nu_i))
        return u_first, getattr(sol, "elapsed_time_solver", None)
