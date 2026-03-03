##
# @file   MPCController.py
# @brief  Model Predictive Control for adaptive BeyondPPA weight scheduling.
#
# Design decisions aligned with the implementation review:
#   (B) Control variables are log-space increments: u_t = Δlog(λ).
#       Actual weights are updated as: log(λ_{t+1}) = log(λ_t) + u_t.
#       Bounds: u_t ∈ [−max_delta, +max_delta] per feature per interval.
#   (C) Density-weight anchoring: MPC adjusts a bounded log-offset from
#       density_weight_base, not multiplicative compounding.
#   (D) Ridge regression (not plain least squares) for dynamics fitting.
#   (F) MPC state uses raw (unweighted) feature metrics.
#
# State vector s ∈ R^6:
#   [overflow, delta_hpwl_norm, density_raw, io_raw, align_raw, notch_raw]
#
# Control vector u ∈ R^5  (all log-space increments):
#   [dw_log,  Δlog_density, Δlog_io, Δlog_align, Δlog_notch]
#
# Dynamics:  s_{t+1} ≈ A·s_t + B·u_t + c   (online ridge fit)
#
# MPC cost:  Σ_{k=0}^{H-1} (s_k − s_ref)ᵀQ(s_k − s_ref) + u_kᵀ R u_k
#
# Activation: before the model is fitted (< min_history samples) or excitation
# is too low, the controller returns zeros (= no change).  This implements a
# safe heuristic fall-back (review point: add rule-based fallback layer later).
#

import math
import logging

import numpy as np


class MPCController:
    """
    Receding-horizon controller that adapts BeyondPPA feature weights during
    the DREAMPlace global-placement optimisation loop.

    Parameters
    ----------
    params : Params
        DREAMPlace parameter object.  Relevant attributes (all optional with
        sensible defaults):
            mpc_horizon        (int,   default 5)
            mpc_interval       (int,   default 50)
            mpc_history_size   (int,   default 200)
            mpc_ridge          (float, default 1e-3)
            mpc_q_weights      (list[float], 6 values)
            mpc_r_weights      (list[float], 5 values)
            stop_overflow      (float, default 0.07)
    """

    STATE_DIM   = 6   # [overflow, Δhpwl_norm, density_raw, io_raw, align_raw, notch_raw]
    CONTROL_DIM = 5   # [dw_log, Δlog_density, Δlog_io, Δlog_align, Δlog_notch]

    # Per-interval cap on any log-space control action (≈ ±10 %)
    MAX_DELTA = 0.095

    # Minimum number of recorded transitions before fitting
    MIN_HISTORY = 15

    # Minimum average control variation to consider model well-excited
    MIN_EXCITATION = 1e-6

    def __init__(self, params):
        self.H        = int(getattr(params, 'mpc_horizon',      5))
        self.interval = int(getattr(params, 'mpc_interval',    50))
        self.hist_sz  = int(getattr(params, 'mpc_history_size',200))
        self.ridge    = float(getattr(params, 'mpc_ridge',     1e-3))

        # History buffer: list of (s_t, u_t, s_{t+1})
        self._history = []

        # Linear dynamics model parameters (reset to neutral at init)
        self.A = np.eye(self.STATE_DIM,   dtype=np.float64)
        self.B = np.zeros((self.STATE_DIM, self.CONTROL_DIM), dtype=np.float64)
        self.c = np.zeros(self.STATE_DIM, dtype=np.float64)
        self._model_fitted    = False
        self._last_excitation = 0.0

        # Anchored density-weight log-offset (cumulative; bounded)
        self._dw_log_offset     = 0.0
        self._dw_log_offset_max = math.log(10.0)   # max 10× baseline

        # Reference state: target overflow, everything else → 0
        stop_ov     = float(getattr(params, 'stop_overflow', 0.07))
        self.s_ref  = np.array([stop_ov, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Cost matrices
        q_raw = list(getattr(params, 'mpc_q_weights',
                             [10.0, 1.0, 2.0, 3.0, 3.0, 3.0]))
        r_raw = list(getattr(params, 'mpc_r_weights',
                             [0.1, 0.1, 0.1, 0.1, 0.1]))
        while len(q_raw) < self.STATE_DIM:   q_raw.append(1.0)
        while len(r_raw) < self.CONTROL_DIM: r_raw.append(0.1)
        self.Q = np.diag(q_raw[:self.STATE_DIM])
        self.R = np.diag(r_raw[:self.CONTROL_DIM])

        logging.info(
            "MPCController: H=%d interval=%d hist=%d ridge=%.2e"
            % (self.H, self.interval, self.hist_sz, self.ridge))

    # ── History ───────────────────────────────────────────────────────────────

    def record(self, s_t, u_t, s_next):
        """
        Record one (state, control, next_state) transition.

        Parameters
        ----------
        s_t, s_next : array-like, length STATE_DIM
        u_t         : array-like, length CONTROL_DIM
        """
        self._history.append((
            np.asarray(s_t,    dtype=np.float64),
            np.asarray(u_t,    dtype=np.float64),
            np.asarray(s_next, dtype=np.float64),
        ))
        if len(self._history) > self.hist_sz:
            self._history.pop(0)

    # ── Model fitting ─────────────────────────────────────────────────────────

    def fit(self):
        """
        Fit linear dynamics from recorded history using ridge regression.

        Only updates the model when there is sufficient data and the recent
        control signals show non-trivial variation (excitation check).
        """
        n = len(self._history)
        if n < self.MIN_HISTORY:
            return

        SC = np.array([[*s, *u] for s, u, _ in self._history])  # [n, S+C]
        SN = np.array([ns      for _, _, ns in self._history])   # [n, S]

        # Augment with bias column, then ridge-regularise
        X   = np.hstack([SC, np.ones((n, 1))])                  # [n, S+C+1]
        XtX = X.T @ X
        XtX += self.ridge * np.eye(XtX.shape[0])

        try:
            theta = np.linalg.solve(XtX, X.T @ SN)             # [S+C+1, S]
        except np.linalg.LinAlgError:
            logging.warning("MPCController: linear solve failed, keeping old model")
            return

        d  = self.STATE_DIM
        cd = self.CONTROL_DIM
        self.A = theta[:d,      :].T   # [S, S]
        self.B = theta[d:d+cd,  :].T   # [S, C]
        self.c = theta[-1,       :]    # [S]
        self._model_fitted = True

        # Measure excitation: mean std of control signals in recent history
        U = np.array([u for _, u, _ in self._history])
        self._last_excitation = float(U.std(axis=0).mean())

    # ── State normalisation ───────────────────────────────────────────────────

    @staticmethod
    def _normalize(s):
        """
        Soft normalisation to bring each component toward O(1).
        Uses fixed design-time scale estimates; no pretraining required.
        """
        scales = np.array([
            1.0,    # overflow      (already 0 – 1)
            0.05,   # Δhpwl_norm   (typical range ±0.05)
            1.0,    # density_raw  (variance ~0 – 1)
            1e4,    # io_raw        (sum of ReLU², design-dependent)
            1e2,    # align_raw
            1e4,    # notch_raw
        ], dtype=np.float64)
        return s / (scales + 1e-12)

    # ── MPC step ──────────────────────────────────────────────────────────────

    def step(self, current_state, current_u):
        """
        Compute optimal control increment for the next interval.

        Parameters
        ----------
        current_state : array-like, length STATE_DIM
        current_u     : array-like, length CONTROL_DIM
            Log-space control vector from the previous interval.

        Returns
        -------
        u_opt : ndarray, length CONTROL_DIM
            Log-space increments to apply this interval.
            Returns zeros (no change) if the model is not yet ready.
        """
        if not self._model_fitted or self._last_excitation < self.MIN_EXCITATION:
            return np.zeros(self.CONTROL_DIM)

        try:
            import scipy.optimize
        except ImportError:
            logging.warning("MPCController: scipy not available, returning zeros")
            return np.zeros(self.CONTROL_DIM)

        s0       = self._normalize(np.asarray(current_state, dtype=np.float64))
        s_ref_n  = self._normalize(self.s_ref)
        u0       = np.asarray(current_u, dtype=np.float64)

        # Pre-compute normalised dynamics
        A, B, c = self.A, self.B, self.c

        def _cost(u_flat):
            U = u_flat.reshape(self.H, self.CONTROL_DIM)
            s = s0.copy()
            total = 0.0
            for k in range(self.H):
                s_next = A @ s + B @ U[k] + c
                ds = s_next - s_ref_n
                du = U[k]           # penalise deviation from zero (= no change)
                total += float(ds @ self.Q @ ds + du @ self.R @ du)
                s = s_next
            return total

        # Bounds: all log-space increments ∈ [−MAX_DELTA, +MAX_DELTA]
        bounds = [(-self.MAX_DELTA, self.MAX_DELTA)] * (self.H * self.CONTROL_DIM)
        u_init = np.zeros(self.H * self.CONTROL_DIM)

        try:
            res = scipy.optimize.minimize(
                _cost, u_init, method='SLSQP', bounds=bounds,
                options={'maxiter': 40, 'ftol': 1e-7})
            return np.clip(res.x[:self.CONTROL_DIM],
                           -self.MAX_DELTA, self.MAX_DELTA)
        except Exception as exc:
            logging.warning("MPCController.step: optimisation failed (%s)" % exc)
            return np.zeros(self.CONTROL_DIM)

    # ── Density-weight anchor management ─────────────────────────────────────

    def apply_density_weight_delta(self, density_weight_base, delta_log):
        """
        Compute new density_weight from base × exp(bounded cumulative offset).

        Parameters
        ----------
        density_weight_base : float
            The original density_weight recorded at the start of placement.
        delta_log : float
            Log-space increment from MPC (control[0]).

        Returns
        -------
        float  New density_weight value.
        """
        self._dw_log_offset = np.clip(
            self._dw_log_offset + delta_log,
            -self._dw_log_offset_max,
             self._dw_log_offset_max)
        return float(density_weight_base) * math.exp(self._dw_log_offset)
