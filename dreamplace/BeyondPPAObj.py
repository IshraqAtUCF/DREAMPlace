##
# @file   BeyondPPAObj.py
# @brief  Human-inspired reliability cost terms for macro placement.
#
# Implements four of the five BeyondPPA features from:
#   "BeyondPPA: Human-Inspired RL for Post-Route Reliability-Aware
#    Macro Placement" (MLCAD 2025).
# Wirelength (feature 1) is already handled by DREAMPlace's PlaceObj.
#
# Features
# --------
#   density  – macro density uniformity  →  IR-drop mitigation
#   io       – IO keepout zone penalty   →  noise coupling reduction
#   align    – macro grid alignment      →  clock-tree quality
#   notch    – notch avoidance           →  EM / congestion resilience
#
# Design decisions aligned with the implementation review:
#   (A) Single instantiation: owned by PlaceObj, not duplicated in BasicPlace.
#   (B) Log-space weights: MPC passes log-space deltas; weights = exp(log_w).
#   (C) Overflow gating: features are inactive until overflow < gate_overflow,
#       then ramped linearly over ramp_iters gradient steps.
#   (F) Raw metrics: forward() returns (weighted_total, raw_breakdown_dict) so
#       the MPC controller can observe unscaled feature magnitudes.
#

import math
import logging

import torch
import torch.nn as nn

from dreamplace.ops.io_keepout.io_keepout    import IOKeepoutOp
from dreamplace.ops.macro_align.macro_align  import MacroGridAlignOp
from dreamplace.ops.macro_notch.macro_notch  import MacroNotchOp
from dreamplace.ops.macro_density.macro_density import MacroDensityUniformityOp


class BeyondPPAObj(nn.Module):
    """
    Combined BeyondPPA objective for macro reliability.

    Instantiate once in PlaceObj.__init__ when params.beyond_ppa_flag is set.
    """

    # Canonical weight order (matches MPC control vector indices 1..4)
    FEAT_NAMES = ('density', 'io', 'align', 'notch')

    def __init__(self, params, placedb, data_collections):
        super().__init__()

        # ── Gate / ramp state ────────────────────────────────────────────────
        self.enabled       = False
        self.gate_overflow = float(getattr(params, 'beyond_ppa_gate_overflow', 0.3))
        self.ramp_iters    = int(getattr(params, 'beyond_ppa_ramp_iters', 50))
        self._ramp_step    = 0

        # ── Macro identification ─────────────────────────────────────────────
        num_movable = placedb.num_movable_nodes
        num_phys    = placedb.num_physical_nodes
        macro_mask_np = placedb.movable_macro_mask        # bool ndarray [N_mov]
        macro_mask    = torch.BoolTensor(macro_mask_np)

        macro_size_x = torch.tensor(
            placedb.node_size_x[:num_movable][macro_mask_np], dtype=torch.float32)
        macro_size_y = torch.tensor(
            placedb.node_size_y[:num_movable][macro_mask_np], dtype=torch.float32)

        num_macros = int(macro_mask.sum().item())
        logging.info("BeyondPPAObj: %d macro nodes identified out of %d movable"
                     % (num_macros, num_movable))

        # ── IO terminal positions (fixed, stored as buffers) ─────────────────
        num_terminals = placedb.num_terminals
        io_x = torch.tensor(
            placedb.node_x[num_movable: num_movable + num_terminals],
            dtype=torch.float32)
        io_y = torch.tensor(
            placedb.node_y[num_movable: num_movable + num_terminals],
            dtype=torch.float32)

        # ── Geometry parameters ──────────────────────────────────────────────
        row_h        = float(placedb.row_height)
        keepout_dist = float(getattr(params, 'io_keepout_distance',    10)) * row_h
        notch_thresh = float(getattr(params, 'notch_threshold',         5)) * row_h
        pitch_x      = float(getattr(params, 'grid_alignment_pitch_x',  8)) * row_h
        pitch_y      = float(getattr(params, 'grid_alignment_pitch_y',  8)) * row_h
        macro_bins   = int(getattr(params,   'beyond_ppa_macro_bins',  32))
        prune_notch  = bool(getattr(params,  'beyond_ppa_prune_notch', False))

        # ── Build ops ────────────────────────────────────────────────────────
        self.density_op = MacroDensityUniformityOp(
            macro_mask, macro_size_x, macro_size_y,
            num_movable, num_phys,
            macro_bins, macro_bins,
            placedb.xl, placedb.xh, placedb.yl, placedb.yh)

        self.io_op = IOKeepoutOp(
            macro_mask, io_x, io_y, keepout_dist,
            num_movable, num_phys)

        self.align_op = MacroGridAlignOp(
            macro_mask, pitch_x, pitch_y,
            num_movable, num_phys)

        self.notch_op = MacroNotchOp(
            macro_mask, macro_size_x, macro_size_y,
            notch_thresh, num_movable, num_phys,
            prune=prune_notch)

        # ── Log-space weights (design B: increments in log-space) ────────────
        # Order: [density, io, align, notch]
        init_w = list(getattr(params, 'beyond_ppa_weights', [1.0, 1.0, 1.0, 1.0]))
        # Pad / truncate to exactly 4 values
        while len(init_w) < 4:
            init_w.append(1.0)
        init_w = init_w[:4]
        self._log_w = [math.log(max(w, 1e-8)) for w in init_w]

        logging.info(
            "BeyondPPAObj: initial weights density=%.3f io=%.3f align=%.3f notch=%.3f"
            % (math.exp(self._log_w[0]), math.exp(self._log_w[1]),
               math.exp(self._log_w[2]), math.exp(self._log_w[3])))

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def weights(self):
        """Current feature weights as a plain list [density, io, align, notch]."""
        return [math.exp(lw) for lw in self._log_w]

    def update_weights_log(self, delta_log):
        """
        Apply MPC log-space increments.

        Parameters
        ----------
        delta_log : sequence of 4 floats
            Log-space increments [Δlog_density, Δlog_io, Δlog_align, Δlog_notch].
            Each increment is clipped to ±MAX_DELTA and blended via EMA.
        """
        MAX_DELTA = 0.095   # ≈ ±10 % change per call in natural-log scale
        EMA_ALPHA = 0.20    # blend factor (80 % history, 20 % new signal)
        for i, d in enumerate(delta_log):
            d_clipped = max(-MAX_DELTA, min(MAX_DELTA, float(d)))
            self._log_w[i] += EMA_ALPHA * d_clipped

    def check_and_enable(self, overflow):
        """
        Activate BeyondPPA costs once overflow drops below the gate threshold.

        Parameters
        ----------
        overflow : float  (scalar overflow value, 0..1)
        """
        if self.enabled:
            return
        if float(overflow) < self.gate_overflow:
            self.enabled = True
            self._ramp_step = 0
            logging.info(
                "BeyondPPAObj: activated (overflow=%.4f < gate=%.4f)"
                % (float(overflow), self.gate_overflow))

    def _ramp_scale(self):
        """Linear ramp 0 → 1 over ramp_iters calls after activation."""
        if self._ramp_step >= self.ramp_iters:
            return 1.0
        scale = self._ramp_step / max(self.ramp_iters, 1)
        self._ramp_step += 1
        return scale

    def forward(self, pos):
        """
        Compute BeyondPPA objective.

        Parameters
        ----------
        pos : 1-D tensor [2 * num_physical_nodes]

        Returns
        -------
        total : scalar tensor
            Weighted and ramped sum of all feature costs (used in back-prop).
        raw : dict  {str → scalar tensor (detached)}
            Unweighted feature costs for MPC state observation and logging.
        """
        # Always compute raw costs (cheap for small N_macro) so MPC can track them
        with torch.set_grad_enabled(self.enabled):
            raw = {
                'density': self.density_op(pos),
                'io':      self.io_op(pos),
                'align':   self.align_op(pos),
                'notch':   self.notch_op(pos),
            }

        if not self.enabled:
            # Return zero gradient contribution; raw still logged
            zero = pos.new_zeros(())
            return zero, {k: v.detach() for k, v in raw.items()}

        ramp = self._ramp_scale()
        w    = self.weights          # [density, io, align, notch]

        total = ramp * (
            w[0] * raw['density'] +
            w[1] * raw['io']      +
            w[2] * raw['align']   +
            w[3] * raw['notch']
        )
        return total, {k: v.detach() for k, v in raw.items()}
