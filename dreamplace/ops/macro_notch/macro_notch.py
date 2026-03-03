##
# @file   macro_notch.py
# @brief  Differentiable penalty for narrow (notch) gaps between adjacent macros.
#
# Notches — concave gaps narrower than notch_thresh — concentrate routing
# demand, raise EM risk, and disrupt power-grid continuity.
#
# Edge-to-edge gap between macros i and j:
#   dx_gap = max(0, |cx_i - cx_j| - (w_i + w_j)/2)
#   dy_gap = max(0, |cy_i - cy_j| - (h_i + h_j)/2)
#   d_ij   = dx_gap + dy_gap
#
# Loss = Σ_{i<j}  ReLU(notch_thresh - d_ij)²
#
# Complexity: O(N_macro²) — dense but fast for ISPD benchmarks (<200 macros).
# Set beyond_ppa_prune_notch=1 in params to skip pairs beyond 2×notch_thresh
# (spatial pre-filter, activated once correctness is validated).
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class MacroNotchOp(nn.Module):
    """
    Parameters
    ----------
    macro_mask : BoolTensor [num_movable_nodes]
    macro_size_x, macro_size_y : 1-D float tensors [N_macro]
        Width / height of each macro node (constant).
    notch_thresh : float
        Minimum allowed edge-to-edge gap in layout units.
    num_movable_nodes, num_physical_nodes : int
    prune : bool
        If True, skip pairs whose centre distance > 2 × notch_thresh (cheap
        pre-filter).  Default False until validated.
    """

    def __init__(self, macro_mask, macro_size_x, macro_size_y,
                 notch_thresh, num_movable_nodes, num_physical_nodes,
                 prune=False):
        super().__init__()
        self.register_buffer('macro_mask',   macro_mask)
        self.register_buffer('macro_size_x', macro_size_x.float())  # [N_macro]
        self.register_buffer('macro_size_y', macro_size_y.float())
        self.notch_thresh = float(notch_thresh)
        self.num_movable  = int(num_movable_nodes)
        self.num_phys     = int(num_physical_nodes)
        self.prune        = bool(prune)

    def forward(self, pos):
        """
        Parameters
        ----------
        pos : 1-D tensor [2 * num_physical_nodes]

        Returns
        -------
        Scalar raw cost.
        """
        x = pos[:self.num_movable][self.macro_mask]
        y = pos[self.num_phys: self.num_phys + self.num_movable][self.macro_mask]
        N = x.shape[0]

        if N < 2:
            return pos.new_zeros(())

        sx = self.macro_size_x.to(dtype=x.dtype, device=x.device)
        sy = self.macro_size_y.to(dtype=x.dtype, device=x.device)

        # Pairwise differences [N, N]
        cx_diff = (x.unsqueeze(1) - x.unsqueeze(0)).abs()
        cy_diff = (y.unsqueeze(1) - y.unsqueeze(0)).abs()

        # Half-sum of widths / heights
        half_w = (sx.unsqueeze(1) + sx.unsqueeze(0)) * 0.5
        half_h = (sy.unsqueeze(1) + sy.unsqueeze(0)) * 0.5

        # Edge-to-edge gap (0 means overlapping or touching)
        dx_gap = F.relu(cx_diff - half_w)
        dy_gap = F.relu(cy_diff - half_h)
        d_ij   = dx_gap + dy_gap                               # [N, N]

        # Optional cheap pre-filter: ignore pairs far apart
        if self.prune:
            centre_dist = (cx_diff.pow(2) + cy_diff.pow(2)).sqrt()
            far_mask    = centre_dist > 2.0 * self.notch_thresh
            d_ij        = d_ij + far_mask.float() * self.notch_thresh * 10

        # Upper-triangle only (exclude self-pairs and double-counting)
        ut_mask  = torch.ones(N, N, dtype=torch.bool, device=x.device).triu(1)
        penalty  = F.relu(self.notch_thresh - d_ij[ut_mask])
        return penalty.pow(2).sum()
