##
# @file   io_keepout.py
# @brief  Differentiable penalty for macros that encroach on IO keepout zones.
#
# Returns a raw (unweighted) scalar cost. Weight is applied by BeyondPPAObj so
# that the MPC controller sees unscaled magnitudes.
#
# Loss = Σ_{macro i, IO port j}  ReLU(keepout_dist - ||center_i - pos_j||)²
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOKeepoutOp(nn.Module):
    """
    Penalise macro nodes that lie within keepout_dist of any fixed IO terminal.

    Parameters
    ----------
    macro_mask : BoolTensor [num_movable_nodes]
        True for nodes classified as macros.
    io_x, io_y : 1-D float tensors
        X/Y coordinates of fixed IO terminals (constant throughout placement).
    keepout_dist : float
        Minimum allowed Euclidean distance (in layout units) from macro centre
        to any IO port.
    num_movable_nodes : int
    num_physical_nodes : int
        Used to index into the flat pos tensor  [x_movable | x_fixed | ... |
        y_movable | y_fixed | ...].
    """

    def __init__(self, macro_mask, io_x, io_y, keepout_dist,
                 num_movable_nodes, num_physical_nodes):
        super().__init__()
        self.register_buffer('macro_mask', macro_mask)          # bool [N_mov]
        # Store IO positions as a constant [N_io, 2] buffer
        io_pos = torch.stack([io_x.float(), io_y.float()], dim=1)
        self.register_buffer('io_pos', io_pos)                  # [N_io, 2]
        self.keepout_dist = float(keepout_dist)
        self.num_movable  = int(num_movable_nodes)
        self.num_phys     = int(num_physical_nodes)

    def forward(self, pos):
        """
        Parameters
        ----------
        pos : 1-D tensor, shape [2 * num_physical_nodes]
            Flat layout [x0..xN, y0..yN].

        Returns
        -------
        Scalar raw cost (no weight applied).
        """
        x = pos[:self.num_movable][self.macro_mask]
        y = pos[self.num_phys: self.num_phys + self.num_movable][self.macro_mask]

        if x.shape[0] == 0 or self.io_pos.shape[0] == 0:
            return pos.new_zeros(())

        macro_centers = torch.stack([x, y], dim=1)              # [N_macro, 2]
        io_pos = self.io_pos.to(dtype=macro_centers.dtype,
                                device=macro_centers.device)
        dists = torch.cdist(macro_centers, io_pos)              # [N_macro, N_io]
        viol  = F.relu(self.keepout_dist - dists)
        return viol.pow(2).sum()
