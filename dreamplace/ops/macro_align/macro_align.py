##
# @file   macro_align.py
# @brief  Differentiable penalty for macro deviation from a virtual placement grid.
#
# Uses a periodic sin² potential — energy is exactly 0 at grid multiples and
# rises smoothly between them.  Gradients guide macros toward grid anchors,
# improving clock-tree synthesis regularity and reducing clock skew.
#
# Loss = Σ_i [ sin²(π·x_i / pitch_x) + sin²(π·y_i / pitch_y) ]
#

import math
import torch
import torch.nn as nn


class MacroGridAlignOp(nn.Module):
    """
    Periodic alignment potential for macro nodes.

    Parameters
    ----------
    macro_mask : BoolTensor [num_movable_nodes]
    pitch_x, pitch_y : float
        Grid pitch in layout units.  Defaults suggest 8 × row_height.
    num_movable_nodes, num_physical_nodes : int
    """

    def __init__(self, macro_mask, pitch_x, pitch_y,
                 num_movable_nodes, num_physical_nodes):
        super().__init__()
        self.register_buffer('macro_mask', macro_mask)
        self.pitch_x     = float(pitch_x)
        self.pitch_y     = float(pitch_y)
        self.num_movable = int(num_movable_nodes)
        self.num_phys    = int(num_physical_nodes)

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

        if x.shape[0] == 0:
            return pos.new_zeros(())

        loss_x = torch.sin(math.pi * x / self.pitch_x).pow(2)
        loss_y = torch.sin(math.pi * y / self.pitch_y).pow(2)
        return (loss_x + loss_y).sum()
