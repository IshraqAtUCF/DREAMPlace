##
# @file   macro_density.py
# @brief  Differentiable macro density uniformity penalty.
#
# Computes the variance of macro area density across a coarse grid.
# Minimising variance discourages macro clustering, reducing localised IR-drop
# and thermal hotspots.
#
# Uses bilinear area scatter — fully differentiable through macro positions.
#

import torch
import torch.nn as nn


class MacroDensityUniformityOp(nn.Module):
    """
    Variance of macro density over a coarse num_bins × num_bins grid.

    Parameters
    ----------
    macro_mask : BoolTensor [num_movable_nodes]
    macro_size_x, macro_size_y : 1-D float tensors [N_macro]
    num_movable_nodes, num_physical_nodes : int
    num_bins_x, num_bins_y : int
        Resolution of the density grid (32 × 32 is usually sufficient).
    xl, xh, yl, yh : float
        Layout bounding box.
    """

    def __init__(self, macro_mask, macro_size_x, macro_size_y,
                 num_movable_nodes, num_physical_nodes,
                 num_bins_x, num_bins_y, xl, xh, yl, yh):
        super().__init__()
        self.register_buffer('macro_mask',   macro_mask)
        self.register_buffer('macro_size_x', macro_size_x.float())
        self.register_buffer('macro_size_y', macro_size_y.float())
        self.num_movable = int(num_movable_nodes)
        self.num_phys    = int(num_physical_nodes)
        self.num_bins_x  = int(num_bins_x)
        self.num_bins_y  = int(num_bins_y)
        self.xl = float(xl);  self.xh = float(xh)
        self.yl = float(yl);  self.yh = float(yh)
        self.bin_w = (xh - xl) / num_bins_x
        self.bin_h = (yh - yl) / num_bins_y

    def forward(self, pos):
        """
        Parameters
        ----------
        pos : 1-D tensor [2 * num_physical_nodes]

        Returns
        -------
        Scalar raw cost (density variance, ~[0, 1] when normalised by bin count).
        """
        x = pos[:self.num_movable][self.macro_mask]
        y = pos[self.num_phys: self.num_phys + self.num_movable][self.macro_mask]
        N = x.shape[0]

        if N == 0:
            return pos.new_zeros(())

        sx = self.macro_size_x.to(dtype=x.dtype, device=x.device)
        sy = self.macro_size_y.to(dtype=x.dtype, device=x.device)

        # Fractional bin coordinates
        fx = ((x - self.xl) / self.bin_w).clamp(0.0, self.num_bins_x - 1 - 1e-6)
        fy = ((y - self.yl) / self.bin_h).clamp(0.0, self.num_bins_y - 1 - 1e-6)

        ix = fx.long()
        iy = fy.long()
        dx = fx - ix.float()
        dy = fy - iy.float()

        # Per-macro area weight (normalised to bin area)
        area = (sx / self.bin_w).clamp(max=1.0) * (sy / self.bin_h).clamp(max=1.0)

        # Bilinear weights
        w00 = (1 - dx) * (1 - dy) * area
        w10 = dx       * (1 - dy) * area
        w01 = (1 - dx) * dy       * area
        w11 = dx       * dy       * area

        BX = self.num_bins_x
        BY = self.num_bins_y
        density = torch.zeros(BX * BY, dtype=x.dtype, device=x.device)

        ix1 = (ix + 1).clamp(max=BX - 1)
        iy1 = (iy + 1).clamp(max=BY - 1)

        density.scatter_add_(0, (iy  * BX + ix ).long(), w00)
        density.scatter_add_(0, (iy  * BX + ix1).long(), w10)
        density.scatter_add_(0, (iy1 * BX + ix ).long(), w01)
        density.scatter_add_(0, (iy1 * BX + ix1).long(), w11)

        return density.var()
