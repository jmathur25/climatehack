import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM


class MS_SSIMLoss(nn.Module):
    """Multi-Scale SSIM Loss"""

    def __init__(self, channels=1, **kwargs):
        """
        Initialize
        Args:
            convert_range: Convert input from -1,1 to 0,1 range
            **kwargs: Kwargs to pass through to MS_SSIM
        """
        super(MS_SSIMLoss, self).__init__()
        self.ssim_module = MS_SSIM(
            data_range=1023.0, size_average=True, win_size=3, channel=channels, **kwargs
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward method
        Args:
            x: tensor one
            y: tensor two
        Returns: multi-scale SSIM Loss
        """
        return 1.0 - self.ssim_module(x, y)
