import torch
import torch.nn as nn
from torchmetrics.functional import multiscale_structural_similarity_index_measure as ms_ssim


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, output, target):
        if output.shape != target.shape:
            target = torch.permute(target, (0, 3, 1, 2))
        return torch.mean(torch.square(output - target))


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, target):
        if output.shape != target.shape:
            target = torch.permute(target, (0, 3, 1, 2))
        return torch.mean(torch.abs(output - target))


class MsSSIMLoss(nn.Module):
    def __init__(
            self,
            data_range=1.0,
            win_size=11,
            win_sigma: float = 1.5,
            k1=0.01,
            k2=0.03,
    ):
        super(MsSSIMLoss, self).__init__()
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2

    def forward(self, output, target):
        # assuming batch of shape (n_samples, n_channels, height, width)
        if output.shape != target.shape:
            target = torch.permute(target, (0, 3, 1, 2))
        ms_ssim_loss = 1. - ms_ssim(
            output,
            target,
            kernel_size=self.win_size,
            sigma=self.win_sigma,
            data_range=self.data_range,
            k1=self.k1,
            k2=self.k2,  # try increasing this if NaN appears
        )

        return ms_ssim_loss


class MsSSIML1Loss(nn.Module):
    def __init__(
            self,
            alpha=0.8,
            data_range=1.0,
            win_size=11,
            win_sigma=1.5,
            k1=0.01,
            k2=0.03,
    ):
        super(MsSSIML1Loss, self).__init__()
        self.alpha = alpha
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2

    def forward(self, output, target):
        # assuming batch of shape (n_samples, n_channels, height, width)
        if output.shape != target.shape:
            target = torch.permute(target, (0, 3, 1, 2))
        ms_ssim_loss = 1. - ms_ssim(
            output,
            target,
            kernel_size=self.win_size,
            sigma=self.win_sigma,
            data_range=self.data_range,
            k1=self.k1,
            k2=self.k2,  # try increasing this if NaN appears
        )

        l1_loss = torch.mean(torch.abs(output - target))

        return self.alpha * ms_ssim_loss + (1. - self.alpha) * l1_loss
