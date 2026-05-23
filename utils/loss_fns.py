from typing import List

import torch
import torch.nn as nn


class RealImagMSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):

        y = y.permute(1, 0, 2, 3)

        y_hat_real, y_hat_imag = y_hat[0], y_hat[1]
        y_real, y_imag = y[0], y[1]
        return (0.5 * self.loss(y_hat_real, y_real)) + (
            0.5 * self.loss(y_hat_imag, y_imag)
        )


class MSEFCLoss(nn.Module):
    def __init__(self, mse_weight: float, fc_weight: float):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.fc_weight = fc_weight

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):

        model_fc = self.corcoef(y_hat.mT, y_hat.mT)
        real_fc = self.corcoef(y.mT, y.mT)

        return (self.mse_weight * self.mse(y_hat, y)) + (
            self.fc_weight * self.mse(model_fc, real_fc)
        )

    def corcoef(self, x: torch.Tensor, y: torch.Tensor):
        x_bar = x.mean(dim=-1)
        y_bar = y.mean(dim=-1)

        x_x_bar = x - x_bar.unsqueeze(-1)
        y_y_bar = y - y_bar.unsqueeze(-1)

        num = x_x_bar @ y_y_bar.mT
        den = (
            (x_x_bar**2).sum(dim=-1, keepdim=True)
            @ (y_y_bar**2).sum(dim=-1, keepdim=True).mT
        ) ** 0.5
        return num / den


class ReconFourierLoss(nn.Module):
    def __init__(
        self, recon_weight: float, fourier_weight: float, freq_weights: List[float]
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.fft_mse = nn.MSELoss(reduction="none")
        self.recon_weight = recon_weight
        self.fourier_weight = fourier_weight
        self.freq_weights = torch.tensor(freq_weights)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):

        self.freq_weights = self.freq_weights.to(y_hat.device)
        y_fft = torch.fft.rfft(y, dim=1)
        y_hat_fft = torch.fft.rfft(y_hat, dim=1)
        B, C, N = y_fft.real.shape
        real_mse = self.fft_mse(y_hat_fft.real, y_fft.real).reshape(B * N, C)
        imag_mse = self.fft_mse(y_hat_fft.imag, y_fft.imag).reshape(B * N, C)
        fft_loss = (self.freq_weights * real_mse.mean(dim=0)).mean() + (
            self.freq_weights * imag_mse.mean(dim=0)
        ).mean()
        return (self.recon_weight * self.mse(y_hat, y)) + (
            self.fourier_weight * fft_loss
        )


class MSE1DLoss(nn.Module):
    def __init__(
        self,
        recon_weight: float,
        first_weight: float,
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.recon_weight = recon_weight
        self.first_weight = first_weight

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):

        y_prime = torch.diff(y, dim=1)
        y_hat_prime = torch.diff(y_hat, dim=1)

        return (self.recon_weight * self.mse(y_hat, y)) + (
            self.first_weight * self.mse(y_hat_prime, y_prime)
        )
