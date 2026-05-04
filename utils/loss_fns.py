import torch.nn as nn


class RealImagMSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, y_hat, y):

        y = y.permute(1, 0, 2, 3)

        y_hat_real, y_hat_imag = y_hat[0], y_hat[1]
        y_real, y_imag = y[0], y[1]
        return (0.5 * self.loss(y_hat_real, y_real)) + (
            0.5 * self.loss(y_hat_imag, y_imag)
        )
