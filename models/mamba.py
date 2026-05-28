import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torchinfo import summary


class MambaModel(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        **kwargs,
    ) -> None:
        super().__init__()

        # self.proj_in = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [Mamba(d_model, d_state, d_conv, expand) for _ in range(num_blocks)]
        )
        # self.proj_out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm((kwargs["steps"], d_model))

    def forward(self, x):

        # x_in = self.proj_in(x)
        # for block in self.blocks:
        #     x_in = self.drop(block(x_in))
        # x_out = x + self.proj_out(x_in)
        for block in self.blocks:
            x = x + self.drop(block(x))
        x = self.layer_norm(x)
        return x[:, [-1], :]


if __name__ == "__main__":
    model = MambaModel()
    summary(model)
    model = model.to("cuda")
    x = torch.rand((82, 20, 1000)).to("cuda")
    y = model(x)
    print(y.shape)
