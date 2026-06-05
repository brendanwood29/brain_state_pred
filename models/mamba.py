import torch
import torch.nn as nn
from mamba_ssm import Mamba


class TrainableRegionEmbedding(nn.Module):
    def __init__(self, in_features: int, steps: int):
        super().__init__()
        self.pos_embed = nn.Embedding(in_features, 1)
        self.temp_embed = nn.Embedding(steps, 1)
        self.pos = torch.arange(0, in_features)
        self.temp = torch.arange(0, steps)

    def forward(self, x):
        device = x.device
        pos_embed = self.pos_embed(self.pos.to(device)).unsqueeze(0)
        temp_embed = self.temp_embed(self.temp.to(device)).unsqueeze(0)
        return x + pos_embed.permute(0, 2, 1).to(device) + temp_embed.to(device)


class MambaModel(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        steps: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        **kwargs,
    ) -> None:
        super().__init__()

        # self.proj_in = nn.Linear(d_model, d_model)
        # self.layer_norm_in = nn.LayerNorm([steps, d_model])
        # self.region_embed = TrainableRegionEmbedding(d_model, steps)
        self.blocks = nn.ModuleList(
            [Mamba(d_model, d_state, d_conv, expand) for _ in range(num_blocks)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm([steps, d_model]) for _ in range(num_blocks)]
        )
        self.drop = nn.Dropout(dropout)
        # self.layer_norm_out = nn.LayerNorm([steps, d_model])

    def forward(self, x):
        B, T, R = x.shape
        # x = self.layer_norm_in(x)
        # x = self.region_embed(x)
        for block, norm in zip(self.blocks, self.norms):
            x = x + self.drop(block(x))
            x = norm(x)
        # x = self.layer_norm_out(x)

        return x[:, [-1], :]
