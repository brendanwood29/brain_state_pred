import torch
import torch.nn as nn
from mamba_ssm import Mamba

from .transformer_based import LoRA


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
        use_lora: bool = False,
        lora_r: int = 64,
        lora_alpha: int = 64,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if use_lora:
            mods = [
                nn.Sequential(
                    Mamba(d_model, d_state, d_conv, expand),
                    LoRA(d_model, lora_r, d_model, lora_alpha, lora_bias, nn.SiLU),
                )
                for _ in range(num_blocks - 1)
            ]
            # Last layer of model should not have activation
            # since regression task so pass identitiy
            mods.append(
                nn.Sequential(
                    Mamba(d_model, d_state, d_conv, expand),
                    LoRA(d_model, lora_r, d_model, lora_alpha, lora_bias, nn.Identity),
                )
            )
            self.blocks = nn.ModuleList(mods)
        else:
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        Mamba(d_model, d_state, d_conv, expand),
                    )
                    for _ in range(num_blocks)
                ]
            )
        self.norms = nn.ModuleList(
            [nn.LayerNorm([steps, d_model]) for _ in range(num_blocks)]
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        for block, norm in zip(self.blocks, self.norms):
            x = x + self.drop(block(x))
            x = norm(x)

        return x[:, [-1], :]
