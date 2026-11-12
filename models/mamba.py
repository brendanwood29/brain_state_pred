import torch.nn as nn
from mamba_ssm import Mamba

from .layers import LoRA, MambaLayer


class OldMamba(nn.Module):
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
                    MambaLayer(
                        steps=steps,
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        dropout=dropout,
                    ),
                    LoRA(d_model, lora_r, d_model, lora_alpha, lora_bias, nn.GELU),
                )
                for _ in range(num_blocks - 1)
            ]
            # Last layer of model should not have activation
            # since regression task so pass identitiy
            mods.append(
                nn.Sequential(
                    MambaLayer(
                        steps=steps,
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        dropout=dropout,
                    ),
                    LoRA(d_model, lora_r, d_model, lora_alpha, lora_bias, nn.Identity),
                )
            )
            self.blocks = nn.ModuleList(mods)
        else:
            self.blocks = nn.ModuleList(
                [
                    MambaLayer(
                        steps=steps,
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x[:, [-1], :]
