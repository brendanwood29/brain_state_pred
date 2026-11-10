from typing import Callable

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding, from google TimesFM, only for Q and K matrices"""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: float = 1.0,
        max_timescale: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(
        self,
        inputs: torch.Tensor,
        position: torch.Tensor | None = None,
    ):
        """Generates a JTensor of sinusoids with different frequencies."""
        if self.embedding_dims != inputs.shape[-1]:
            raise ValueError(
                "The embedding dims of the rotary position embedding"
                "must match the hidden dimension of the inputs."
            )
        half_embedding_dim = self.embedding_dims // 2
        fraction = (
            2
            * torch.arange(0, half_embedding_dim, device=inputs.device)
            / self.embedding_dims
        )
        timescale = (
            self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        ).to(inputs.device)
        if position is None:
            seq_length = inputs.shape[1]
            position = torch.arange(
                seq_length, dtype=torch.float32, device=inputs.device
            )[None, :]

        if len(inputs.shape) == 4:
            position = position[..., None, None]
            timescale = timescale[None, None, None, :]
        elif len(inputs.shape) == 3:
            position = position[..., None]
            timescale = timescale[None, None, :]
        else:
            raise ValueError("Inputs must be of rank 3 or 4.")

        sinusoid_inp = position / timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        first_half, second_half = torch.chunk(inputs, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat([first_part, second_part], dim=-1)


class TrainableRegionEmbedding(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.pos_embed = nn.Embedding(in_features, 1)
        self.pos = torch.arange(0, in_features)

    def forward(self, x):
        device = x.device
        pos_embed = self.pos_embed(self.pos.to(device)).unsqueeze(0)
        return x + pos_embed.to(x.device)


class TrainableTemporalEmbedding(nn.Module):
    def __init__(
        self,
        steps: int,
    ):
        super().__init__()
        self.pos_embed = nn.Embedding(steps, 1)
        self.pos = torch.arange(0, steps)

    def forward(self, x):
        device = x.device
        pos_embed = self.pos_embed(self.pos.to(device)).unsqueeze(0)
        return x + pos_embed.to(device)


class LoRA(nn.Module):
    def __init__(
        self,
        in_dim,
        r,
        out_dim,
        alpha,
        use_bias: bool = False,
        activation: Callable[[], nn.Module] = nn.GELU,
        **kwargs,
    ):
        super().__init__()

        self.r = r
        self.alpha = alpha

        a = nn.Linear(in_dim, r, use_bias)
        nn.init.constant_(a.weight, 0.0)
        b = nn.Linear(r, out_dim, use_bias)

        self.lora = nn.Sequential(a, activation(), b)

    def forward(self, x):

        x = self.lora(x)
        x *= self.alpha // self.r

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        in_features,
        num_heads,
        steps: int = 3,
        use_lora: bool = False,
        masked_attn: bool = False,
        attn_drop_prob: float = 0.1,
        final_drop_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.use_lora = use_lora
        self.num_heads = num_heads
        self.masked_attn = masked_attn
        self.factor = (in_features // self.num_heads) ** -0.5
        self.qkv_proj = nn.Linear(in_features, 3 * in_features, bias=True)
        self.final_proj = nn.Linear(in_features, in_features, bias=True)
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.final_drop = nn.Dropout(final_drop_prob)
        if self.use_lora:
            self.lora = LoRA(in_dim=in_features, out_dim=2 * in_features, **kwargs)

        self.register_buffer(
            "mask", torch.triu(torch.ones(steps, steps), diagonal=1).bool()
        )

    def forward(self, x):

        B, N, C = x.shape
        qkv = (
            self.qkv_proj(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        if self.use_lora:
            lora_qv = (
                self.lora(x)
                .reshape(B, N, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0] + lora_qv[0], qkv[1], qkv[2] + lora_qv[1]
        else:
            q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)
        attn *= self.factor
        if self.masked_attn:
            attn = attn.masked_fill(self.mask, -torch.inf)
        attn = attn.softmax(dim=-1)
        attn_drop = self.attn_drop(attn)

        out = (attn_drop @ v).transpose(1, 2).reshape(B, N, C)
        out = self.final_proj(out)
        out = self.final_drop(out)

        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        in_features,
        steps,
        num_heads,
        use_lora: bool = False,
        masked_attn: bool = False,
        attn_drop_prob: float = 0.1,
        final_drop_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.factor = (in_features // self.num_heads) ** -0.5
        self.masked_attn = masked_attn
        self.q_proj = nn.Linear(in_features, in_features, bias=True)
        self.kv_proj = nn.Linear(in_features, 2 * in_features, bias=True)
        self.attn_drop = nn.Dropout(attn_drop_prob)
        self.final_proj = nn.Linear(in_features, in_features, bias=True)
        self.final_drop = nn.Dropout(final_drop_prob)
        self.use_lora = use_lora
        if self.use_lora:
            self.lora = LoRA(in_dim=in_features, out_dim=2 * in_features, **kwargs)
        self.register_buffer(
            "mask", torch.triu(torch.ones(steps, steps), diagonal=1).bool()
        )

    def forward(self, x, y):

        B, N, C = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        kv = (
            self.kv_proj(y)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        if self.use_lora:
            lora_qv = (
                self.lora(x)
                .reshape(B, N, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            k = kv[0]
            q, v = q + lora_qv[0], kv[1] + lora_qv[1]
        else:
            k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1)
        attn *= self.factor
        if self.masked_attn:
            attn = attn.masked_fill(self.mask, -torch.inf)
        attn = attn.softmax(dim=-1)
        attn_drop = self.attn_drop(attn)

        out = (attn_drop @ v).transpose(1, 2).reshape(B, N, C)
        out = self.final_proj(out)
        out = self.final_drop(out)

        return out


class FFN(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_features: int,
        ffn_dropout: float,
        activation: Callable[[], nn.Module] = nn.SELU,
        last_layer: bool = False,
    ):
        super().__init__()

        if isinstance(activation, nn.SELU):
            dropout = nn.AlphaDropout
        else:
            dropout = nn.Dropout

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, in_features))
            is_final = i == num_layers - 1
            if not (is_final and last_layer):
                layers.append(activation())
                layers.append(dropout(ffn_dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        return self.model(x)


class MambaLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        steps: int,
        **kwargs,
    ) -> None:
        super().__init__()

        self.layer = nn.Sequential(
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                **kwargs,
            ),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm([steps, d_model])

    def forward(self, x):

        x = x + self.layer(x)
        x = self.norm(x)
        return x
