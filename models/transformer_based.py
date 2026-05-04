from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        activation: Callable = nn.GELU,
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

        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1)
        attn *= self.factor
        if self.masked_attn:
            attn = attn.masked_fill(self.mask, -torch.inf)
        attn = attn.softmax(dim=-1)
        attn_drop = self.attn_drop(attn)

        out = (attn_drop @ v).transpose(1, 2).reshape(B, N, C)
        out = self.final_drop(out)
        out = self.final_drop(out)

        return out


class SmoothFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ffn_dropout: float,
        last_layer: bool = False,
    ):
        super().__init__()

        factors = [int((in_features - out_features) / 3 * i) for i in range(1, 4)]
        self.last_layer = last_layer
        self.layer1 = nn.Linear(in_features, in_features - factors[0])
        self.layer2 = nn.Linear(in_features - factors[0], in_features - factors[1])
        self.layer3 = nn.Linear(in_features - factors[1], in_features - factors[2])
        self.dropout = nn.Dropout(ffn_dropout)
        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer3(x)
        if not self.last_layer:
            x = self.activation(x)
            x = self.dropout(x)

        return x


class FFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        ffn_dropout: float,
        last_layer: bool = False,
    ):
        super().__init__()

        self.last_layer = last_layer
        self.layer1 = nn.Linear(in_features, in_features)
        self.layer2 = nn.Linear(in_features, in_features)
        self.layer3 = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(ffn_dropout)
        self.activation = nn.SELU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer3(x)
        if not self.last_layer:
            x = self.activation(x)
            x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_heads: int,
        steps: int,
        last_layer: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm([steps, in_features])
        self.attn = MultiHeadSelfAttention(in_features, num_heads, steps, **kwargs)
        self.ffn = FFN(
            in_features, ffn_dropout=kwargs["ffn_dropout"], last_layer=last_layer
        )
        self.layer_norm2 = nn.LayerNorm([steps, in_features])

    def forward(self, x):
        """
        x must have shape (Batch, Num timepoints, Num Regions)
        """
        x = self.layer_norm1(x)
        x = x + self.attn(x)
        x = self.layer_norm2(x)
        x = x + self.ffn(x)

        return x


class STBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_heads: int,
        steps: int,
        last_layer: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm([steps, in_features])
        self.embed_s = TrainableTemporalEmbedding(steps)
        self.attn_s = MultiHeadSelfAttention(
            in_features, num_heads, steps, masked_attn=True, **kwargs
        )
        self.layer_norm_s = nn.LayerNorm([steps, in_features])
        self.embed_t = TrainableRegionEmbedding(in_features)
        self.attn_t = MultiHeadSelfAttention(
            steps, num_heads, in_features, masked_attn=False, **kwargs
        )
        self.layer_norm_t = nn.LayerNorm([in_features, steps])
        self.cross_attn = MultiHeadCrossAttention(
            in_features, steps, num_heads, masked_attn=True, **kwargs
        )
        self.layer_norm2 = nn.LayerNorm([steps, in_features])
        self.ffn_s = FFN(
            in_features, ffn_dropout=kwargs["ffn_dropout"], last_layer=False
        )
        self.ffn_t = FFN(steps, ffn_dropout=kwargs["ffn_dropout"], last_layer=False)
        self.ffn = FFN(in_features, ffn_dropout=kwargs["ffn_dropout"], last_layer=True)

        # self.temporal_feats = RegionalFeatureExtractor(5, 64)
        # self.temporal_norm1 = nn.LayerNorm([in_features, steps])
        # self.temp_embed = TrainableRegionEmbedding(in_features)
        # self.temp_attn = MultiHeadSelfAttention(
        #     steps, 2, in_features, False, False, **kwargs
        # )

        # self.regional_feats = nn.Sequential(
        #     nn.Linear(in_features, in_features),
        #     nn.SELU(),
        #     nn.Linear(in_features, in_features),
        #     nn.SELU(),
        #     nn.Linear(in_features, in_features),
        #     # nn.SELU(),
        # )
        # self.regional_embed = TrainableTemporalEmbedding(steps)
        # self.regional_norm1 = nn.LayerNorm([steps, in_features])
        # self.regional_attn = MultiHeadSelfAttention(
        #     in_features, 2, steps, False, True, **kwargs
        # )

        # self.combine = MultiHeadCrossAttention(in_features, steps, 2, False, **kwargs)

        # self.layer_norm2 = nn.LayerNorm([steps, in_features])
        # self.ffn = FFN(11, ffn_dropout=kwargs["ffn_dropout"], last_layer=True)
        # self.smooth_real_ffn = SmoothFFN(steps, 11, 0.1, True)

        # self.smooth_img_ffn = SmoothFFN(steps, 11, 0.1, True)

    def forward(self, x):
        """
        x must have shape (Batch, Num timepoints, Num Regions)
        """
        # B, T, R = x.shape
        # x_temp = self.temporal_feats(x.reshape(B * R, 1, T)).reshape(B, R, T)
        # x_temp = self.temp_embed(
        #     x_temp
        # )  # This adds embedding vector of length regions, to each temporal sequence
        # x_temp = self.temporal_norm1(x_temp)
        # x_temp = x_temp + self.temp_attn(x_temp)

        # x_space = self.regional_feats(x)
        # x_space = self.regional_embed(x_space)
        # x_space = self.regional_norm1(x_space)
        # x_space = x_space + self.regional_attn(x_space)

        # x = self.combine(x_temp.permute(0, 2, 1), x_space)
        # x = self.layer_norm2(x)
        # x_real = self.smooth_real_ffn(x.permute(0, 2, 1))
        # x_img = self.smooth_img_ffn(x.permute(0, 2, 1))
        # x = torch.stack([x_real, x_img], dim=0)

        # # x = self.layer_norm2(x)
        # x = x + self.ffn(x.reshape(2 * B, R, 11)).reshape(2, B, R, 11)

        x = self.layer_norm1(x)
        x_s = self.embed_s(x)
        x_s = x_s + self.attn_s(x_s)
        x_s = self.layer_norm_s(x_s)
        x_s = x_s + self.ffn_s(x_s)
        x_t = self.embed_t(x.permute(0, 2, 1))
        x_t = x_t + self.attn_t(x_t)
        x_t = self.layer_norm_t(x_t)
        x_t = x_t + self.ffn_t(x_t)
        x = x + self.cross_attn(x_t.permute(0, 2, 1), x_s)
        x = self.layer_norm2(x)
        x = x + self.ffn(x)

        return x


class RegionalFeatureExtractor(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        num_filters: int,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_feats1 = nn.Conv1d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )
        self.conv_feats2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )
        self.conv_feats3 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )
        self.act = nn.SELU()
        self.ffn = nn.Linear(num_filters, 1)
        # self.combine = nn.Conv1d(
        #     in_channels=num_filters,
        #     out_channels=1,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0
        # )

    def forward(self, x):
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0), mode="constant", value=0)
        x = self.conv_feats1(x)
        x = self.act(x)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0), mode="constant", value=0)
        x = self.conv_feats2(x)
        x = self.act(x)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0), mode="constant", value=0)
        x = self.conv_feats3(x)
        # x = self.act(x)
        weights = self.ffn(x.permute(0, 2, 1))
        weights = torch.softmax(weights, dim=1)
        x = (x * weights.permute(0, 2, 1)).sum(dim=1)
        return x


class RegionSpecificLinear(nn.Module):
    def __init__(self, num_regions: int, steps: int) -> None:
        super().__init__()
        # initalizing the same way its done in nn.Linear
        self.weights = nn.Parameter(
            torch.empty(num_regions, steps, steps)
        )  # Make num_region copies of a steps -> steps fcn
        self.bias = nn.Parameter(
            torch.empty(1, steps, num_regions)
        )  # Make num_region bias vectors of size steps
        nn.init.xavier_uniform_(self.weights, 5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        self.register_buffer("mask", torch.tril(torch.ones(steps, steps)))

    def forward(self, x) -> torch.Tensor:
        x = torch.einsum(
            "btr,rtt->btr", x, self.weights * self.mask
        )  # matmul input (t, r) with weights (t, t) `r` times #type: ignore
        return x + self.bias


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_features: int,
        num_heads: int,
        steps: int,
        pred_len: int = 10,
        **kwargs,
    ) -> None:
        super().__init__()

        # self.pos_embed = TrainablePositionEmbedding(in_features, steps)
        # self.mod_list = [
        #     STBlock(in_features, num_heads, steps, i == (num_blocks - 1), **kwargs) for i in range(num_blocks)
        # ]
        self.mod_list = [
            STBlock(in_features, num_heads, steps, True, **kwargs),
            # Block(in_features, num_heads, steps, True, **kwargs)
        ]
        self.pred_len = pred_len
        self.model = nn.Sequential(*self.mod_list)
        self.steps = steps

    def forward(self, x):
        """
        x must have shape (Batch, Num timepoints, Num Regions)
        """
        # x = self.pos_embed(x)
        # for _ in range(self.pred_len):
        #     x = torch.cat(
        #         [x, self.model(x[:, -self.steps:, :])[:, [-1], :]],
        #         dim=1
        #     )

        # return x[:, -self.pred_len:, :]
        x = self.model(x)
        return x


if __name__ == "__main__":
    pass
