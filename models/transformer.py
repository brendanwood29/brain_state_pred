import torch.nn as nn

from .layers import (
    FFN,
    MambaLayer,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    TrainableRegionEmbedding,
    TrainableTemporalEmbedding,
)


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
            num_layers=3,
            in_features=in_features,
            ffn_dropout=kwargs["ffn_dropout"],
            last_layer=last_layer,
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
        ffn_layers: int = 3,
        ffn_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        # First layer norm
        self.layer_norm1 = nn.LayerNorm([steps, in_features])
        # Spatial stuff
        self.embed_s = TrainableTemporalEmbedding(steps)
        self.attn_s = MultiHeadSelfAttention(
            in_features, num_heads, steps, masked_attn=True, **kwargs
        )
        self.layer_norm_s = nn.LayerNorm([steps, in_features])
        self.ffn_s = FFN(
            num_layers=ffn_layers,
            in_features=in_features,
            ffn_dropout=ffn_dropout,
            last_layer=False,
        )
        # Temporal stuff
        self.embed_t = TrainableRegionEmbedding(in_features)
        self.attn_t = MultiHeadSelfAttention(
            steps, num_heads, in_features, masked_attn=False, **kwargs
        )
        self.layer_norm_t = nn.LayerNorm([in_features, steps])
        self.ffn_t = FFN(
            num_layers=ffn_layers,
            in_features=steps,
            ffn_dropout=ffn_dropout,
            last_layer=False,
        )
        # Combining attn
        self.cross_attn = MultiHeadCrossAttention(
            in_features, steps, num_heads, masked_attn=True, **kwargs
        )
        self.layer_norm2 = nn.LayerNorm([steps, in_features])
        self.ffn = FFN(
            num_layers=ffn_layers,
            in_features=in_features,
            ffn_dropout=ffn_dropout,
            last_layer=True,
        )

    def forward(self, x):
        """
        x must have shape (Batch, Num timepoints, Num Regions)
        """
        x = self.layer_norm1(x)
        x_s = self.embed_s(x)
        x_s = x_s + self.attn_s(x_s)
        x_s = self.layer_norm_s(x_s)
        x_s = x_s + self.ffn_s(x_s)
        x_t = self.embed_t(x.permute(0, 2, 1))
        x_t = x_t + self.attn_t(x_t)
        x_t = self.layer_norm_t(x_t)
        x_t = x_t + self.ffn_t(x_t)
        x = x + self.cross_attn(x_s, x_t.permute(0, 2, 1))
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        # x = self.proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class TMBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        steps: int,
        num_mamba_blocks: int,
        mamba_kwargs: dict,
        trfrm_kwargs: dict,
        ffn_kwargs: dict,
        **kwargs,
    ):
        super().__init__()

        # Setting some custom kwargs
        trfrm_kwargs["in_features"] = in_features
        trfrm_kwargs["steps"] = steps
        # First layer norm
        self.layer_norm1 = nn.LayerNorm([steps, in_features])
        # Spatial stuff
        self.embed_s = TrainableTemporalEmbedding(steps)
        self.attn_s = MultiHeadSelfAttention(masked_attn=True, **trfrm_kwargs)
        self.layer_norm_s = nn.LayerNorm([steps, in_features])
        self.ffn_s = FFN(
            **ffn_kwargs,
            in_features=in_features,
            last_layer=False,
        )

        # Temporal stuff
        self.embed_t = TrainableRegionEmbedding(in_features)
        self.mamba = nn.Sequential(
            *[MambaLayer(steps=steps, **mamba_kwargs) for _ in range(num_mamba_blocks)]
        )
        self.ffn_t = FFN(
            **ffn_kwargs,
            in_features=in_features,
            last_layer=False,
        )

        # Combining attn
        self.cross_attn = MultiHeadCrossAttention(masked_attn=True, **trfrm_kwargs)
        self.layer_norm2 = nn.LayerNorm([steps, in_features])
        self.ffn = FFN(
            **ffn_kwargs,
            in_features=in_features,
            last_layer=True,
        )

    def forward(self, x):
        """
        x must have shape (Batch, Num timepoints, Num Regions)
        """
        x = self.layer_norm1(x)
        x_s = self.embed_s(x)
        x_s = x_s + self.attn_s(x_s)
        x_s = self.layer_norm_s(x_s)
        x_s = x_s + self.ffn_s(x_s)
        x_t = self.embed_t(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_t = x_t + self.mamba(x_t)
        x_t = x_t + self.ffn_t(x_t)
        x = x + self.cross_attn(x_s, x_t)
        x = self.layer_norm2(x)
        x = x + self.ffn(x)
        return x


class TransformerMambaModel(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_features: int,
        steps: int,
        num_mamba_blocks: int,
        mamba_kwargs: dict,
        trfrm_kwargs: dict,
        ffn_kwargs: dict,
        **kwargs,
    ) -> None:
        super().__init__()

        self.mod_list = [
            TMBlock(
                in_features=in_features,
                steps=steps,
                num_mamba_blocks=num_mamba_blocks,
                mamba_kwargs=mamba_kwargs,
                trfrm_kwargs=trfrm_kwargs,
                ffn_kwargs=ffn_kwargs,
                **kwargs,
            )
            for i in range(num_blocks)
        ]
        self.model = nn.Sequential(*self.mod_list)

    def forward(self, x):
        """
        x must have shape (Batch, Num timepoints, Num Regions)
        """
        x = self.model(x)
        return x[:, [-1], :]


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

        self.mod_list = [
            STBlock(in_features, num_heads, steps, i == (num_blocks - 1), **kwargs)
            for i in range(num_blocks)
        ]
        self.pred_len = pred_len
        self.model = nn.Sequential(*self.mod_list)
        self.steps = steps

    def forward(self, x):
        """
        x must have shape (Batch, Num timepoints, Num Regions)
        """
        # outputs = [x[:, i, :] for i in range(x.shape[1])]
        # for _ in range(self.pred_len):
        #     inputs = torch.stack(outputs[-self.steps :], dim=1)
        #     outputs.append(self.model(inputs)[:, -1, :])
        # outputs = torch.stack(outputs[-self.pred_len :], dim=1)
        # return outputs

        x = self.model(x)
        return x[:, [-1], :]


if __name__ == "__main__":
    pass
