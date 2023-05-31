"""
Adapted from: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/rvt.py
"""
import random
from math import pi
from typing import List, Tuple, Union

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn

from src.models.modules.attention_modules import *


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, freq_type="lucidrains", **kwargs):
        super().__init__()
        self.dim = dim
        self.freq_type = freq_type
        if freq_type == "lucidrains":
            scales = torch.linspace(1.0, kwargs["max_freq"] / 2, self.dim // 4)
        elif freq_type == "vaswani":
            scales = 1 / (
                kwargs["base"] ** (torch.arange(0, self.dim, 4).float() / self.dim)
            )
        else:
            NotImplementedError(
                f"Only 'lucidrains' and 'vaswani' frequencies are implemented, but you chose {freq_type}."
            )
        self.register_buffer("scales", scales)

    def forward(self, coords: torch.Tensor):
        """
        Assumes that coordinates do not change throughout the batches.
        Args:
            coords (torch.Tensor): Coordinates of shape [B, 2, H, W]
        """
        seq_x = coords[:, 0, 0, :]
        seq_x = seq_x.unsqueeze(-1)
        seq_y = coords[:, 1, :, 0]
        seq_y = seq_y.unsqueeze(-1)

        scales = self.scales[(*((None, None)), Ellipsis)]
        scales = scales.to(coords)

        if self.freq_type == "lucidrains":
            seq_x = seq_x * scales * pi
            seq_y = seq_y * scales * pi
        elif self.freq_type == "vaswani":
            seq_x = seq_x * scales
            seq_y = seq_y * scales

        x_sinu = repeat(seq_x, "b i d -> b i j d", j=seq_y.shape[1])
        y_sinu = repeat(seq_y, "b j d -> b i j d", i=seq_x.shape[1])

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim=-1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim=-1)

        sin, cos = map(lambda t: rearrange(t, "b i j d -> b (i j) d"), (sin, cos))
        sin, cos = map(lambda t: repeat(t, "b n d -> b n (d j)", j=2), (sin, cos))
        return sin, cos


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, num_frames, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, T, C]
        """
        x += self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        image_size: Union[List[int], Tuple[int], int],
        dropout: float = 0.0,
        use_rotary: bool = True,
        use_glu: bool = True,
    ):
        super().__init__()
        self.image_size = image_size

        self.blocks = nn.ModuleList([])

        for _ in range(depth):
            self.blocks.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            SelfAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                use_rotary=use_rotary,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                        ),
                    ]
                )
            )

    def forward(
        self,
        src: torch.Tensor,
        src_pos_emb: torch.Tensor,
    ):
        """
        Performs the following computation in each layer:
            1. Self-Attention on the source sequence
            2. FFN on the source sequence
        Args:
            src: Source sequence of shape [B, N, D]
            src_pos_emb: Positional embedding of source sequence's tokens of shape [B, N, D]
        """

        attention_scores = {}
        for i in range(len(self.blocks)):
            sattn, sff = self.blocks[i]

            out, sattn_scores = sattn(src, pos_emb=src_pos_emb)
            attention_scores["self_attention"] = sattn_scores
            src = out + src
            src = sff(src) + src

        return src, attention_scores


class CrossTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        image_size: Union[List[int], Tuple[int], int],
        dropout: float = 0.0,
        use_rotary: bool = True,
        use_glu: bool = True,
    ):
        super().__init__()
        self.image_size = image_size

        self.cross_layers = nn.ModuleList([])

        for _ in range(depth):
            self.cross_layers.append(
                nn.ModuleList(
                    [
                        CrossPreNorm(
                            dim,
                            CrossAttention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                use_rotary=use_rotary,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout=dropout, use_glu=use_glu),
                        ),
                    ]
                )
            )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pos_emb: torch.Tensor,
        tgt_pos_emb: torch.Tensor,
    ):
        """
        Performs the following computation in each layer:
            1. Self-Attention on the source sequence
            2. FFN on the source sequence
            3. Cross-Attention between target and source sequence
            4. FFN on the target sequence
        Args:
            src: Source sequence of shape [B, N, D]
            tgt: Target sequence of shape [B, M, D]
            src_pos_emb: Positional embedding of source sequence's tokens of shape [B, N, D]
            tgt_pos_emb: Positional embedding of target sequence's tokens of shape [B, M, D]
        """

        attention_scores = {}
        for i in range(len(self.cross_layers)):
            cattn, cff = self.cross_layers[i]

            out, cattn_scores = cattn(src, src_pos_emb, tgt, tgt_pos_emb)
            attention_scores["cross_attention"] = cattn_scores
            tgt = out + tgt
            tgt = cff(tgt) + tgt

        return tgt, attention_scores


class RoCrossViViT_bis(nn.Module):
    def __init__(
        self,
        image_size: Union[List[int], Tuple[int]],
        patch_size: Union[List[int], Tuple[int]],
        time_coords_encoder: nn.Module,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: int = 4,
        ctx_channels: int = 3,
        ts_channels: int = 3,
        ts_length: int = 48,
        out_dim: int = 1,
        dim_head: int = 64,
        dropout: float = 0.0,
        freq_type: str = "lucidrains",
        use_rotary: bool = True,
        num_mlp_heads: int = 1,
        use_glu: bool = True,
        ctx_masking_ratio: float = 0.9,
        ts_masking_ratio: float = 0.9,
        decoder_dim: int = 128,
        decoder_depth: int = 4,
        decoder_heads: int = 6,
        decoder_dim_head: int = 128,
        **kwargs,
    ):
        super().__init__()
        assert (
            ctx_masking_ratio >= 0 and ctx_masking_ratio < 1
        ), "ctx_masking_ratio must be in [0,1)"
        self.time_coords_encoder = time_coords_encoder
        self.ctx_channels = ctx_channels
        self.ts_channels = ts_channels
        if hasattr(self.time_coords_encoder, "dim"):
            self.ctx_channels += self.time_coords_encoder.dim
            self.ts_channels += self.time_coords_encoder.dim

        self.image_size = image_size
        self.patch_size = patch_size
        self.ctx_masking_ratio = ctx_masking_ratio
        self.ts_masking_ratio = ts_masking_ratio
        self.num_mlp_heads = num_mlp_heads

        for i in range(2):
            ims = self.image_size[i]
            ps = self.patch_size[i]
            assert (
                ims % ps == 0
            ), "Image dimensions must be divisible by the patch size."

        patch_dim = self.ctx_channels * self.patch_size[0] * self.patch_size[1]

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            ),
            nn.Linear(patch_dim, dim),
        )
        self.enc_pos_emb = AxialRotaryEmbedding(dim_head, freq_type, **kwargs)
        self.ts_embedding = nn.Linear(self.ts_channels, dim)
        self.ctx_encoder = VisionTransformer(
            dim,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            image_size,
            dropout,
            use_rotary,
            use_glu,
        )

        self.mixer = CrossTransformer(
            dim,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            image_size,
            dropout,
            use_rotary,
            use_glu,
        )
        self.ctx_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.ts_encoder = Transformer(
            dim,
            ts_length,
            depth,
            heads,
            dim_head,
            dim * mlp_ratio,
            dropout=dropout,
        )
        self.ts_enctodec = nn.Linear(dim, decoder_dim)
        self.temporal_transformer = Transformer(
            decoder_dim,
            ts_length,
            decoder_depth,
            decoder_heads,
            decoder_dim_head,
            decoder_dim * mlp_ratio,
            dropout=dropout,
        )
        self.ts_mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.mlp_heads = nn.ModuleList([])
        for i in range(num_mlp_heads):
            self.mlp_heads.append(
                nn.Sequential(
                    nn.LayerNorm(decoder_dim),
                    nn.Linear(decoder_dim, out_dim, bias=True),
                    nn.ReLU(),
                )
            )

        self.quantile_masker = nn.Sequential(
            nn.Conv1d(decoder_dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            Rearrange(
                "b c t -> b t c",
            ),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_mlp_heads),
        )

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward(
        self,
        ctx: torch.Tensor,
        ctx_coords: torch.Tensor,
        ts: torch.Tensor,
        ts_coords: torch.Tensor,
        time_coords: torch.Tensor,
        mask: bool = True,
    ):
        """
        Args:
            ctx (torch.Tensor): Context frames of shape [B, T, C, H, W]
            ctx_coords (torch.Tensor): Coordinates of context frames of shape [B, 2, H, W]
            ts (torch.Tensor): Station timeseries of shape [B, T, C]
            ts_coords (torch.Tensor): Station coordinates of shape [B, 2, 1, 1]
            time_coords (torch.Tensor): Time coordinates of shape [B, T, C, H, W]
            mask (bool): Whether to mask or not. Useful for inference
        Returns:

        """
        B, T, _, H, W = ctx.shape
        time_coords = self.time_coords_encoder(time_coords)

        ctx = torch.cat([ctx, time_coords], axis=2)
        ts = torch.cat([ts, time_coords[..., 0, 0]], axis=-1)

        ctx = rearrange(ctx, "b t c h w -> (b t) c h w")

        ctx_coords = repeat(ctx_coords, "b c h w -> (b t) c h w", t=T)
        ts_coords = repeat(ts_coords, "b c h w -> (b t) c h w", t=T)
        src_enc_pos_emb = self.enc_pos_emb(ctx_coords)
        tgt_pos_emb = self.enc_pos_emb(ts_coords)

        ctx = self.to_patch_embedding(ctx)  # BT, N, D

        if self.ctx_masking_ratio > 0 and mask:
            p = self.ctx_masking_ratio * random.random()
            ctx, _, ids_restore, ids_keep = self.random_masking(ctx, p)
            src_enc_pos_emb = tuple(
                torch.gather(
                    pos_emb,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_emb.shape[-1]),
                )
                for pos_emb in src_enc_pos_emb
            )
        latent_ctx, self_attention_scores = self.ctx_encoder(ctx, src_enc_pos_emb)

        ts = self.ts_embedding(ts)
        if self.ts_masking_ratio > 0 and mask:
            p = self.ts_masking_ratio * random.random()
            ts, _, ids_restore, ids_keep = self.random_masking(ts, p)
            mask_tokens = self.ts_mask_token.repeat(ts.shape[0], T - ts.shape[1], 1)
            ts = torch.cat([ts, mask_tokens], dim=1)
            ts = torch.gather(
                ts, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, ts.shape[2])
            )

        latent_ts = self.ts_encoder(ts)
        latent_ts = rearrange(latent_ts, "b t c -> (b t) c").unsqueeze(1)

        latent_ts, cross_attention_scores = self.mixer(
            latent_ctx, latent_ts, src_enc_pos_emb, tgt_pos_emb
        )
        latent_ts = latent_ts.squeeze(1)
        latent_ts = self.ts_enctodec(rearrange(latent_ts, "(b t) c -> b t c", b=B))

        y = self.temporal_transformer(latent_ts)

        # Handles the multiple MLP heads
        outputs = []
        for i in range(self.num_mlp_heads):
            mlp = self.mlp_heads[i]
            output = mlp(y)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=2)

        quantile_mask = self.quantile_masker(rearrange(y.detach(), "b t c -> b c t"))

        return (outputs, quantile_mask, self_attention_scores, cross_attention_scores)
