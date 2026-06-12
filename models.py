# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        text_conditioning=False,
        text_embed_dim=1024,
        max_text_len=128,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.text_conditioning = text_conditioning
        self.max_text_len = max_text_len
        self.class_dropout_prob = class_dropout_prob

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        if text_conditioning:
            self.text_token_proj = nn.Linear(text_embed_dim, hidden_size)
            self.text_pool_proj = nn.Linear(text_embed_dim, hidden_size)
            self.null_text_tokens = nn.Parameter(torch.zeros(1, max_text_len, text_embed_dim))
            self.null_text_pooled = nn.Parameter(torch.zeros(1, text_embed_dim))
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        if self.text_conditioning:
            nn.init.xavier_uniform_(self.text_token_proj.weight)
            nn.init.constant_(self.text_token_proj.bias, 0)
            nn.init.xavier_uniform_(self.text_pool_proj.weight)
            nn.init.constant_(self.text_pool_proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def prepare_text_conditioning(self, text_tokens, text_pooled, text_mask=None, drop_caption=None):
        """
        Applies text dropout for CFG and projects frozen text encoder embeddings
        into the DiT hidden space. This path is used for caption conditioning.
        """
        assert self.text_conditioning, "Text inputs require text_conditioning=True."
        assert text_tokens is not None and text_pooled is not None, "Text conditioning needs token and pooled embeddings."
        bsz, text_len, _ = text_tokens.shape
        assert text_len <= self.max_text_len, f"text_len={text_len} exceeds max_text_len={self.max_text_len}"

        if drop_caption is None:
            if self.training and self.class_dropout_prob > 0:
                drop_ids = torch.rand(bsz, device=text_tokens.device) < self.class_dropout_prob
            else:
                drop_ids = torch.zeros(bsz, device=text_tokens.device, dtype=torch.bool)
        else:
            drop_ids = drop_caption.to(device=text_tokens.device).bool()

        null_tokens = self.null_text_tokens[:, :text_len].to(dtype=text_tokens.dtype, device=text_tokens.device)
        null_tokens = null_tokens.expand(bsz, -1, -1)
        null_pooled = self.null_text_pooled.to(dtype=text_pooled.dtype, device=text_pooled.device).expand(bsz, -1)
        text_tokens = torch.where(drop_ids[:, None, None], null_tokens, text_tokens)
        text_pooled = torch.where(drop_ids[:, None], null_pooled, text_pooled)

        text_tokens = self.text_token_proj(text_tokens)
        if text_mask is not None:
            text_tokens = text_tokens * text_mask.to(device=text_tokens.device, dtype=text_tokens.dtype).unsqueeze(-1)
        text_pooled = self.text_pool_proj(text_pooled)
        return text_tokens, text_pooled, drop_ids

    def prepare_feedback_tokens(self, feedback_tokens, feedback_mask=None, drop_ids=None):
        """Project feedback text embeddings with the existing text projection."""
        if feedback_tokens is None:
            return None
        feedback_tokens = self.text_token_proj(feedback_tokens)
        if feedback_mask is not None:
            feedback_tokens = feedback_tokens * feedback_mask.to(
                device=feedback_tokens.device,
                dtype=feedback_tokens.dtype,
            ).unsqueeze(-1)
        if drop_ids is not None:
            feedback_tokens = torch.where(
                drop_ids[:, None, None].to(device=feedback_tokens.device),
                torch.zeros_like(feedback_tokens),
                feedback_tokens,
            )
        return feedback_tokens

    def prepare_attempt_tokens(self, attempt_latent, drop_ids=None):
        """Patch/project attempted-image VAE latents with the existing x_embedder."""
        if attempt_latent is None:
            return None
        attempt_tokens = self.x_embedder(attempt_latent) + self.pos_embed
        if drop_ids is not None:
            attempt_tokens = torch.where(
                drop_ids[:, None, None].to(device=attempt_tokens.device),
                torch.zeros_like(attempt_tokens),
                attempt_tokens,
            )
        return attempt_tokens

    def forward(
        self,
        x,
        t,
        y=None,
        text_tokens=None,
        text_mask=None,
        text_pooled=None,
        feedback_tokens=None,
        feedback_mask=None,
        feedback_pooled=None,
        attempt_latent=None,
        drop_all_cond=None,
        drop_caption=None,
    ):
        """
        Forward pass of DiT.
        x: target noisy latent. Optional feedback/image context is prepended as
        transformer context tokens and removed before the final layer.
        """
        target_tokens = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        context_tokens = []

        if text_tokens is not None:
            bsz = text_tokens.shape[0]
            if drop_all_cond is None:
                if self.training and self.class_dropout_prob > 0:
                    full_drop_ids = torch.rand(bsz, device=text_tokens.device) < self.class_dropout_prob
                else:
                    full_drop_ids = torch.zeros(bsz, device=text_tokens.device, dtype=torch.bool)
            else:
                full_drop_ids = drop_all_cond.to(device=text_tokens.device).bool()
            caption_drop_ids = full_drop_ids
            if drop_caption is not None:
                caption_drop_ids = caption_drop_ids | drop_caption.to(device=text_tokens.device).bool()
            caption_tokens, caption_pooled, drop_ids = self.prepare_text_conditioning(
                text_tokens=text_tokens,
                text_pooled=text_pooled,
                text_mask=text_mask,
                drop_caption=caption_drop_ids,
            )
            context_tokens.append(caption_tokens)
            feedback_tokens = self.prepare_feedback_tokens(
                feedback_tokens=feedback_tokens,
                feedback_mask=feedback_mask,
                drop_ids=full_drop_ids,
            )
            if feedback_tokens is not None:
                context_tokens.append(feedback_tokens)
            attempt_tokens = self.prepare_attempt_tokens(attempt_latent=attempt_latent, drop_ids=full_drop_ids)
            if attempt_tokens is not None:
                context_tokens.append(attempt_tokens)
            c = t + caption_pooled
        else:
            drop_ids = None
            if y is None:
                c = t
            else:
                y = self.y_embedder(y, self.training)
                c = t + y

        context_len = sum(tokens.shape[1] for tokens in context_tokens)
        if context_tokens:
            x = torch.cat([*context_tokens, target_tokens], dim=1)
        else:
            x = target_tokens

        for block in self.blocks:
            x = block(x, c)
        if context_len:
            x = x[:, context_len:]
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_text_cfg(
        self,
        x,
        t,
        text_tokens,
        text_mask,
        text_pooled,
        cfg_scale,
        feedback_tokens=None,
        feedback_mask=None,
        feedback_pooled=None,
        attempt_latent=None,
        drop_caption=None,
    ):
        """
        Text/adaptive classifier-free guidance. The incoming batch is expected
        to contain conditional rows followed by unconditional rows.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        drop_all_cond = torch.cat([
            torch.zeros(len(half), device=x.device, dtype=torch.bool),
            torch.ones(len(half), device=x.device, dtype=torch.bool),
        ], dim=0)
        model_out = self.forward(
            combined,
            t,
            text_tokens=text_tokens,
            text_mask=text_mask,
            text_pooled=text_pooled,
            feedback_tokens=feedback_tokens,
            feedback_mask=feedback_mask,
            feedback_pooled=feedback_pooled,
            attempt_latent=attempt_latent,
            drop_all_cond=drop_all_cond,
            drop_caption=drop_caption,
        )
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
