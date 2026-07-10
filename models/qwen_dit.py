"""Qwen-conditioned DiT: the transformer, the model registry, and the QwenDiT training wrapper."""
import math

import numpy as np
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithms.utils import diffusion_loss, unwrap_model
from diffusion import create_diffusion


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, context_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
        )

    def forward(self, x, context, context_mask=None):
        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask.bool()
        out, _ = self.attn(
            x,
            context,
            context,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return out

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, context_dim=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.cross_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, context_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c, context, context_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.cross_attn(self.cross_norm(x), context, context_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    """Diffusion transformer conditioned on Qwen context tokens via cross-attention."""

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        context_dropout_prob=0.1,
        learn_sigma=True,
        context_dim=2560,
        null_context_len=1,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.context_dropout_prob = context_dropout_prob
        self.context_dim = context_dim

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.null_context = nn.Parameter(torch.zeros(1, null_context_len, context_dim))
        self.context_pool_proj = nn.Sequential(
            nn.LayerNorm(context_dim, eps=1e-6),
            nn.Linear(context_dim, hidden_size),
        )

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, context_dim=context_dim)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5),
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    def prepare_context(self, context_tokens, context_mask=None, drop_context=None):
        bsz = context_tokens.shape[0]
        if context_mask is None:
            context_mask = torch.ones(
                context_tokens.shape[:2],
                device=context_tokens.device,
                dtype=torch.bool,
            )
        else:
            context_mask = context_mask.to(device=context_tokens.device).bool()

        if drop_context is None:
            if self.training and self.context_dropout_prob > 0:
                drop_context = torch.rand(bsz, device=context_tokens.device) < self.context_dropout_prob
            else:
                drop_context = torch.zeros(bsz, device=context_tokens.device, dtype=torch.bool)
        else:
            drop_context = drop_context.to(device=context_tokens.device).bool()

        if drop_context.any():
            null_context = torch.zeros_like(context_tokens)
            null_len = min(self.null_context.shape[1], context_tokens.shape[1])
            null_context[:, :null_len] = self.null_context[:, :null_len].to(
                device=context_tokens.device,
                dtype=context_tokens.dtype,
            )
            null_mask = torch.zeros_like(context_mask)
            null_mask[:, :null_len] = True
            context_tokens = torch.where(drop_context[:, None, None], null_context, context_tokens)
            context_mask = torch.where(drop_context[:, None], null_mask, context_mask)
        else:
            context_tokens = context_tokens + self.null_context.sum().to(dtype=context_tokens.dtype) * 0
        return context_tokens, context_mask

    def adapt_context(self, context_tokens, context_mask):
        mask = context_mask.to(device=context_tokens.device, dtype=context_tokens.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_context = (context_tokens * mask.unsqueeze(-1)).sum(dim=1) / denom
        pooled_context = self.context_pool_proj(pooled_context)
        return context_tokens, context_mask, pooled_context

    def forward(self, x, t, context_tokens=None, context_mask=None, drop_context=None):
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t)
        context_tokens, context_mask = self.prepare_context(context_tokens, context_mask, drop_context)
        context_tokens, context_mask, pooled_context = self.adapt_context(context_tokens, context_mask)
        c = c + pooled_context
        for block in self.blocks:
            x = block(x, c, context_tokens, context_mask)
        x = self.final_layer(x, c)
        return self.unpatchify(x)

    def forward_with_cfg(self, x, t, cfg_scale=1.0, context_tokens=None, context_mask=None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        drop_context = torch.cat([
            torch.zeros(len(half), device=x.device, dtype=torch.bool),
            torch.ones(len(half), device=x.device, dtype=torch.bool),
        ])
        model_out = self.forward(
            combined,
            t,
            context_tokens=context_tokens,
            context_mask=context_mask,
            drop_context=drop_context,
        )
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


DiT_models = {
    "DiT-XL/2": lambda **kwargs: DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs),
    "DiT-XL/4": lambda **kwargs: DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs),
    "DiT-XL/8": lambda **kwargs: DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs),
    "DiT-L/2": lambda **kwargs: DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs),
    "DiT-L/4": lambda **kwargs: DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs),
    "DiT-L/8": lambda **kwargs: DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs),
    "DiT-B/2": lambda **kwargs: DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs),
    "DiT-B/4": lambda **kwargs: DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs),
    "DiT-B/8": lambda **kwargs: DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs),
    "DiT-S/2": lambda **kwargs: DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs),
    "DiT-S/4": lambda **kwargs: DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs),
    "DiT-S/8": lambda **kwargs: DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs),
}


class QwenDiT:
    """Qwen-conditioned DiT with its frozen VAE, training diffusion, and optional context encoder."""

    def __init__(self, name, image_size, context_dim, context_dropout_prob, vae, device, context_encoder=None):
        self.name = name
        self.image_size = int(image_size)
        self.latent_size = self.image_size // 8
        self.device = device
        self.encoder_cfg = context_encoder

        # The encoder is a trainable component only when not frozen; a frozen encoder is
        # built lazily via ensure_encoder() by the consumers that actually encode.
        self.encoder = None
        if context_encoder is not None and not context_encoder.freeze_encoder:
            self.encoder = self._build_encoder()

        if context_dim is None:
            if self.encoder is not None:
                context_dim = self.encoder.hidden_size
            elif context_encoder is not None:
                from transformers import AutoConfig

                config = AutoConfig.from_pretrained(context_encoder.model_id, trust_remote_code=True)
                context_dim = config.text_config.hidden_size
            else:
                raise ValueError("Dataset has no precomputed context tokens; set model.context_encoder.")

        self.context_dim = int(context_dim)
        self.net = DiT_models[name](
            input_size=self.latent_size,
            context_dim=self.context_dim,
            context_dropout_prob=context_dropout_prob,
        ).to(device)
        self.vae = AutoencoderKL.from_pretrained(vae).to(device).eval()
        self.diffusion = create_diffusion(timestep_respacing="")

    def _build_encoder(self):
        from models.qwen_vlm import QwenEncoder

        cfg = self.encoder_cfg
        if cfg is None:
            raise ValueError("model.context_encoder is not configured; cannot build a context encoder.")
        return QwenEncoder(
            model_id=cfg.model_id,
            device=self.device,
            dtype=cfg.dtype,
            max_length=cfg.max_length,
            freeze=cfg.freeze_encoder,
            lora_rank=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_target_modules=cfg.lora_target_modules,
        )

    def ensure_encoder(self):
        if self.encoder is None:
            self.encoder = self._build_encoder()
        return self.encoder

    def load(self, path, use_ema=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint
        if isinstance(checkpoint, dict) and ("model" in checkpoint or "ema" in checkpoint):
            key = "ema" if use_ema else "model"
            assert key in checkpoint, f"Checkpoint {path} has no '{key}' state dict."
            state = checkpoint[key]
        missing, unexpected = unwrap_model(self.net).load_state_dict(state, strict=False)
        if (
            self.encoder is not None
            and not self.encoder.freeze
            and isinstance(checkpoint, dict)
            and "context_encoder" in checkpoint
        ):
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(unwrap_model(self.encoder.model), checkpoint["context_encoder"])
        return missing, unexpected

    @torch.no_grad()
    def encode(self, images):
        return self.vae.encode(images).latent_dist.sample().mul_(self.vae.config.scaling_factor)

    def encode_batch_context(self, batch):
        """Encode a dataset batch's contexts: caption-only rows get empty histories,
        single-feedback rows become 1-step histories."""
        assert self.encoder is not None, "No context encoder available to encode batch contexts."
        feedback_histories = []
        image_histories = []
        for feedback, image in zip(batch["feedback"], batch["generated_image"]):
            assert bool(feedback) == (image is not None), (
                "Feedback rows must carry a generated image and caption-only rows must not."
            )
            feedback_histories.append([feedback] if feedback else [])
            image_histories.append([image] if image is not None else [])
        return self.encoder.encode_history(batch["caption"], feedback_histories, image_histories)

    def loss(self, batch):
        x_latent = self.encode(batch["image"].to(self.device, non_blocking=True))
        if "context_tokens" in batch and (self.encoder is None or self.encoder.freeze):
            context_tokens = batch["context_tokens"].to(self.device, non_blocking=True)
            context_mask = batch["context_mask"].to(self.device, non_blocking=True)
        else:
            context_tokens, context_mask = self.encode_batch_context(batch)
        return diffusion_loss(self.net, self.diffusion, x_latent, context_tokens, context_mask)

    @torch.no_grad()
    def generate(self, batch, num_sampling_steps, cfg_scale=1.0, ddim_eta=0.0, seed=None):
        from algorithms.on_policy import PolicySampler

        if "context_tokens" in batch and batch["context_tokens"] is not None:
            context_tokens = batch["context_tokens"]
            context_mask = batch["context_mask"]
        else:
            encoder = self.ensure_encoder()
            captions = batch["caption"]
            context_tokens, context_mask = encoder.encode_history(
                captions, [[] for _ in captions], [[] for _ in captions]
            )
        sampler = PolicySampler(
            create_diffusion(str(num_sampling_steps)),
            latent_size=self.latent_size,
            vae_scaling_factor=self.vae.config.scaling_factor,
            cfg_scale=cfg_scale,
            sampler="ddim",
            ddim_eta=ddim_eta,
        )
        _, images = sampler.sample(self.net, self.vae, context_tokens, context_mask, self.device, seed=seed)
        return images

    def ddp(self, device):
        self.net = DDP(self.net, device_ids=[device], find_unused_parameters=False)
        if self.encoder is not None and not self.encoder.freeze:
            self.encoder.model = DDP(self.encoder.model, device_ids=[device], find_unused_parameters=False)

    def trainable_parameters(self):
        params = [p for p in self.net.parameters() if p.requires_grad]
        if self.encoder is not None and not self.encoder.freeze:
            params += [p for p in self.encoder.model.parameters() if p.requires_grad]
        return params

    def checkpoint_state(self):
        return unwrap_model(self.net).state_dict()

    def encoder_state(self):
        if self.encoder is None or self.encoder.freeze:
            return None
        from peft.utils import get_peft_model_state_dict

        return get_peft_model_state_dict(unwrap_model(self.encoder.model))

    def save_extras(self, checkpoint_dir):
        pass

    def prepare_dataset(self, dataset):
        from datasets.clevr.dataset import context_collate

        if self.encoder is not None and not self.encoder.freeze:
            dataset.set_return_generated_images(True)
        return dataset, context_collate
