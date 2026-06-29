"""
Minimal Clay v1.5 encoder for inference-only use.

Extracted from the Clay Foundation Model (made-with-clay/Clay v1.5,
Apache 2.0 license): https://github.com/Clay-foundation/model

Bundled here to avoid the broken claymodel PyPI wheel (files land at
site-packages root, breaking all 'from src.*' imports) and the heavy
lightning/vit-pytorch transitive dependencies that aren't needed for
inference.  Only torch and einops are required — both already present in
the embedding-generation virtualenv.

Architecture: Clay ViT-Large encoder
  dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4, patch_size=8
"""
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


# ---------------------------------------------------------------------------
# Positional encodings (from src/utils.py)
# ---------------------------------------------------------------------------

def posemb_sincos_1d(waves, dim, temperature=10000, dtype=torch.float32):
    """1-D sinusoidal positional encoding for wavelengths."""
    assert dim % 2 == 0, "dim must be multiple of 2"
    waves = torch.arange(waves) if isinstance(waves, int) else waves
    omega = torch.arange(dim // 2, device=waves.device) / (dim // 2 - 1)
    omega = 1.0 / (temperature ** omega)
    scaled = waves[:, None] * omega[None, :]  # (N, dim/2)
    return torch.cat((scaled.sin(), scaled.cos()), dim=1).to(dtype)  # (N, dim)


def posemb_sincos_2d_with_gsd(h, w, dim, gsd=1.0, temperature=10000,
                               dtype=torch.float32):
    """2-D sinusoidal positional encoding scaled by ground sampling distance."""
    assert dim % 4 == 0, "dim must be multiple of 4"
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    gsd = gsd.to(x.device)
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1).to(dtype)


# ---------------------------------------------------------------------------
# Transformer blocks (from src/backbone.py)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, fused_attn=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.fused_attn = fused_attn
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = (torch.matmul(q, k.transpose(-1, -2)) * self.scale).softmax(dim=-1)
            x = torch.matmul(attn, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.to_out(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, fused_attn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, fused_attn=fused_attn),
                FeedForward(dim, mlp_dim),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# ---------------------------------------------------------------------------
# DOFA dynamic patch embedding (from src/factory.py)
# ---------------------------------------------------------------------------

class FCBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.l1 = nn.Linear(size, size)
        self.l2 = nn.Linear(size, size)

    def forward(self, x):
        y = F.gelu(self.l1(x))
        y = F.gelu(self.l2(y))
        return x + y


class WavesTransformer(nn.Module):
    def __init__(self, wave_dim, output_dim, num_latent_tokens, embed_dim,
                 is_decoder, num_heads=4, num_layers=1):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.is_decoder = is_decoder
        layer = nn.TransformerEncoderLayer(
            d_model=wave_dim, nhead=num_heads, activation="gelu",
            dropout=0, norm_first=False, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc_weight = nn.Linear(wave_dim, output_dim)
        self.fc_bias = None if is_decoder else nn.Linear(wave_dim, embed_dim)
        self.weight_tokens = nn.Parameter(
            torch.randn(num_latent_tokens, wave_dim) * 0.02
        )
        self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * 0.02)

    def forward(self, x):
        x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
        out = self.encoder(x)
        weights = self.fc_weight(
            out[self.num_latent_tokens:-1] + x[self.num_latent_tokens:-1]
        )
        bias = None if self.is_decoder else self.fc_bias(out[-1])
        return weights, bias


class DynamicEmbedding(nn.Module):
    def __init__(self, wave_dim, num_latent_tokens, patch_size, embed_dim,
                 is_decoder=False):
        super().__init__()
        self.wave_dim = wave_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder
        self.output_dim = (patch_size ** 2) * embed_dim
        self.weight_generator = WavesTransformer(
            wave_dim, self.output_dim, num_latent_tokens, embed_dim, is_decoder
        )
        self.fclayer = FCBlock(wave_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch, waves):
        waves = posemb_sincos_1d(waves, self.wave_dim).to(batch.device)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)
        if self.is_decoder:
            dynamic_weight = rearrange(
                weight,
                "cin (k1 k2 cout) -> (cin k1 k2) cout",
                k1=self.patch_size, k2=self.patch_size, cout=self.embed_dim,
            )
            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            return F.linear(batch, dynamic_weight * 0.02, bias=bias), waves
        else:
            dynamic_weight = rearrange(
                weight,
                "cin (cout k1 k2) -> cout cin k1 k2",
                k1=self.patch_size, k2=self.patch_size,
            )
            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            out = F.conv2d(batch, dynamic_weight * 0.02, bias=bias,
                           stride=self.patch_size)
            return rearrange(out, "b c h w -> b (h w) c"), waves


# ---------------------------------------------------------------------------
# Clay Encoder (extracted from src/model.py's Encoder class)
# ---------------------------------------------------------------------------

class ClayEncoder(nn.Module):
    """Clay ViT encoder.

    Inputs (via ``datacube`` dict):
      pixels  (B, C, H, W)  — normalized imagery
      time    (B, 4)         — [sin_week, cos_week, sin_hour, cos_hour]
      latlon  (B, 4)         — [sin_lat, cos_lat, sin_lon, cos_lon]
      gsd     scalar tensor  — ground sampling distance in metres
      waves   (C,)           — band centre wavelengths in micrometres

    Returns with mask_ratio=0, shuffle=False:
      encoded  (B, 1 + L, dim)  — [CLS token, L spatial patch tokens]
      (plus three index/mask tensors from the masking step, all unused at
       inference time since no patches are masked out)

    Spatial reconstruction with L=1024 (32×32 grid for 256-px chip):
      patches = encoded[:, 1:, :].reshape(B, 32, 32, dim)
    """

    def __init__(self, mask_ratio, patch_size, shuffle, dim,
                 depth, heads, dim_head, mlp_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.patch_embedding = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=False,
        )
        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads,
            dim_head=dim_head, mlp_dim=int(dim * mlp_ratio),
            fused_attn=True,
        )

    def _add_encodings(self, patches, time, latlon, gsd):
        B, L, D = patches.shape
        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size ** 2
        pos_enc = posemb_sincos_2d_with_gsd(
            h=grid_size, w=grid_size, dim=(self.dim - 8), gsd=gsd,
        ).to(patches.device).detach()        # (L, D-8)
        time_latlon = torch.hstack((time, latlon)).to(patches.device).detach()  # (B, 8)
        pos_enc = repeat(pos_enc, "L D -> B L D", B=B)
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)
        return patches + torch.cat((pos_enc, time_latlon), dim=-1)

    def _mask_out(self, patches):
        B, L, D = patches.shape
        if self.shuffle:
            noise = torch.randn((B, L), device=patches.device)
        else:
            noise = rearrange(
                torch.arange(B * L, device=patches.device), "(B L) -> B L", B=B, L=L
            )
        rand_idx = torch.argsort(noise, dim=-1)
        rev_idx = torch.argsort(rand_idx, dim=-1)
        n_masked = int(self.mask_ratio * self.num_patches)
        masked_idx, unmasked_idx = rand_idx[:, :n_masked], rand_idx[:, n_masked:]
        masked_mat = torch.zeros((B, L), device=patches.device)
        masked_mat[:, :n_masked] = 1
        masked_mat = torch.gather(masked_mat, dim=1, index=rev_idx)
        bidx = rearrange(torch.arange(B, device=patches.device), "B -> B 1")
        unmasked = patches[bidx, unmasked_idx, :]
        return unmasked, unmasked_idx, masked_idx, masked_mat

    def forward(self, datacube):
        cube = datacube["pixels"]    # (B, C, H, W)
        time = datacube["time"]      # (B, 4)
        latlon = datacube["latlon"]  # (B, 4)
        gsd = datacube["gsd"]        # scalar tensor
        waves = datacube["waves"]    # (C,)

        patches, _ = self.patch_embedding(cube, waves)  # (B, L, D)
        patches = self._add_encodings(patches, time, latlon, gsd)
        unmasked, unmasked_idx, masked_idx, masked_mat = self._mask_out(patches)
        B = cube.shape[0]
        cls = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)
        encoded = self.transformer(torch.cat((cls, unmasked), dim=1))
        return encoded, unmasked_idx, masked_idx, masked_mat
