"""PromptIR (NeurIPS 2023) reimplemented on top of the Restormer backbone.

Faithful to the mechanism in Potlapalli et al., "PromptIR: Prompting for
All-in-One Blind Image Restoration": a Restormer encoder-decoder with a
PromptGenBlock at each decoder stage that generates an input-conditioned prompt
from a learnable prompt bank and fuses it back into the decoder features.
"""
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# --------------------------------------------------------------------------- #
# Restormer building blocks
# --------------------------------------------------------------------------- #
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, ln_type='WithBias'):
        super().__init__()
        self.body = BiasFreeLayerNorm(dim) if ln_type == 'BiasFree' else WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """GDFN."""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    """MDTA: multi-Dconv-head transposed attention."""
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, ln_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, ln_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, ln_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# --------------------------------------------------------------------------- #
# Prompt block
# --------------------------------------------------------------------------- #
class PromptGenBlock(nn.Module):
    """Input-conditioned prompt: soft-weight a learnable prompt bank by a global
    descriptor of the incoming feature, upsample to the feature resolution, and
    smooth with a 3x3 conv.
    """
    def __init__(self, prompt_dim, prompt_len, prompt_size, lin_dim):
        super().__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        emb = x.mean(dim=(-2, -1))  # (b, c)
        weights = F.softmax(self.linear_layer(emb), dim=1)  # (b, prompt_len)
        prompt = weights.view(b, -1, 1, 1, 1) * self.prompt_param.repeat(b, 1, 1, 1, 1)
        prompt = prompt.sum(dim=1)  # (b, prompt_dim, ps, ps)
        prompt = F.interpolate(prompt, (h, w), mode='bilinear', align_corners=False)
        return self.conv3x3(prompt)


# --------------------------------------------------------------------------- #
# PromptIR
# --------------------------------------------------------------------------- #
class PromptIR(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=(4, 6, 6, 8),
                 num_refinement_blocks=4,
                 heads=(1, 2, 4, 8),
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 prompt_len=5):
        super().__init__()

        def block(d, h, n):
            return nn.Sequential(*[TransformerBlock(d, h, ffn_expansion_factor, bias, LayerNorm_type)
                                   for _ in range(n)])

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)

        self.encoder_level1 = block(dim, heads[0], num_blocks[0])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = block(dim * 2, heads[1], num_blocks[1])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = block(dim * 4, heads[2], num_blocks[2])
        self.down3_4 = Downsample(dim * 4)
        self.latent = block(dim * 8, heads[3], num_blocks[3])

        # prompts operate on the feature entering each fusion point
        self.prompt3 = PromptGenBlock(dim * 8, prompt_len, 16, dim * 8)
        self.noise_level3 = TransformerBlock(dim * 16, heads[3], ffn_expansion_factor, bias, LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(dim * 16, dim * 8, 1, bias=bias)

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = block(dim * 4, heads[2], num_blocks[2])

        self.prompt2 = PromptGenBlock(dim * 4, prompt_len, 32, dim * 4)
        self.noise_level2 = TransformerBlock(dim * 8, heads[2], ffn_expansion_factor, bias, LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = block(dim * 2, heads[1], num_blocks[1])

        self.prompt1 = PromptGenBlock(dim * 2, prompt_len, 64, dim * 2)
        self.noise_level1 = TransformerBlock(dim * 4, heads[1], ffn_expansion_factor, bias, LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)

        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1 = block(dim * 2, heads[0], num_blocks[0])

        self.refinement = block(dim * 2, heads[0], num_refinement_blocks)
        self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img):
        x1 = self.patch_embed(inp_img)
        e1 = self.encoder_level1(x1)
        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))
        latent = self.latent(self.down3_4(e3))

        latent = self.reduce_noise_level3(self.noise_level3(
            torch.cat([latent, self.prompt3(latent)], 1)))

        d3 = self.up4_3(latent)
        d3 = self.decoder_level3(self.reduce_chan_level3(torch.cat([d3, e3], 1)))
        d3 = self.reduce_noise_level2(self.noise_level2(
            torch.cat([d3, self.prompt2(d3)], 1)))

        d2 = self.up3_2(d3)
        d2 = self.decoder_level2(self.reduce_chan_level2(torch.cat([d2, e2], 1)))
        d2 = self.reduce_noise_level1(self.noise_level1(
            torch.cat([d2, self.prompt1(d2)], 1)))

        d1 = self.up2_1(d2)
        d1 = self.decoder_level1(torch.cat([d1, e1], 1))
        d1 = self.refinement(d1)
        return self.output(d1) + inp_img

    @torch.no_grad()
    def prompt_weights(self, inp_img):
        """Return the three stages' softmax prompt weights for a batch, for
        UMAP / separability analysis of whether prompts distinguish D1-D6."""
        x1 = self.patch_embed(inp_img)
        e1 = self.encoder_level1(x1)
        e2 = self.encoder_level2(self.down1_2(e1))
        e3 = self.encoder_level3(self.down2_3(e2))
        latent = self.latent(self.down3_4(e3))
        w3 = F.softmax(self.prompt3.linear_layer(latent.mean(dim=(-2, -1))), dim=1)
        latent = self.reduce_noise_level3(self.noise_level3(torch.cat([latent, self.prompt3(latent)], 1)))
        d3 = self.decoder_level3(self.reduce_chan_level3(torch.cat([self.up4_3(latent), e3], 1)))
        w2 = F.softmax(self.prompt2.linear_layer(d3.mean(dim=(-2, -1))), dim=1)
        d3 = self.reduce_noise_level2(self.noise_level2(torch.cat([d3, self.prompt2(d3)], 1)))
        d2 = self.decoder_level2(self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], 1)))
        w1 = F.softmax(self.prompt1.linear_layer(d2.mean(dim=(-2, -1))), dim=1)
        return w1, w2, w3
