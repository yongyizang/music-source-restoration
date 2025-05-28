from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from dataclasses import dataclass

class Fourier(nn.Module):
    
    def __init__(self, 
        n_fft=2048, 
        hop_length=441, 
        return_complex=True, 
        normalized=True
    ):
        super(Fourier, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.return_complex = return_complex
        self.normalized = normalized

    def stft(self, waveform):
        """
        Args:
            waveform: (b, c, samples_num)

        Returns:
            complex_sp: (b, c, t, f)
        """

        B, C, T = waveform.shape

        x = rearrange(waveform, 'b c t -> (b c) t')

        x = torch.stft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            normalized=self.normalized,
            return_complex=self.return_complex
        )
        # shape: (batch_size * channels_num, freq_bins, frames_num)

        complex_sp = rearrange(x, '(b c) f t -> b c t f', b=B, c=C)
        # shape: (batch_size, channels_num, frames_num, freq_bins)

        return complex_sp

    def istft(self, complex_sp):
        """
        Args:
            complex_sp: (batch_size, channels_num, frames_num, freq_bins)

        Returns:
            waveform: (batch_size, channels_num, samples_num)
        """

        B, C, T, F = complex_sp.shape

        x = rearrange(complex_sp, 'b c t f -> (b c) f t')

        x = torch.istft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            normalized=self.normalized,
        )
        # shape: (batch_size * channels_num, samples_num)

        x = rearrange(x, '(b c) t -> b c t', b=B, c=C)
        # shape: (batch_size, channels_num, samples_num)
        
        return x

class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.att_norm = RMSNorm(config.n_embd)
        self.att = SelfAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""

        Args:
            x: (b, t, d)
            rope: (t, head_dim/2)
            mask: (1, 1, t, t)

        Outputs:
            x: (b, t, d)
        """
        x = x + self.att(self.att_norm(x), rope, mask)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalization.

    Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""RMSNorm.

        Args:
            x: (b, t, d)
           
        Outputs:
            x: (b, t, d)
        """
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.scale
        return output


class SelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""Causal self attention.

        b: batch size
        t: time steps
        d: latent dim
        h: heads num

        Args:
            x: (b, t, d)
            rope: (t, head_dim/2, 2)
            mask: (1, 1, )

        Outputs:
            x: (b, t, d)
        """
        B, T, D = x.shape

        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # q, k, v shapes: (b, t, d)

        k = k.view(B, T, self.n_head, D // self.n_head)
        q = q.view(B, T, self.n_head, D // self.n_head)
        v = v.view(B, T, self.n_head, D // self.n_head)
        # q, k, v shapes: (b, t, h, head_dim)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)
        # q, k shapes: (b, t, h, head_dim)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # q, k, v shapes: (b, h, t, head_dim)

        # Efficient attention using Flash Attention CUDA kernels
        x = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=mask, 
            dropout_p=0.0
        )
        # shape: (b, h, t, head_dim)

        x = x.transpose(1, 2).contiguous().view(B, T, D)  # shape: (b, t, d)

        # output projection
        x = self.c_proj(x)  # shape: (b, t, d)
        
        return x


class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # The hyper-parameters follow https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3) 

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Causal self attention.

        Args:
            x: (b, t, d)
           
        Outputs:
            x: (b, t, d)
        """
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x

def build_rope(
    seq_len: int, head_dim: int, base: int = 10000
) -> torch.Tensor:
    r"""Rotary Position Embedding.
    Modified from: https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py

    Args:
        seq_len: int, e.g., 1024
        head_dim: head dim, e.g., 768/24
        base: int

    Outputs:
        cache: (t, head_dim/2, 2)
    """
    
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2) / head_dim))

    seq_idx = torch.arange(seq_len)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    return cache


def apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


@dataclass
class UFormerConfig:
    sr: float = 44100
    n_fft: int = 2048
    hop_length: int = 441
    
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256

class UFormer(Fourier):
    def __init__(self, config: UFormerConfig) -> None:
        
        super(UFormer, self).__init__(
            n_fft=config.n_fft, 
            hop_length=config.hop_length, 
            return_complex=True, 
            normalized=True
        )

        self.ds_factor = 16  # Downsample factor
        self.fps = config.sr // config.hop_length
        
        self.audio_channels = 2
        self.cmplx_num = 2
        in_channels = self.audio_channels * self.cmplx_num

        self.encoder_block1 = EncoderBlock(in_channels, 16)
        self.encoder_block2 = EncoderBlock(16, 64)
        self.encoder_block3 = EncoderBlock(64, 256)
        self.encoder_block4 = EncoderBlock(256, config.n_embd)
        self.decoder_block1 = DecoderBlock(config.n_embd, 256)
        self.decoder_block2 = DecoderBlock(256, 64)
        self.decoder_block3 = DecoderBlock(64, 16)
        self.decoder_block4 = DecoderBlock(16, 16)
        
        self.t_blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.f_blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.head_dim = config.n_embd // config.n_head
        
        t_rope = build_rope(seq_len=config.n_fft // 16, head_dim=self.head_dim)
        f_rope = build_rope(seq_len=self.fps * 20, head_dim=self.head_dim)
        self.register_buffer(name="t_rope", tensor=t_rope)  # shape: (t, head_dim/2, 2)
        self.register_buffer(name="f_rope", tensor=f_rope)  # shape: (t, head_dim/2, 2)

        self.post_fc = nn.Conv2d(
            in_channels=16, 
            out_channels=in_channels, 
            kernel_size=1, 
            padding=0,
        )

    def forward(self, audio):
        """Separation model.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: freq_bins

        Args:
            audio: (b, c, t)

        Outputs:
            output: (b, c, t)
        """

        # Complex spectrum
        complex_sp = self.stft(audio)  # shape: (b, c, t, f)

        x = torch.view_as_real(complex_sp)  # shape: (b, c, t, f, 2)
        x = rearrange(x, 'b c t f k -> b (c k) t f')  # shape: (b, d, t, f)

        # pad stft
        x, pad_t = self.pad_tensor(x)  # x: (b, d, t, f)
        B = x.shape[0]
        
        x1, latent1 = self.encoder_block1(x)
        x2, latent2 = self.encoder_block2(x1)
        x3, latent3 = self.encoder_block3(x2)
        x, latent4 = self.encoder_block4(x3)
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, self.t_rope, mask=None)  # shape: (b*f, t, d)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, self.f_rope, mask=None)  # shape: (b*t, f, d)

            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)
        x5 = self.decoder_block1(x, latent4)
        x6 = self.decoder_block2(x5, latent3)
        x7 = self.decoder_block3(x6, latent2)
        x8 = self.decoder_block4(x7, latent1)
        x = self.post_fc(x8)

        x = rearrange(x, 'b (c k) t f -> b c t f k', k=self.cmplx_num).contiguous()
        x = x.to(torch.float)  # compatible with bf16
        mask = torch.view_as_complex(x)  # shape: (b, c, t, f)
        
        # Unpad mask to the original shape
        mask = self.unpad_tensor(mask, pad_t)  # shape: (b, c, t, f)

        # Calculate stft of separated audio
        # sep_stft = mask * complex_sp  # shape: (b, c, t, f)

        # ISTFT
        output = self.istft(mask)  # shape: (b, c, l)

        return output

    def pad_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad a spectrum that can be evenly divided by downsample_ratio.

        Args:
            x: E.g., (b, c, t=201, f=1025)
        
        Outpus:
            output: E.g., (b, c, t=208, f=1024)
        """

        # Pad last frames, e.g., 201 -> 208
        T = x.shape[2]
        pad_t = -T % self.ds_factor
        x = F.pad(x, pad=(0, 0, 0, pad_t))

        # Remove last frequency bin, e.g., 1025 -> 1024
        x = x[:, :, :, 0 : -1]

        return x, pad_t

    def unpad_tensor(self, x: torch.Tensor, pad_t: int) -> torch.Tensor:
        """Unpad a spectrum to the original shape.

        Args:
            x: E.g., (b, c, t=208, f=1024)
        
        Outpus:
            x: E.g., (b, c, t=201, f=1025)
        """

        # Pad last frequency bin, e.g., 1024 -> 1025
        x = F.pad(x, pad=(0, 1))

        # Unpad last frames, e.g., 208 -> 201
        x = x[:, :, 0 : -pad_t, :]

        return x


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size):
        r"""Residual block."""
        super(ConvBlock, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, c_in, t, f)

        Returns:
            output: (b, c_out, t, f)
        """
        h = self.conv1(F.leaky_relu_(self.bn1(x)))
        h = self.conv2(F.leaky_relu_(self.bn2(h)))

        if self.is_shortcut:
            return self.shortcut(x) + h
        else:
            return x + h


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(EncoderBlock, self).__init__()

        self.pool_size = 2

        self.conv_block = ConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (b, c_in, t, f)

        Returns:
            latent: (b, c_out, t, f)
            output: (b, c_out, t/2, f/2)
        """

        latent = self.conv_block(x)  # shape: (b, c_out, t, f)
        output = F.avg_pool2d(latent, kernel_size=self.pool_size)  # shape: (b, c_out, t/2, f/2)
        return output, latent 


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(DecoderBlock, self).__init__()

        stride = 2

        self.upsample = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=stride,
            stride=stride,
            padding=(0, 0),
            bias=False,
        )

        self.conv_block = ConvBlock(in_channels * 2, out_channels, kernel_size)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, c_in, t/2, f/2)

        Returns:
            output: (b, c_out, t, f)
        """

        x = self.upsample(x)  # shape: (b, c_in, t, f)
        x = torch.cat((x, latent), dim=1)  # shape: (b, 2*c_in, t, f)
        x = self.conv_block(x)  # shape: (b, c_out, t, f)
        
        return x
    
if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = UFormerConfig()
    model = UFormer(config)
    checkpoint_path = None
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    audio = torch.randn(1, 2, 10*44100).to(device)  # Example audio input (batch_size=1, channels=2, samples=88200)
    output = model(audio)
    print(output.shape)  # Output shape