from __future__ import annotations

from collections import OrderedDict
from typing import Literal

import librosa
import numpy as np
import torch
import yaml

import torch
import torchaudio
import torchaudio.transforms as T

from music_source_restoration.sdr import fast_evaluate


def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


class LinearWarmUp:
    r"""Linear learning rate warm up scheduler.
    """
    def __init__(self, warm_up_steps: int) -> None:
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        if step <= self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            return 1.


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay=0.999) -> None:

    # Moving average of parameters
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    # Moving average of buffers. Patch for BN, etc
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())

    for name, buffer in model_buffers.items():
        if buffer.dtype in [torch.long]:
            continue
        ema_buffers[name].mul_(decay).add_(buffer.data, alpha=1 - decay)


def requires_grad(model: nn.Module, flag=True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def calculate_sdr(
    output: np.ndarray, 
    target: np.ndarray, 
) -> float:
    r"""Calculate the SDR of separation result.

    Args:
        output: (c, l)
        target: (c, l)

    Returns:
        sdr: float
    """
    sdrs = fast_evaluate(
        references=target,  # shape: (c, l)
        estimates=output  # shape: (c, l)
    )

    sdr = np.nanmedian(sdrs)

    return sdr

def audio_to_mel(audio, sample_rate):
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    ).to(audio.device)
    mel = mel_spectrogram(audio)
    # 将Power Spectrogram转为DB单位
    mel_db = T.AmplitudeToDB()(mel)
    return mel_db

def batch_mel(audio_data, sample_rate=44100, window_length=None, n_mels=128, f_min=0.0, f_max=None):

    window = torch.hann_window(window_length).to(audio_data.device)
    batch_size, num_channels, _ = audio_data.shape

    # Compute STFT
    stft_data = torch.stft(
        audio_data.reshape(-1, audio_data.size(-1)),  # Flatten batch and channel
        n_fft=window_length,
        hop_length=window_length // 4,
        win_length=window_length,
        window=window,
        normalized=True,
        onesided=True,
        return_complex=True,
        center=False
    )

    # Convert STFT to power spectrogram
    power_spec = stft_data.abs() ** 2

    # Create Mel filter
    mel_scale = torchaudio.transforms.MelScale(n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max, n_stft=window_length//2 + 1,norm='slaney').to(audio_data.device)
    
    # Apply Mel filter
    mel_spec = mel_scale(power_spec)

    # Reshape back to (B, C, F, T) format
    _, n_mels, time_frames = mel_spec.shape
    mel_spec = mel_spec.view(batch_size, num_channels, n_mels, time_frames)
    
    return mel_spec