import torch
import torch.nn.functional as F
from typing import List
from music_source_restoration.utils import batch_mel

def L1Loss(x,y):
    loss = F.l1_loss(x, y)
    return loss
    
def mel_spectrogram_loss(
    x: torch.Tensor, y: torch.Tensor, 
    sample_rate: int = 16000,
    n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320], 
    window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
    clamp_eps: float = 1e-5, 
    mag_weight: float = 1.0, 
    log_weight: float = 1.0, 
    pow: float = 2.0, 
    mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0], 
    mel_fmax: List[float] = None,):

    if mel_fmax is None:
        mel_fmax = [sample_rate / 2] * len(n_mels)
    
    loss = 0.0
    for n_mel, win_len, fmin, fmax in zip(n_mels, window_lengths, mel_fmin, mel_fmax):
        
        x_mel = batch_mel(x,sample_rate=sample_rate,window_length=win_len,n_mels=n_mel,f_min=fmin,f_max=fmax)
        y_mel = batch_mel(y,sample_rate=sample_rate,window_length=win_len,n_mels=n_mel,f_min=fmin,f_max=fmax)

        log_loss = F.l1_loss(
            x_mel.clamp(min=clamp_eps).pow(pow).log10(),
            y_mel.clamp(min=clamp_eps).pow(pow).log10(),
        )

        mag_loss = F.l1_loss(
            x_mel, y_mel
        )

        loss += log_weight * log_loss + mag_weight * mag_loss
    
    return loss / len(n_mels)
    
def discriminator_loss(d_fake, d_real):

    d_fake = d_fake
    
    loss_d = 0
    for x_fake, x_real in zip(d_fake, d_real):
        loss_d += torch.mean(x_fake[-1] ** 2)
        loss_d += torch.mean((1 - x_real[-1]) ** 2)
    
    return loss_d

def generator_loss(d_fake, d_real):
    
    loss_g = 0
    for x_fake in d_fake:
        loss_g += torch.mean((1 - x_fake[-1]) ** 2)
    
    loss_feature = 0
    for i in range(len(d_fake)):
        for j in range(len(d_fake[i]) - 1):
            loss_feature += L1Loss(d_fake[i][j], d_real[i][j].detach())
    
    return loss_g, loss_feature

def sisdr_loss(x: torch.Tensor, y: torch.Tensor, scaling: bool = True, zero_mean: bool = True, clip_min: float = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio (SISDR) between two audio signals.

    Args:
        x (torch.Tensor): Reference audio signal (B x T).
        y (torch.Tensor): Estimated audio signal (B x T).
        scaling (bool): Whether to apply scale-invariant normalization. Defaults to True.
        zero_mean (bool): Whether to zero-mean the signals before computation. Defaults to True.
        clip_min (float): Minimum possible loss value to clip. Defaults to None.
        eps (float): Small constant to avoid division by zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: The computed SISDR loss (negative SISDR in dB).
    """
    # Ensure inputs have the same shape
    assert x.shape == y.shape, "Input tensors x and y must have the same shape"

    # Reshape tensors for batch processing
    references = x.reshape(x.size(0), 1, -1).permute(0, 2, 1)  # B x T x 1
    estimates = y.reshape(y.size(0), 1, -1).permute(0, 2, 1)   # B x T x 1

    # Zero-mean the signals if specified
    if zero_mean:
        references = references - references.mean(dim=1, keepdim=True)
        estimates = estimates - estimates.mean(dim=1, keepdim=True)

    # Projection of estimates onto references
    references_projection = (references**2).sum(dim=1, keepdim=True) + eps
    references_on_estimates = (references * estimates).sum(dim=1, keepdim=True) + eps

    # Scale normalization
    if scaling:
        scale = references_on_estimates / references_projection
    else:
        scale = 1

    # True and residual components
    e_true = scale * references
    e_res = estimates - e_true

    # Signal and noise energies
    signal = (e_true**2).sum(dim=1)
    noise = (e_res**2).sum(dim=1)

    # Compute SISDR
    sisdr = 10 * torch.log10(signal / (noise + eps) + eps)

    # Negative SISDR for loss
    sisdr_loss = sisdr

    # Apply minimum clipping if specified
    if clip_min is not None:
        sisdr_loss = torch.clamp(sisdr_loss, min=clip_min)

    # Return mean SISDR loss across the batch
    return sisdr_loss.mean()
