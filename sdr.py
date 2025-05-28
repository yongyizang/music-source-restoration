import librosa
import numpy as np


def fast_evaluate(
    references: np.ndarray, 
    estimates: np.ndarray, 
    win: int =1 * 44100, 
    hop: int =1 * 44100
):
    r"""Fast version to calculate SDR of separation result. This function is 
    200 times faster than museval.evaluate(). The error is within 0.001. 

    Args:
        output: (c, l)
        target: (c, l)

    Returns:
        sdr: float
    """

    refs = librosa.util.frame(references, frame_length=win, hop_length=hop)  # (c, t, n)
    ests = librosa.util.frame(estimates, frame_length=win, hop_length=hop)  # (c, t, n)

    segs_num = refs.shape[2]
    sdrs = []

    for n in range(segs_num):
        sdr = fast_sdr(ref=refs[:, :, n], est=ests[:, :, n])
        sdrs.append(sdr)

    return sdrs


def fast_sdr(
    ref: np.ndarray, 
    est: np.ndarray, 
    eps: float = 1e-10
):
    r"""Calcualte SDR.
    """
    # Calculate the scaling factor (alpha)
    alpha = np.sum(est * ref) / np.clip(a=np.sum(ref ** 2), a_min=eps, a_max=None)
    
    # Scale the reference signal
    scaled_ref = alpha * ref
    
    # Calculate the error/noise
    noise = est - scaled_ref
    
    # Calculate SI-SDR
    numerator = np.clip(a=np.mean(scaled_ref ** 2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)
    si_sdr = 10. * np.log10(numerator / denominator)
    
    return si_sdr