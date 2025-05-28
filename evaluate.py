from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import scipy.io.wavfile as wavfile

from utils import parse_yaml


def evaluate(args) -> None:
    r"""Evaluate a music source restoration system."""

    # Arguments
    config_path = args.config
    eval_dir = args.eval_dir
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]
    
    print(f"Loading model from config: {config_path}")
    print(f"Evaluating on directory: {eval_dir}")
    
    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=args.checkpoint if args.checkpoint else configs["train"]["resume_ckpt_path"]
    ).to(device)
    model.eval()
    
    # Find all mixture and target pairs
    mixture_files, target_files = find_audio_pairs(eval_dir)
    
    if len(mixture_files) == 0:
        print("No mixture/target pairs found in the directory.")
        return
    
    print(f"Found {len(mixture_files)} mixture/target pairs for evaluation.")
    
    # Initialize metrics storage
    sdr_scores = []
    ssim_scores = []
    
    sdr_scores_mixture = []
    ssim_scores_mixture = []
    
    # Process each pair
    for i, (mixture_path, target_path) in enumerate(zip(mixture_files, target_files)):
        print(f"\nProcessing pair {i+1}/{len(mixture_files)}: {os.path.basename(mixture_path)}")
        
        # Load audio files
        mixture, sr = librosa.load(mixture_path, sr=configs["sample_rate"], mono=False)
        target, _ = librosa.load(target_path, sr=configs["sample_rate"], mono=False)
        
        # Ensure audio is stereo
        if mixture.ndim == 1:
            mixture = np.stack([mixture, mixture])
        if target.ndim == 1:
            target = np.stack([target, target])
        
        # Ensure 10-second length (as specified)
        target_length = int(10 * configs["sample_rate"])
        
        # Pad or crop mixture
        if mixture.shape[1] < target_length:
            mixture = librosa.util.fix_length(mixture, target_length, axis=1)
        else:
            mixture = mixture[:, :target_length]
        
        # Pad or crop target
        if target.shape[1] < target_length:
            target = librosa.util.fix_length(target, target_length, axis=1)
        else:
            target = target[:, :target_length]
        
        # Convert to tensor
        mixture_tensor = torch.from_numpy(mixture).float().to(device)
        
        # No need for clip duration parameters since we're doing direct inference
        
        # Process with model - direct inference
        prediction = process_audio(model, mixture_tensor)
        
        # Ensure prediction has same length as target
        if prediction.shape[1] != target.shape[1]:
            prediction = librosa.util.fix_length(prediction, target.shape[1], axis=1)
        
        # Calculate SI-SDR
        sdr = calculate_si_sdr(target, prediction)
        sdr_scores.append(sdr)
        
        sdr_mixture = calculate_si_sdr(mixture, prediction)
        sdr_scores_mixture.append(sdr_mixture)
        
        # Calculate SSIM on mel spectrograms
        ssim_score = calculate_spectral_ssim(target, prediction, configs["sample_rate"])
        ssim_scores.append(ssim_score)
        
        ssim_mixture = calculate_spectral_ssim(mixture, prediction, configs["sample_rate"])
        ssim_scores_mixture.append(ssim_mixture)
        
        # print(f"Prediction SI-SDR: {sdr:.4f} dB, SSIM: {ssim_score:.4f}; Mixture SI-SDR: {sdr_mixture:.4f} dB, SSIM: {ssim_mixture:.4f}")
        
        # Optional: Save the restored audio
        if args.save_outputs:
            output_dir = Path(args.output_dir) if args.output_dir else Path(eval_dir) / "restored"
            output_dir.mkdir(exist_ok=True, parents=True)
            
            # Normalize audio to prevent clipping
            prediction = prediction / np.max(np.abs(prediction)) * 0.95
            
            # Save as WAV file
            output_path = output_dir / f"restored_{os.path.basename(mixture_path)}"
            wavfile.write(output_path, configs["sample_rate"], prediction.T.astype(np.float32))
            print(f"Saved restored audio to {output_path}")
            
    # Calculate mean and confidence intervals
    sdr_mean = np.mean(sdr_scores)
    sdr_ci = 1.96 * np.std(sdr_scores) / np.sqrt(len(sdr_scores))
    
    ssim_mean = np.mean(ssim_scores)
    ssim_ci = 1.96 * np.std(ssim_scores) / np.sqrt(len(ssim_scores))
    
    sdr_mixture_mean = np.mean(sdr_scores_mixture)
    sdr_mixture_ci = 1.96 * np.std(sdr_scores_mixture) / np.sqrt(len(sdr_scores_mixture))
    
    ssim_mixture_mean = np.mean(ssim_scores_mixture)
    ssim_mixture_ci = 1.96 * np.std(ssim_scores_mixture) / np.sqrt(len(ssim_scores_mixture))
    
    print("\n" + "="*50)
    print("Evaluation Results - Prediction")
    print(f"SI-SDR: {sdr_mean:.2f} ± {sdr_ci:.2f} dB")
    print(f"SSIM: {ssim_mean:.4f} ± {ssim_ci:.4f}")
    print("\n" + "="*50)
    print("Evaluation Results - Mixture")
    print(f"SI-SDR: {sdr_mixture_mean:.2f} ± {sdr_mixture_ci:.2f} dB")
    print(f"SSIM: {ssim_mixture_mean:.4f} ± {ssim_mixture_ci:.4f}")
    print("="*50)
    
    
    # Save results to file
    if args.save_results:
        results_file = Path(eval_dir) / "evaluation_results.txt"
        with open(results_file, "w") as f:
            f.write("Evaluation Results:\n")
            f.write(f"Model config: {config_path}\n")
            f.write(f"Checkpoint: {args.checkpoint if args.checkpoint else configs['train']['resume_ckpt_path']}\n")
            f.write(f"Number of samples: {len(sdr_scores)}\n")
            f.write(f"SI-SDR: {sdr_mean:.2f} ± {sdr_ci:.2f} dB\n")
            f.write(f"SSIM: {ssim_mean:.4f} ± {ssim_ci:.4f}\n")
            
            # Individual file results
            f.write("\nIndividual file results:\n")
            for i, (mix, tgt) in enumerate(zip(mixture_files, target_files)):
                f.write(f"{os.path.basename(mix)}: SI-SDR={sdr_scores[i]:.2f} dB, SSIM={ssim_scores[i]:.4f}\n")
                
        print(f"Results saved to {results_file}")


def calculate_si_sdr(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    
    SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    where s_target = (prediction·target)target/||target||^2
    and e_noise = prediction - s_target
    
    Args:
        target: Target/reference audio of shape (channels, samples)
        prediction: Predicted/estimated audio of shape (channels, samples)
        
    Returns:
        SI-SDR value in dB
    """
    # Ensure arrays have the same shape
    assert target.shape == prediction.shape, "Target and prediction must have the same shape"
    
    # Flatten arrays if multichannel
    if target.ndim > 1:
        target = target.reshape(-1)
        prediction = prediction.reshape(-1)
    
    # Calculate the scaling factor
    scaling = np.dot(prediction, target) / (np.dot(target, target) + 1e-8)
    
    # Calculate the scaled target
    scaled_target = scaling * target
    
    # Calculate the noise component
    noise = prediction - scaled_target
    
    # Calculate SI-SDR
    si_sdr = 10 * np.log10(np.dot(scaled_target, scaled_target) / np.dot(noise, noise) + 1e-8)
    
    return si_sdr


def find_audio_pairs(directory: str) -> Tuple[List[str], List[str]]:
    """Find all mixture/target pairs in the directory."""
    directory = Path(directory)
    
    # Find all mixture and target files
    mixture_files = sorted(list(directory.glob("mixture_*.wav")))
    target_files = sorted(list(directory.glob("target_*.wav")))
    
    # Ensure they match
    valid_mixture_files = []
    valid_target_files = []
    
    for mix_file in mixture_files:
        # Extract the index from mixture_{i}.wav
        try:
            index = mix_file.stem.split('_')[1]
            target_file = directory / f"target_{index}.wav"
            
            if target_file.exists():
                valid_mixture_files.append(str(mix_file))
                valid_target_files.append(str(target_file))
        except:
            continue
    
    return valid_mixture_files, valid_target_files

# Normalize to 0-1 range for SSIM calculation with protection against invalid values
def safe_normalize(mel_db):
    min_val = np.min(mel_db)
    max_val = np.max(mel_db)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (flat spectrogram)
    if range_val == 0 or np.isnan(range_val) or np.isinf(range_val):
        return np.ones_like(mel_db) * 0.5  # Return constant value if flat
    
    # Normal case: proceed with normalization
    normalized = (mel_db - min_val) / range_val
    
    # Final safety check to ensure valid values
    normalized = np.clip(normalized, 0, 1)
    normalized = np.nan_to_num(normalized, nan=0.5, posinf=1.0, neginf=0.0)
    
    return normalized


def calculate_spectral_ssim(target: np.ndarray, prediction: np.ndarray, sr: int) -> float:
    """
    Calculate SSIM between mel spectrograms of target and prediction audio.
    
    Args:
        target: Target audio of shape (channels, samples)
        prediction: Predicted audio of shape (channels, samples)
        sr: Sample rate
    
    Returns:
        SSIM score
    """
    # Convert to mono if stereo for spectrogram calculation
    if target.ndim > 1 and target.shape[0] > 1:
        target_mono = np.mean(target, axis=0)
    else:
        target_mono = target[0]
        
    if prediction.ndim > 1 and prediction.shape[0] > 1:
        prediction_mono = np.mean(prediction, axis=0)
    else:
        prediction_mono = prediction[0]
    
    # Calculate mel spectrograms (as per the specs in the instructions)
    n_fft = 4096
    hop_length = 1024
    n_mels = 256
    
    target_mel = librosa.feature.melspectrogram(
        y=target_mono, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length,
        n_mels=n_mels,
        window='hann'
    )
    
    prediction_mel = librosa.feature.melspectrogram(
        y=prediction_mono, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length,
        n_mels=n_mels,
        window='hann'
    )
    
    # Convert to power (amplitude squared)
    target_mel_power = target_mel + 1e-8  # Avoid log(0)
    prediction_mel_power = prediction_mel + 1e-8  # Avoid log(0)
    
    # Convert to log scale for better SSIM comparison
    target_mel_db = librosa.power_to_db(target_mel_power, ref=np.max)
    prediction_mel_db = librosa.power_to_db(prediction_mel_power, ref=np.max)
    
    # Normalize to 0-1 range for SSIM calculation
    target_norm = safe_normalize(target_mel_db)
    prediction_norm = safe_normalize(prediction_mel_db)
    
    # Calculate SSIM
    ssim_score = ssim(target_norm, prediction_norm, data_range=1.0)
    
    return ssim_score


def process_audio(
    model: nn.Module, 
    audio: torch.Tensor
):
    """Simple direct inference on audio without any overlap-add processing.
    
    Args:
        model: nn.Module
        audio: (c, audio_samples) tensor
        
    Returns:
        output: (c, audio_samples) numpy array
    """
    device = next(model.parameters()).device
    audio = audio.to(device)
    
    # Add batch dimension if needed
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)  # (1, c, audio_samples)
    
    # Direct inference
    with torch.no_grad():
        model.eval()
        output = model(audio)
        
    # Remove batch dimension if it was added
    if output.dim() == 3:
        output = output.squeeze(0)  # (c, audio_samples)
        
    output = output.cpu().numpy()
    # nan to 0
    output[np.isnan(output)] = 0
    return output


def get_model(
    configs: dict, 
    ckpt_path: str
) -> nn.Module:
    r"""Initialize model."""

    name = configs["model"]["name"]

    if name == "UNet":
        from music_source_restoration.models.unet import UNet, UNetConfig

        config = UNetConfig(
            n_fft=configs["model"]["n_fft"],
            hop_length=configs["model"]["hop_length"],
        )
        model = UNet(config)
        
    elif name == "UFormer":
        from music_source_restoration.models.uformer import UFormer, UFormerConfig

        config = UFormerConfig(
            sr=configs["sample_rate"],
            n_fft=configs["model"]["n_fft"],
            hop_length=configs["model"]["hop_length"],
            n_layer=configs["model"]["n_layer"],
            n_head=configs["model"]["n_head"],
            n_embd=configs["model"]["n_embd"],
        )
        model = UFormer(config)

    elif name == "BSRoformer":
        import ast
        from music_source_restoration.models.bsroformer import BSRoformer, BSRoformerConfig

        config = BSRoformerConfig(
            sr=configs["sample_rate"],
            n_fft=configs["model"]["n_fft"],
            hop_length=configs["model"]["hop_length"],
            mel_bins=configs["model"]["mel_bins"],
            mel_channels=configs["model"]["mel_channels"],
            patch_size=ast.literal_eval(configs["model"]["patch_size"]),
            n_layer=configs["model"]["n_layer"],
            n_head=configs["model"]["n_head"],
            n_embd=configs["model"]["n_embd"],
        )
        model = BSRoformer(config)        

    else:
        raise ValueError(f"Unknown model name: {name}")    

    if ckpt_path:
        print(f"Loading model checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(ckpt)
    else:
        print("Warning: No checkpoint path provided. Using randomly initialized model.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a music source restoration model")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml file")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory containing mixture/target pairs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (overrides config)")
    parser.add_argument("--save_outputs", action="store_true", help="Save restored audio files")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save restored audio files")
    parser.add_argument("--save_results", action="store_true", help="Save evaluation results to file")
    
    args = parser.parse_args()
    evaluate(args)