from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from discriminator import Discriminator
from torch.utils.tensorboard import SummaryWriter
from losses import L1Loss
from dataset import RawStems
import matplotlib.pyplot as plt
from utils import (LinearWarmUp, calculate_sdr, parse_yaml)

max_norm = 2.0

def train(args) -> None:
    r"""Train a music source separation system."""

    # Arguments
    config_path = args.config
    gan_mode = args.gan
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]
    exp_name = configs["exp_name"]
    save_path = configs["save_path"]
    
    # recursively create directories
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Checkpoints directory
    config_name = Path(config_path).stem
    mode_name = "gan" if gan_mode else "recon"
    ckpts_dir = Path(save_path, exp_name, "checkpoints", filename, f"{config_name}_{mode_name}")
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)
    
    # TensorBoard
    log_dir = Path(save_path, exp_name, "logs", filename, f"{config_name}_{mode_name}")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Datasets
    train_dataset = get_dataset(configs, split="train")

    # Sampler
    train_sampler = get_sampler(configs, train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True
    )
    
    test_dataset = get_dataset(configs, split="test")
    
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True
    )

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    ).to(device)
    
    # Discriminator
    discriminator = None
    optimizer_d = None
    scheduler_d = None
    if gan_mode:
        discriminator = Discriminator().to(device)
        optimizer_d = optim.AdamW(params=discriminator.parameters(), lr=float(configs["train"]["lr_discriminator"]))

    # Generator optimizer
    optimizer_g = optim.AdamW(params=model.parameters(), lr=float(configs["train"]["lr_generator"]))

    # Learning rate schedulers
    warm_up_steps = configs["train"]["warm_up_steps"]
    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler_g = optim.lr_scheduler.LambdaLR(optimizer=optimizer_g, lr_lambda=lr_lambda)
        if gan_mode:
            scheduler_d = optim.lr_scheduler.LambdaLR(optimizer=optimizer_d, lr_lambda=lr_lambda)
    else:
        scheduler_g = None
        scheduler_d = None

    # Train
    global_step = 0
    for step, data in enumerate(tqdm(train_dataloader, desc="Training", total=configs["train"]["training_steps"])):
        global_step += 1

        # ------ 1. Data Preparation ------
        target = data["target"].to(device)
        mixture = data["mixture"].to(device)
        
        # ------ 2. Train Discriminator (Only in GAN mode) ------
        d_loss = torch.tensor(0.0).to(device) 
        if gan_mode and discriminator is not None:
            # Set discriminator to training mode
            discriminator.train()
            
            # Get real and fake samples
            with torch.no_grad():
                model.eval()
                fake = model(mixture)
            
            # Real and fake discriminator outputs
            d_real = discriminator(target)
            d_fake = discriminator(fake.detach())
            
            # Calculate discriminator loss
            d_loss = discriminator_loss(d_fake, d_real)
            
            # Optimize discriminator
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()
        
        # ------ 3. Train Generator ------
        # Set model to training mode
        model.train()
        
        # Forward pass
        output = model(mixture)
        
        # Calculate reconstruction loss
        reconstruction_loss = L1Loss(target, output)
        
        # In GAN mode, add adversarial losses
        g_loss = torch.tensor(0.0).to(device)
        feature_loss = torch.tensor(0.0).to(device)
        if gan_mode and discriminator is not None:
            d_real = discriminator(target)
            d_fake = discriminator(output)
            g_loss, feature_loss = generator_loss(d_fake, d_real)
            combined_loss = reconstruction_loss + 0.001 * g_loss + 0.01 * feature_loss
        else:
            # In Reconstruction mode, use only reconstruction loss
            combined_loss = reconstruction_loss
        
        # Optimize generator
        optimizer_g.zero_grad()
        combined_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer_g.step()
        
        # Learning rate scheduler steps
        if scheduler_g:
            scheduler_g.step()
        if gan_mode and scheduler_d:
            scheduler_d.step()

        # ------ 4. Logging ------
        if step % 10 == 0:
            # Log losses
            writer.add_scalar('Loss/generator', combined_loss.item(), global_step)
            writer.add_scalar('Loss/reconstruction', reconstruction_loss.item(), global_step)
            
            if gan_mode:
                writer.add_scalar('Loss/discriminator', d_loss.item(), global_step)
                writer.add_scalar('Loss/gan', g_loss.item(), global_step)
                writer.add_scalar('Loss/feature', feature_loss.item(), global_step)
            
            if scheduler_g:
                writer.add_scalar('LR/generator', scheduler_g.get_last_lr()[0], global_step)
                if gan_mode and scheduler_d:
                    writer.add_scalar('LR/discriminator', scheduler_d.get_last_lr()[0], global_step)

        # Display losses periodically
        if step % 100 == 0:
            if gan_mode:
                print(f"Step {step}, Generator Loss: {combined_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")
            else:
                print(f"Step {step}, Reconstruction Loss: {combined_loss.item():.4f}")
            
            # Log audio samples
            if step % configs["train"]["log_audio_every_n_steps"] == 0:
                # Log original and generated audio
                mixture_wave = mixture[0].cpu().mean(dim=0)
                target_wave = target[0].cpu().mean(dim=0)
                output_wave = output[0].cpu().mean(dim=0)
                writer.add_audio('Audio/mixture (Mono)', mixture_wave, global_step, sample_rate=configs["sample_rate"])
                writer.add_audio('Audio/original (Mono)', target_wave, global_step, sample_rate=configs["sample_rate"])
                writer.add_audio('Audio/generated (Mono)', output_wave, global_step, sample_rate=configs["sample_rate"])
                
                # Generate and log spectrograms
                log_spectrograms(writer, target_wave, output_wave, configs["sample_rate"], global_step)

        # ------ 5. Evaluation ------
        if step % configs["train"]["test_every_n_steps"] == 0 and step > 0:
            # Set model to evaluation mode
            model.eval()
            
            # Initialize SDR list
            sdr_list = []
            test_loss_list = []
            
            # Evaluate on test dataset
            with torch.no_grad():
                for test_data in tqdm(test_dataloader, desc="Evaluating", total=len(test_dataloader)):
                    test_target = test_data["target"].to(device)
                    test_mixture = test_data["mixture"].to(device)
                    
                    test_prediction = model(test_mixture)
                    
                    test_loss = L1Loss(test_target, test_prediction)
                    test_loss_list.append(test_loss.item())
                    
                    test_sdr = calculate_sdr(test_target.cpu().numpy(), test_prediction.cpu().numpy())
                    sdr_list.append(test_sdr)
            
            # Log SDR
            mean_sdr = np.mean(sdr_list)
            writer.add_scalar('test/SI-SDR', mean_sdr, global_step)
            mean_loss = np.mean(test_loss_list)
            writer.add_scalar('test/loss', mean_loss, global_step)
            
            print(f"Test SI-SDR: {mean_sdr:.4f}, Test Loss: {mean_loss:.4f}")
        
        # ------ 6. Save model ------
        if step % configs["train"]["save_every_n_steps"] == 0:
            # Save generator model
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))
            
            # Save discriminator model (only in GAN mode)
            if gan_mode and discriminator is not None:
                ckpt_path = Path(ckpts_dir, "step={}_discriminator.pth".format(step))
                torch.save(discriminator.state_dict(), ckpt_path)
                print("Save discriminator to {}".format(ckpt_path))

        if step == configs["train"]["training_steps"]:
            break
    
    # Close TensorBoard writer
    writer.close()


def discriminator_loss(d_fake, d_real):
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
            loss_feature += nn.functional.l1_loss(d_fake[i][j], d_real[i][j].detach())
    
    return loss_g, loss_feature


def log_spectrograms(writer, original, generated, sample_rate, step):
    """Generate and log mel-spectrograms to TensorBoard."""
    try:
        import librosa.display
        
        # Convert to numpy for librosa
        original_np = original.detach().numpy()
        generated_np = generated.detach().numpy()
        
        # Create mel spectrograms
        mel_original = librosa.feature.melspectrogram(y=original_np, sr=sample_rate)
        mel_generated = librosa.feature.melspectrogram(y=generated_np, sr=sample_rate)
        
        # Convert to dB
        mel_original_db = librosa.power_to_db(mel_original, ref=np.max)
        mel_generated_db = librosa.power_to_db(mel_generated, ref=np.max)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original spectrogram
        img = librosa.display.specshow(mel_original_db, sr=sample_rate, 
                                      hop_length=512, x_axis='time', 
                                      y_axis='mel', ax=axes[0])
        axes[0].set_title('Original')
        fig.colorbar(img, ax=axes[0], format='%+2.0f dB')
        
        # Plot generated spectrogram
        img = librosa.display.specshow(mel_generated_db, sr=sample_rate, 
                                      hop_length=512, x_axis='time', 
                                      y_axis='mel', ax=axes[1])
        axes[1].set_title('Generated')
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        # Add to TensorBoard
        writer.add_figure('Spectrograms', fig, step)
        plt.close(fig)
    except Exception as e:
        print(f"Error generating spectrograms: {e}")


def get_dataset(
    configs: dict, 
    split: str
) -> Dataset:
    r"""Get datasets."""
    
    if split == "train":
        config_set = configs["train_data"]
    elif split == "test":
        config_set = configs["test_data"]
    else:
        raise ValueError("Invalid split: {}".format(split))

    dataset = RawStems(
            target_stem=config_set["target_stem"],
            root_directory=config_set["root_directory"],
            file_list=config_set["file_list"],
            sr=configs["sample_rate"],
            clip_duration=configs["clip_duration"],
            apply_augmentation=config_set["apply_augmentation"],
        )
    
    return dataset


def get_sampler(configs: dict, dataset: Dataset) -> Iterable:
    r"""Get sampler."""

    from dataset import InfiniteSampler
    return InfiniteSampler(dataset)


def get_model(
    configs: dict, 
    ckpt_path: str
) -> nn.Module:
    r"""Initialize model."""
    from model import UFormer, UFormerConfig

    config = UFormerConfig(
        sr=configs["sample_rate"],
        n_fft=configs["model"]["n_fft"],
        hop_length=configs["model"]["hop_length"],
        n_layer=configs["model"]["n_layer"],
        n_head=configs["model"]["n_head"],
        n_embd=configs["model"]["n_embd"],
    )
    model = UFormer(config)

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--gan", action="store_true", help="Enable GAN training mode")
    args = parser.parse_args()

    train(args)