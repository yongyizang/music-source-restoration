from pathlib import Path
import random
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import List, Optional, Dict, Union, Tuple, Any
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from augment import StemAugmentation, MixtureAugmentation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
AUDIO_EXTENSIONS = ['.flac', '.mp3', '.wav']
DEFAULT_GAIN_RANGE = (0.5, 1.0)

def calculate_rms(audio: np.ndarray) -> float:
    """Calculate the RMS of the audio signal."""
    return np.sqrt(np.mean(audio**2))


def contains_audio_signal(audio: np.ndarray, rms_threshold: float = 0.01) -> bool:
    """
    Check if audio array contains actual signal using RMS value.
    
    Args:
        audio: Audio array of shape (channels, samples)
        rms_threshold: Minimum RMS value to consider as containing audio
        
    Returns:
        Boolean indicating if audio contains signal above threshold
    """
    if audio is None:
        return False
    
    # Calculate RMS
    rms = calculate_rms(audio)
    return rms > rms_threshold

def fix_length(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Fix the length of target to match the source."""
    target_length = target.shape[-1]
    source_length = source.shape[-1]

    if target_length < source_length:
        target = np.pad(target, ((0, 0), (0, source_length - target_length)), mode='constant')
    elif target_length > source_length:
        target = target[:, :source_length]

    return target

def fix_length_to_duration(target: np.ndarray, duration: float, sr: int) -> np.ndarray:
    """Fix the length of target to match the duration."""
    target_length = target.shape[-1]
    target_duration = target_length / sr

    if target_duration < duration:
        target = np.pad(target, ((0, 0), (0, int((duration - target_duration) * sr))), mode='constant')
    elif target_duration > duration:
        target = target[:, :int(duration * sr)]

    return target


def get_audio_duration(file_path: Path) -> float:
    """Get the duration of an audio file."""
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception as e:
        logger.error(f"Error getting duration for {file_path}: {e}")
        return 0.0


def load_audio(
    file_path: Path, 
    offset: float, 
    duration: float, 
    sr: int
) -> np.ndarray:
    """Load audio file at the specified offset."""
    try:
        audio, _ = librosa.load(
            file_path, 
            sr=sr, 
            offset=offset, 
            duration=duration,
            mono=False
        )
        
        # Make sure audio is 2D: (channels, samples)
        if len(audio.shape) == 1:
            audio = audio.reshape(1, -1)
                
        # If samples is 0, then just return silence
        if audio.shape[1] == 0:
            return np.zeros((2, int(sr * duration)))
        
        # Ensure stereo
        if audio.shape[0] == 1:
            audio = np.vstack([audio, audio])
        
        return audio
    except Exception as e:
        logger.error(f"Error loading {file_path} at offset {offset}: {e}")
        return np.zeros((2, int(sr * duration)))

def mix_to_target_snr(target: np.ndarray, noise: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, float, float]:
    """
    Mix target and noise signals to achieve a target SNR.
    
    Args:
        target: Target signal (clean) - shape (channels, samples)
        noise: Noise signal (interference) - shape (channels, samples)
        target_snr_db: Desired SNR in dB
    
    Returns:
        Tuple containing:
        - Mixed signal with target SNR
        - Scale factor applied to target
        - Scale factor applied to noise
    """    
    # Calculate power of both signals
    target_power = np.mean(target ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Avoid division by zero
    if noise_power < 1e-8:
        # If noise is essentially zero, just return target
        return target.copy(), 1.0, 0.0
    
    if target_power < 1e-8:
        # If target is essentially zero, return a very quiet noise
        return noise * 0.001, 0.0, 0.001
    
    # Convert target SNR from dB to linear scale
    target_snr_linear = 10 ** (target_snr_db / 10)
    
    # Calculate the scaling factor for noise to achieve target SNR
    # SNR = target_power / (noise_power * scale^2)
    # scale = sqrt(target_power / (noise_power * SNR))
    noise_scale = np.sqrt(target_power / (noise_power * target_snr_linear))
    
    # Scale the noise
    scaled_noise = noise * noise_scale
    
    # Mix signals
    mixture = target + scaled_noise
    
    # Normalize to prevent clipping (optional but recommended)
    max_amplitude = np.max(np.abs(mixture))
    if max_amplitude > 1.0:
        normalization_factor = 0.95 / max_amplitude  # Leave some headroom
        mixture = mixture * normalization_factor
        target_scale = normalization_factor
        noise_scale = noise_scale * normalization_factor
    else:
        target_scale = 1.0
    
    return mixture, target_scale, noise_scale

class RawStems(Dataset):
    def __init__(
        self,
        target_stem: str,
        root_directory: Union[str, Path],
        file_list: Optional[Union[str, Path]] = None,
        sr: int = 44100,
        clip_duration: float = 3.0,
        snr_range: Tuple[float, float] = (0.0, 10.0),
        apply_augmentation: bool = True,
    ) -> None:
        """
        Initialize the dataset.
        
        Args:
            folders: List of folder paths containing audio files
            sr: Sample rate
            clip_duration: Duration of audio clips in seconds
            snr_range: Range of Signal-to-Noise Ratio in dB (min, max)
            apply_augmentation: Whether to apply audio augmentations
        """
        
        self.folders = []
        with open(file_list, 'r') as f:
            for line in f:
                folder = Path(line.strip())
                folder = Path(root_directory) / folder
                # Check if the folder exists
                if folder.exists():
                    self.folders.append(folder)
                else:
                    logger.warning(f"Folder does not exist: {folder}")
                
        self.sr = sr
        # diverse deg. simulation may cause length to change. + 1 sec and - 1 sec in the end.
        self.clip_duration = clip_duration + 1.
        self.snr_range = snr_range
        self.apply_augmentation = apply_augmentation
        
        target_stem = target_stem.split("_")
        self.target_stem_1 = target_stem[0].strip()
        if len(target_stem) > 1:
            self.target_stem_2 = target_stem[1].strip()
        else:
            self.target_stem_2 = None
        
        # Find all audio files in the specified folders
        self.audio_files = self._find_audio_files()
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in the specified folders")
        
        # Initialize augmentation classes
        self.stem_augmentation = StemAugmentation()
        self.mixture_augmentation = MixtureAugmentation()
    
    def _find_audio_files(self) -> Dict[str, List[Path]]:
        """Find all audio files in the specified folders."""
        audio_files = []
        audio_dict = {
            "target_stems": [],
            "others": []
        }
        
        for folder in tqdm(self.folders, desc="Indexing audio files"):
            if not folder.exists():
                logger.warning(f"Folder does not exist: {folder}")
                continue
            
            # Check for target stem files
            target_folder = folder / self.target_stem_1
            
            # If target_stem_2 exists, check in the nested folder
            if self.target_stem_2:
                target_folder = target_folder / self.target_stem_2
            
            # Find audio files in target folder
            if target_folder.exists():
                for file_path in target_folder.rglob('*'):
                    if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                        audio_dict["target_stems"].append(file_path)
            
            # Find all other audio files (not in target stem folders)
            for file_path in folder.rglob('*'):
                if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    # Check if this file is not in the target stem folder(s)
                    relative_path = file_path.relative_to(folder)
                    parts = relative_path.parts
                    
                    # Skip if it's in the target stem path
                    if len(parts) > 0 and parts[0] == self.target_stem_1:
                        if self.target_stem_2 is None:
                            continue  # Skip target_stem_1 files
                        elif len(parts) > 1 and parts[1] == self.target_stem_2:
                            continue  # Skip target_stem_1/target_stem_2 files
                    
                    audio_dict["others"].append(file_path)
            
            audio_files.append(audio_dict)
        
        return audio_files
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a random segment from audio files with random stem mixing."""
        # Set a maximum retry limit to avoid infinite loops
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Select a random offset that will be used for all files
                duration = self.clip_duration
                
                current_dict = self.audio_files[index]
                
                # Get all available target stems
                target_stems_available = current_dict["target_stems"]
                other_stems_available = current_dict["others"]
                
                if len(target_stems_available) == 0:
                    raise ValueError("No target stems available")
                
                if len(other_stems_available) == 0:
                    raise ValueError("No other stems available")
                
                # Randomly decide how many target stems to mix (1 to all available)
                num_target_stems = random.randint(1, min(len(target_stems_available), 10))  # Cap at 10 for memory
                selected_target_files = random.sample(target_stems_available, num_target_stems)
                
                # For others, ensure we select at least one
                num_other_stems = random.randint(1, min(len(other_stems_available), 20))  # Cap at 20
                selected_other_files = random.sample(other_stems_available, num_other_stems)
                
                # Find the minimum duration among selected files to ensure valid offset
                min_duration = float('inf')
                for file_path in selected_target_files + selected_other_files:
                    file_duration = get_audio_duration(file_path)
                    min_duration = min(min_duration, file_duration)
                
                # Get a random offset based on minimum duration
                max_offset = max(0, min_duration - self.clip_duration)
                offset = random.uniform(0, max_offset) if max_offset > 0 else 0
                
                # Load and mix target stems
                target_mix = None
                
                for file_path in selected_target_files:
                    audio = load_audio(file_path, offset, self.clip_duration, self.sr)
                    
                    if target_mix is None:
                        target_mix = audio.copy()
                    else:
                        target_mix += fix_length(audio, target_mix)
                
                # Average the mixed target
                if num_target_stems > 1:
                    target_mix = target_mix / num_target_stems
                
                # Load and mix other stems
                other_mix = None

                for file_path in selected_other_files:
                    audio = load_audio(file_path, offset, self.clip_duration, self.sr)
                    
                    if other_mix is None:
                        other_mix = audio.copy()
                    else:
                        other_mix += fix_length(audio, other_mix)
                
                # Average the mixed others
                if num_other_stems > 1:
                    other_mix = other_mix / num_other_stems
                
                # Check if both target and other mixes contain actual audio
                if not contains_audio_signal(target_mix) or not contains_audio_signal(other_mix):
                    # If either doesn't contain audio, retry with different selections
                    retry_count += 1
                    continue
                    
                # Store clean version
                target_clean = target_mix.copy()

                if self.apply_augmentation:
                    target_augmented = self.stem_augmentation.apply(target_mix, self.sr)
                else:
                    target_augmented = target_mix
                    
                target_augmented = fix_length_to_duration(target_augmented, self.clip_duration, self.sr)
                other_mix = fix_length_to_duration(other_mix, self.clip_duration, self.sr)
                
                # add gaussian noise
                if np.random.rand() < 0.5 and other_mix is not None:
                    noise = np.random.normal(0, 0.01, other_mix.shape)
                    other_mix += noise * np.random.uniform(0.1, 0.5)
                
                # Create the final mixture
                # Mix target and other stems to achieve target SNR
                mixture, target_scale, noise_scale = mix_to_target_snr(
                    target_augmented, other_mix, random.uniform(self.snr_range[0], self.snr_range[1])
                )
                target_clean = target_clean * target_scale

                if self.apply_augmentation:
                    mixture_augmented = self.mixture_augmentation.apply(mixture, self.sr)
                else:
                    mixture_augmented = mixture.copy()
                    
                mixture_augmented = fix_length_to_duration(mixture_augmented, self.clip_duration, self.sr)
                
                # Normalize
                max_scale = np.max(np.abs(mixture_augmented)) + 1e-7
                mixture_augmented = mixture_augmented / max_scale
                target_augmented = target_augmented / max_scale
                target_clean = target_clean / max_scale
                
                rescale = np.random.uniform(*DEFAULT_GAIN_RANGE)
                
                target_clean = fix_length_to_duration(target_clean, self.clip_duration, self.sr)
                mixture_augmented = fix_length_to_duration(mixture_augmented, self.clip_duration, self.sr)
                
                # fix potential nan
                target_clean = np.nan_to_num(target_clean)
                mixture_augmented = np.nan_to_num(mixture_augmented)
                
                return {
                    "target": target_clean[:int((self.clip_duration-1)*self.sr)] * rescale,
                    "mixture": mixture_augmented[:int((self.clip_duration-1)*self.sr)] * rescale
                }
                
            except Exception as e:
                logger.error(f"Error in __getitem__ at index {index}: {e}")
                retry_count += 1
        
        # If we've exhausted our retries, try a different index
        return self.__getitem__(random.randint(0, len(self.audio_files) - 1))

    def __len__(self) -> int:
        return len(self.audio_files)


class InfiniteSampler(Sampler):
    """Sampler that yields infinite random samples without replacement."""
    
    def __init__(self, dataset: Dataset) -> None:
        self.dataset_size = len(dataset)
        self.indexes = list(range(self.dataset_size))
        self.reset()
    
    def reset(self) -> None:
        """Reshuffle indexes and reset pointer."""
        random.shuffle(self.indexes)
        self.pointer = 0
        
    def __iter__(self):
        while True:
            if self.pointer >= self.dataset_size:
                self.reset()
                
            index = self.indexes[self.pointer]
            self.pointer += 1
            yield index

if __name__ == "__main__":    
    from tqdm import tqdm
    target_stems = ["Voc", "Bass", "Gtr", "Kbs", "Rhy", "Orch", "Synth", "Gtr_AG", "Gtr_EG"]
    
    for target_stem in tqdm(target_stems):
        output_dir = "/root/autodl-tmp/msr_test_set/" + target_stem + "/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created: {output_dir}")
    
        # Create dataset
        dataset = RawStems(
            target_stem=target_stem,
            root_directory="/root/autodl-tmp/",
            file_list="/root/music_source_restoration/configs/data_split/" + target_stem + "_test.txt",
            sr=44100,
            clip_duration=10.0,
            apply_augmentation=True
        )
        
        # Create a sampler
        sampler = InfiniteSampler(dataset)
        
        # Sample for 1000 iterations
        for i in tqdm(range(1000), desc="Sampling"):
            index = next(iter(sampler))
            sample = dataset[index]
            sample["mixture"] = sample["mixture"][:, :441000]
            sample["target"] = sample["target"][:, :441000]
            
            # Save the mixture and target
            mixture_path = Path(output_dir) / f"mixture_{i}.wav"
            target_path = Path(output_dir) / f"target_{i}.wav"
            
            sf.write(mixture_path, sample["mixture"].T, dataset.sr)
            sf.write(target_path, sample["target"].T, dataset.sr)