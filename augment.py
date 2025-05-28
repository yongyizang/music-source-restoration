import numpy as np
from eq_utils import apply_random_eq
from pedalboard import Pedalboard, Resample, Compressor, Distortion, Reverb, Limiter, MP3Compressor

def fix_length_to_duration(target: np.ndarray, duration: float) -> np.ndarray:
    """Fix the length of target to match the duration."""
    target_duration = target.shape[-1]

    if target_duration < duration:
        target = np.pad(target, ((0, 0), (0, int(duration - target_duration))), mode='constant')
    elif target_duration > duration:
        target = target[:, :int(duration)]

    return target

def calculate_rms(audio: np.ndarray) -> float:
    """Calculate the RMS of the audio signal."""
    return np.sqrt(np.mean(audio**2))

class StemAugmentation:
    """Augmentation class for individual stems/targets that preserves original length."""
    
    def __init__(self):
        pass
    
    def apply(self, audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        # If all zero, return early
        if np.max(np.abs(audio)) == 0:
            return audio
        
        # Store original length
        original_length = audio.shape[-1]
        original_rms = calculate_rms(audio)
        if original_rms == 0:
            return audio
        
        # Normalize the audio
        normalize_scale = np.max(np.abs(audio)) + 1e-6
        audio = audio / normalize_scale
        
        do_eq, do_resample, do_compressor, do_distortion, do_reverb = np.random.randint(0, 2, 5)  # 5 random choices
        
        if do_eq:
            audio = apply_random_eq(audio, sample_rate)  # Assuming this preserves length
        
        # Start building a pedalboard
        board = Pedalboard()
        
        if do_resample:
            board.append(Resample(target_sample_rate=np.random.randint(8000, 32000)))
        
        # Add other effects
        if do_compressor:
            board.append(Compressor(
                threshold_db=np.random.uniform(-20, 0),
                ratio=np.random.uniform(1.5, 10.0),
                attack_ms=np.random.uniform(1, 10),
                release_ms=np.random.uniform(50, 200)
            ))
        
        if do_distortion:
            board.append(Distortion(drive_db=np.random.uniform(0, 5)))
            
        if do_reverb:
            board.append(Reverb(
                room_size=np.random.uniform(0.1, 1.0),
                damping=np.random.uniform(0.1, 1.0),
                wet_level=np.random.uniform(0.1, 0.5),
                width=np.random.uniform(0.1, 1.0)
            ))
        
        # Apply effects if any are added
        if len(board) > 0:
            audio = board(audio, sample_rate=sample_rate)
        
        # Ensure exact original length
        audio = fix_length_to_duration(audio, original_length)
        
        new_rms = calculate_rms(audio)
        
        return audio * (original_rms / new_rms)  # Rescale the audio back to original RMS


class MixtureAugmentation:
    """Augmentation class for mixed audio that preserves original length."""
    
    def __init__(self):
        pass
    
    def apply(self, audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
        # If all zero, return early
        if np.max(np.abs(audio)) == 0:
            return audio
        
        # Store original length
        original_length = audio.shape[-1]
        original_rms = calculate_rms(audio)
        if original_rms == 0:
            return audio
        
        normalize_scale = np.max(np.abs(audio)) + 1e-6
        audio = audio / normalize_scale
        
        do_limiter, do_resample, do_codec = np.random.randint(0, 2, 3)  # 2 random choices
        
        board = Pedalboard()
        if do_limiter:
            board.append(Limiter(
                threshold_db=np.random.uniform(-10, 0),
                release_ms=np.random.uniform(50, 200)
            ))
            
        if do_resample:
            board.append(Resample(target_sample_rate=np.random.randint(8000, 32000)))
        
        if do_codec:
            board.append(MP3Compressor(vbr_quality=np.random.uniform(1.0, 9.0)))
            
        # Apply effects if any are added
        if len(board) > 0:
            audio = board(audio, sample_rate=sample_rate)
            
        # Ensure exact original length
        audio = fix_length_to_duration(audio, original_length)
        new_rms = calculate_rms(audio)
        
        return audio * (original_rms / new_rms)  # Rescale the audio back to original RMS