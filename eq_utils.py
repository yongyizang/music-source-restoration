import numpy as np
import scipy.signal as signal

def apply_random_eq(audio_buffer, sr):
    """
    Apply random EQ to an audio buffer.
    
    Args:
        audio_buffer: numpy array of shape [n_channels, length]
        sr: sample rate in Hz
        
    Returns:
        numpy array of shape [n_channels, length] with EQ applied
    """
    # Get the shape of the input
    n_channels, length = audio_buffer.shape
    
    # Create output buffer
    output_buffer = audio_buffer.copy()
    
    # Randomly choose EQ type
    eq_types = ['parametric', 'graphic']
    eq_type = np.random.choice(eq_types)
    
    if eq_type == 'graphic':
        # Apply 16-band graphic EQ
        output_buffer = apply_graphic_eq(output_buffer, sr)
    else:
        # Apply parametric EQ
        output_buffer = apply_parametric_eq(output_buffer, sr)
    
    return output_buffer

def apply_graphic_eq(audio_buffer, sr):
    """Apply 16-band graphic EQ with random gains"""
    n_channels, length = audio_buffer.shape
    output_buffer = audio_buffer.copy()
    
    # Standard 16-band graphic EQ frequencies
    frequencies = [25, 40, 63, 100, 160, 250, 400, 630, 
                  1000, 1600, 2500, 4000, 6300, 10000, 16000, 20000]
    
    # Generate random gains for each band (-12 to +12 dB)
    gains_db = np.random.uniform(-6, 6, 16)
    
    # Apply each band filter
    for i, (freq, gain_db) in enumerate(zip(frequencies, gains_db)):
        # Skip if frequency is above Nyquist
        if freq > sr / 2:
            continue
            
        # Calculate Q factor for graphic EQ
        if i == 0:
            # First band - use lowshelf
            q = 0.7
            sos = design_parametric_filter('lowshelf', freq, q, gain_db, sr)
        elif i == len(frequencies) - 1 or frequencies[i + 1] > sr / 2:
            # Last band - use highshelf
            q = 0.7
            sos = design_parametric_filter('highshelf', freq, q, gain_db, sr)
        else:
            # Middle bands - use peak filters
            lower_freq = frequencies[i - 1] if i > 0 else frequencies[i] / 1.6
            upper_freq = frequencies[i + 1] if i < len(frequencies) - 1 else frequencies[i] * 1.6
            
            # Q for graphic EQ bands
            q = freq / (upper_freq - lower_freq) * 2.5
            sos = design_parametric_filter('peak', freq, q, gain_db, sr)
        
        # Apply filter to each channel
        for ch in range(n_channels):
            output_buffer[ch] = signal.sosfilt(sos, output_buffer[ch])
    
    return output_buffer

def apply_parametric_eq(audio_buffer, sr):
    """Apply parametric EQ with random parameters"""
    n_channels, length = audio_buffer.shape
    output_buffer = audio_buffer.copy()
    
    # Randomly choose number of EQ bands (1-5)
    num_bands = np.random.randint(1, 6)
    
    # Apply multiple EQ bands
    for _ in range(num_bands):
        # Randomly choose filter type
        filter_types = ['lowpass', 'highpass', 'bandpass', 'bandstop', 'peak', 'lowshelf', 'highshelf']
        filter_type = np.random.choice(filter_types)
        
        # Set frequency based on filter type
        if filter_type == 'lowpass':
            frequency = np.random.uniform(200, 8000)
        elif filter_type == 'highpass':
            frequency = np.random.uniform(20, 2000)
        elif filter_type in ['bandpass', 'bandstop', 'peak']:
            frequency = np.random.uniform(100, 8000)
        elif filter_type == 'lowshelf':
            frequency = np.random.uniform(50, 1000)
        elif filter_type == 'highshelf':
            frequency = np.random.uniform(1000, 10000)
        
        # Random Q factor (affects bandwidth)
        Q = np.random.uniform(0.5, 10)
        
        # Random gain for shelving and peak filters (in dB)
        gain_db = np.random.uniform(-6, 6)
        
        # Design the filter based on type
        if filter_type in ['lowpass', 'highpass', 'bandpass', 'bandstop']:
            sos = design_basic_filter(filter_type, frequency, Q, sr)
        elif filter_type in ['peak', 'lowshelf', 'highshelf']:
            sos = design_parametric_filter(filter_type, frequency, Q, gain_db, sr)
        
        # Apply the filter to each channel
        for ch in range(n_channels):
            output_buffer[ch] = signal.sosfilt(sos, output_buffer[ch])
    
    return output_buffer

def design_basic_filter(filter_type, frequency, Q, sr):
    """Design basic filters with different implementations"""
    nyquist = sr / 2
    normalized_frequency = frequency / nyquist
    
    # Randomly choose filter implementation
    filter_implementations = ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']
    implementation = np.random.choice(filter_implementations)
    
    # Random filter order (2-8, even numbers for stability)
    order = np.random.choice([2, 4, 6, 8])
    
    # Parameters for different filter types
    if implementation == 'butter':
        # Butterworth - maximally flat passband
        if filter_type == 'lowpass':
            return signal.butter(order, normalized_frequency, btype='low', output='sos')
        elif filter_type == 'highpass':
            return signal.butter(order, normalized_frequency, btype='high', output='sos')
        elif filter_type == 'bandpass':
            bandwidth = frequency / Q
            low = (frequency - bandwidth/2) / nyquist
            high = (frequency + bandwidth/2) / nyquist
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            return signal.butter(order//2, [low, high], btype='band', output='sos')
        elif filter_type == 'bandstop':
            bandwidth = frequency / Q
            low = (frequency - bandwidth/2) / nyquist
            high = (frequency + bandwidth/2) / nyquist
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            return signal.butter(order//2, [low, high], btype='bandstop', output='sos')
    
    elif implementation == 'cheby1':
        # Chebyshev Type I - ripple in passband
        ripple_db = np.random.uniform(0.1, 3.0)  # Random ripple amount
        if filter_type == 'lowpass':
            return signal.cheby1(order, ripple_db, normalized_frequency, btype='low', output='sos')
        elif filter_type == 'highpass':
            return signal.cheby1(order, ripple_db, normalized_frequency, btype='high', output='sos')
        elif filter_type == 'bandpass':
            bandwidth = frequency / Q
            low = (frequency - bandwidth/2) / nyquist
            high = (frequency + bandwidth/2) / nyquist
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            return signal.cheby1(order//2, ripple_db, [low, high], btype='band', output='sos')
        elif filter_type == 'bandstop':
            bandwidth = frequency / Q
            low = (frequency - bandwidth/2) / nyquist
            high = (frequency + bandwidth/2) / nyquist
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            return signal.cheby1(order//2, ripple_db, [low, high], btype='bandstop', output='sos')
    
    elif implementation == 'cheby2':
        # Chebyshev Type II - ripple in stopband
        stopband_attenuation_db = np.random.uniform(20, 60)  # Random stopband attenuation
        if filter_type == 'lowpass':
            return signal.cheby2(order, stopband_attenuation_db, normalized_frequency, btype='low', output='sos')
        elif filter_type == 'highpass':
            return signal.cheby2(order, stopband_attenuation_db, normalized_frequency, btype='high', output='sos')
        elif filter_type == 'bandpass':
            bandwidth = frequency / Q
            low = (frequency - bandwidth/2) / nyquist
            high = (frequency + bandwidth/2) / nyquist
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            return signal.cheby2(order//2, stopband_attenuation_db, [low, high], btype='band', output='sos')
        elif filter_type == 'bandstop':
            bandwidth = frequency / Q
            low = (frequency - bandwidth/2) / nyquist
            high = (frequency + bandwidth/2) / nyquist
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            return signal.cheby2(order//2, stopband_attenuation_db, [low, high], btype='bandstop', output='sos')
    
    elif implementation == 'ellip':
        # Elliptic (Cauer) - ripple in both passband and stopband
        ripple_db = np.random.uniform(0.1, 3.0)
        stopband_attenuation_db = np.random.uniform(20, 60)
        if filter_type == 'lowpass':
            return signal.ellip(order, ripple_db, stopband_attenuation_db, normalized_frequency, btype='low', output='sos')
        elif filter_type == 'highpass':
            return signal.ellip(order, ripple_db, stopband_attenuation_db, normalized_frequency, btype='high', output='sos')
        elif filter_type == 'bandpass':
            bandwidth = frequency / Q
            low = (frequency - bandwidth/2) / nyquist
            high = (frequency + bandwidth/2) / nyquist
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            return signal.ellip(order//2, ripple_db, stopband_attenuation_db, [low, high], btype='band', output='sos')
        elif filter_type == 'bandstop':
            bandwidth = frequency / Q
            low = (frequency - bandwidth/2) / nyquist
            high = (frequency + bandwidth/2) / nyquist
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            return signal.ellip(order//2, ripple_db, stopband_attenuation_db, [low, high], btype='bandstop', output='sos')
    
    elif implementation == 'bessel':
        # Bessel - linear phase response
        if filter_type == 'lowpass':
            return signal.bessel(order, normalized_frequency, btype='low', output='sos', norm='phase')
        elif filter_type == 'highpass':
            return signal.bessel(order, normalized_frequency, btype='high', output='sos', norm='phase')
        elif filter_type == 'bandpass':
            bandwidth = frequency / Q
            low = (frequency - bandwidth/2) / nyquist
            high = (frequency + bandwidth/2) / nyquist
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            return signal.bessel(order//2, [low, high], btype='band', output='sos', norm='phase')
        elif filter_type == 'bandstop':
            bandwidth = frequency / Q
            low = (frequency - bandwidth/2) / nyquist
            high = (frequency + bandwidth/2) / nyquist
            low = max(0.01, min(low, 0.99))
            high = max(low + 0.01, min(high, 0.99))
            return signal.bessel(order//2, [low, high], btype='bandstop', output='sos', norm='phase')

def design_parametric_filter(filter_type, frequency, Q, gain_db, sr):
    """Design parametric filters (peak, lowshelf, highshelf)"""
    # Convert to linear gain
    gain_linear = 10 ** (gain_db / 20)
    
    # Normalized frequency
    w0 = 2 * np.pi * frequency / sr
    
    # Compute alpha
    alpha = np.sin(w0) / (2 * Q)
    
    # Compute filter coefficients based on type
    if filter_type == 'peak':
        a0 = 1 + alpha / gain_linear
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / gain_linear
        b0 = 1 + alpha * gain_linear
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * gain_linear
    
    elif filter_type == 'lowshelf':
        A = gain_linear
        sqrt_A = np.sqrt(A)
        
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * sqrt_A * alpha
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * sqrt_A * alpha)
    
    elif filter_type == 'highshelf':
        A = gain_linear
        sqrt_A = np.sqrt(A)
        
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * sqrt_A * alpha
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * sqrt_A * alpha)
    
    # Normalize coefficients
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    
    # Convert to second-order sections
    sos = np.array([[b0, b1, b2, 1, a1, a2]])
    
    return sos

if __name__ == "__main__":
    import numpy as np
    import soundfile as sf
    sr = 44100  # Sample rate
    length = 44100  # 1 second of audio
    n_channels = 2  # Stereo audio
    audio_buffer = np.random.randn(n_channels, length)  # Random audio buffer
    
    output_buffer = apply_random_eq(audio_buffer, sr)
    sf.write('input_audio.wav', audio_buffer.T, sr)  # Save the processed audio
    sf.write('output_audio.wav', output_buffer.T, sr)  # Save the processed audio
    