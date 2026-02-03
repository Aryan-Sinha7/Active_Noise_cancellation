import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
import soundfile as sf

def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np + 1] *= phases
    f[-1:-1 - Np:-1] = np.conj(f[1:Np + 1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    f[np.logical_and(freqs >= min_freq, freqs <= max_freq)] = 1
    return fftnoise(f)

def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window='hann')

def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)

def _amp_to_db(x):
    return librosa.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

def _db_to_amp(x):
    return librosa.db_to_amplitude(x, ref=1.0)

# Low-Pass Filter for Final Cleanup 
from scipy.signal import butter, filtfilt
def lowpass_filter(data, cutoff, fs, order=6):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, data)

# Main Noise Reduction Function 
def removeNoise(audio_clip, noise_clip, n_grad_freq=3, n_grad_time=5,
                n_fft=2048, win_length=2048, hop_length=512,
                n_std_thresh=1.8, prop_decrease=0.85):
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

    # STFT over signal
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))

    # Create mask
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    sig_mask = sig_stft_db < db_thresh

    # Smooth mask
    smoothing_filter = np.outer(
        np.concatenate([
            np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
            np.linspace(1, 0, n_grad_freq + 2),
        ])[1:-1],
        np.concatenate([
            np.linspace(0, 1, n_grad_time + 1, endpoint=False),
            np.linspace(1, 0, n_grad_time + 2),
        ])[1:-1],
    )
    smoothing_filter /= np.sum(smoothing_filter)
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease

    # Apply mask
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(sig_stft_db)) * mask_gain_dB * sig_mask
    )

    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (1j * sig_imag_masked)

    # Reconstruct signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    return recovered_signal

# Example Usage
if __name__ == "__main__":
    # === Load Audio ===
    wav_loc = r"C:\Users\Aryan\Downloads\Noise-Reduction-master\Noise-Reduction-master\Audiofile.wav"
    rate, data = wavfile.read(wav_loc)

    # Convert to mono if stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Normalize
    data = data.astype(float)
    data = data / np.max(np.abs(data))

    # === Create Artificial Noise for Demo ===
    noise_len = 2  # seconds
    noise = band_limited_noise(4000, 12000, samples=len(data), samplerate=rate) * 10
    noise_clip = noise[: rate * noise_len]
    noisy_signal = data + noise

    # === Perform Noise Reduction ===
    clean = removeNoise(noisy_signal, noise_clip)

    # === Post Filter (Low-Pass) ===
    clean = lowpass_filter(clean, cutoff=14000, fs=rate)

    # === Save High-Quality Output ===
    sf.write("output_clean.wav", clean, rate, subtype='FLOAT')
    print("High-quality noise reduction complete → saved as output_clean.wav")

    # ===== Plot Waveforms =====
    plt.figure(figsize=(14,6))
    plt.subplot(2,1,1)
    librosa.display.waveshow(noisy_signal, sr=rate, color='red')
    plt.title("Input Noisy Signal")

    plt.subplot(2,1,2)
    librosa.display.waveshow(clean, sr=rate, color='green')
    plt.title("Output Cleaned Signal")
    plt.tight_layout()
    plt.show()

    # ===== Plot Spectrograms =====
    plt.figure(figsize=(14,6))
    plt.subplot(2,1,1)
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_signal)), ref=np.max)
    librosa.display.specshow(D_noisy, sr=rate, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Input Noisy Audio Spectrogram")

    plt.subplot(2,1,2)
    D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean)), ref=np.max)
    librosa.display.specshow(D_clean, sr=rate, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Output Cleaned Audio Spectrogram")
    plt.tight_layout()
    plt.show()
what type of sound i use to show its working properly