# Active_Noise_cancellation

# Audio Noise Reduction Using Spectral Subtraction (Python)

This project demonstrates a complete audio noise reduction system using Spectral Subtraction and Short-Time Fourier Transform (STFT) techniques in Python.  
It effectively removes stationary background noise such as white noise, fan noise, or hiss while preserving the original audio signal.

This project is suitable for:
- Signal Processing demonstrations
- Final-year engineering projects
- Research and academic submissions
- Audio preprocessing pipelines

---

## Key Features

- Spectral subtraction–based noise suppression
- STFT and inverse STFT (ISTFT) processing
- Frequency and time-domain mask smoothing
- Artificial noise generation for testing
- Low-pass filtering for final cleanup
- Waveform and spectrogram visualization
- High-quality floating-point WAV output
- Supports mono and stereo audio
  

## Algorithm Explanation

1. **Noise Estimation**
   - A noise-only segment is analyzed
   - Mean and standard deviation are computed for each frequency bin

2. **Spectral Thresholding**
   - Noise thresholds are generated using statistical measures

3. **Mask Creation**
   - A soft mask identifies noise-dominant regions in the signal

4. **Mask Smoothing**
   - Time–frequency smoothing reduces musical noise artifacts

5. **Spectral Subtraction**
   - Noise components are attenuated
   - Phase information is preserved

6. **Signal Reconstruction**
   - Clean signal is reconstructed using inverse STFT
   - Low-pass filtering improves final quality


## Recommended Audio for Best Results

### Best Signal Types
- Human speech (single speaker) 
- Piano or violin notes 
- Pure sine or multi-tone signals 

### Recommended Noise Types
- White noise
- Fan or AC noise
- Broadband hiss


---

## Requirements

Install the required Python libraries:


---

## How to Run

1. Place your input WAV file in the project folder.
2. Update the audio path in the script:


3. Run the program:

4. Output file will be generated as:


---

## Output Visualization

The program displays:

- Waveform comparison
  - Input noisy signal
  - Output cleaned signal

- Spectrogram comparison
  - Before noise reduction
  - After noise reduction

These plots clearly show noise suppression in both time and frequency domains.


---

## Output Visualization

The program displays:

- Waveform comparison
  - Input noisy signal
  - Output cleaned signal

- Spectrogram comparison
  - Before noise reduction
  - After noise reduction

These plots clearly show noise suppression in both time and frequency domains.

---

## Example Noise Generation

Artificial band-limited noise is used for testing:


This simulates real-world background noise conditions.

---

## Applications

- Speech enhancement
- Audio restoration
- Communication systems
- Signal processing education
- Machine learning audio preprocessing

---

## Future Enhancements

- Adaptive noise estimation
- Real-time noise reduction
- Deep learning–based denoising
- Stereo and multi-channel support
- Voice activity detection (VAD)

---

## Author

Aryan Kr Sinha  
Electronics and Communication Engineering  






