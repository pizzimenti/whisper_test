# whisperTTS.py

import torch
import numpy as np
from whisperspeech.pipeline import Pipeline
import tempfile
import soundfile as sf
import warnings
import logging
from scipy.signal import resample

# Set PyTorch logging to show only critical issues, suppressing profiler warnings
logging.getLogger("torch").setLevel(logging.CRITICAL)

def generate_speech(text):
    """Generate and play synthesized speech from text using WhisperSpeech with detailed logging and 48kHz sample rate."""

    print("Starting TTS process with WhisperSpeech...")

    # Suppress specific FutureWarnings and profiler warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Profiler function.*")

    # Define model names for T2S and S2A stages
    t2s_model = 'collabora/whisperspeech:t2s-tiny-en+pl.model'
    s2a_model = 'collabora/whisperspeech:s2a-q4-tiny-en+pl.model'

    # Initialize the WhisperSpeech pipeline with specific models for T2S and S2A
    print(f"Loading Text-to-Semantic (T2S) model: {t2s_model}")
    print(f"Loading Semantic-to-Audio (S2A) model: {s2a_model}")
    pipe = Pipeline(
        t2s_ref=t2s_model,
        s2a_ref=s2a_model,
        torch_compile=True
    )

    # Execute Text-to-Semantic (T2S) stage
    print("Running Text-to-Semantic (T2S) stage...")
    audio = pipe.generate(text) if hasattr(pipe, 'generate') else pipe(text)
    
    # Define the assumed original sample rate and target sample rate
    original_sample_rate = 24000  # Assume 24kHz if the model outputs at this rate
    target_sample_rate = 48000

    # Log details about audio processing
    print(f"Original audio sample rate: {original_sample_rate} Hz")
    print(f"Resampling to target sample rate: {target_sample_rate} Hz")

    # Ensure audio is in a format compatible with soundfile and resample to 48kHz if necessary
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()  # Convert to NumPy if it's a tensor
    if audio.ndim > 1:
        audio = audio.squeeze()  # Ensure it's a 1D array if needed
    audio = audio.astype(np.float32)  # Convert to float32 for compatibility

    # Resample audio to 48kHz if original sample rate differs
    if original_sample_rate != target_sample_rate:
        num_samples = int(len(audio) * target_sample_rate / original_sample_rate)
        audio = resample(audio, num_samples)
        print("Resampling complete.")

    # Save the resampled audio to a temporary file at 48kHz
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        sf.write(temp_audio_file.name, audio, target_sample_rate)
        print("TTS audio generated, resampled to 48kHz, and saved.")
        return temp_audio_file.name
                                    