# whisperTTS.py

import torch
from whisperspeech.pipeline import Pipeline
import tempfile
import soundfile as sf
import warnings

def generate_speech(text):
    """Generate and play synthesized speech from text using WhisperSpeech with smallest models and 48kHz sample rate."""

    print("Rendering transcription as speech using WhisperSpeech...")

       # Suppress specific FutureWarnings and profiler warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Profiler function.*")


    # Initialize the WhisperSpeech pipeline with the smallest available models for T2S and S2A
    pipe = Pipeline(
        t2s_ref='collabora/whisperspeech:t2s-tiny-en+pl.model',
        s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model',
        torch_compile=True
    )

    # Generate speech from text
    audio = pipe.generate(text) if hasattr(pipe, 'generate') else pipe(text)

    # Save the generated speech to a temporary file at 48kHz
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        sf.write(temp_audio_file.name, audio, 48000)  # Save at 48kHz
        print("TTS audio generated at 48kHz.")
        return temp_audio_file.name
