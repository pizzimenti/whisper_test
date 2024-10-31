# whisperTTS.py

import torch
from whisperspeech.pipeline import Pipeline  # Updated import to use Pipeline class
import tempfile
import soundfile as sf

def generate_speech(text):
    """Generate and play synthesized speech from text using WhisperSpeech."""
    print("Rendering transcription as speech using WhisperSpeech...")

    # Initialize the WhisperSpeech pipeline
    pipe = Pipeline(torch_compile=True)

    # Generate speech from text
    audio = pipe(text)

    # Save the speech to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        sf.write(temp_audio_file.name, audio, 22050)  # Assuming 22.05 kHz sample rate
        print("TTS audio generated.")

        return temp_audio_file.name
