# whisperTTS.py

import torch
from whisperspeech.pipeline import Pipeline
import tempfile
import soundfile as sf

def generate_speech(text):
    """Generate and play synthesized speech from text using WhisperSpeech with reduced precision and security settings."""
    print("Rendering transcription as speech using WhisperSpeech...")

    # Initialize the WhisperSpeech pipeline with reduced precision for CPU
    pipe = Pipeline(torch_compile=True)

    # Convert model to reduced precision, use float16 if supported, else bfloat16
    dtype = torch.float16 if torch.cuda.is_available() or torch.has_cpu_float16 else torch.bfloat16
    pipe.to(dtype=dtype)

    # Update torch.load settings within the pipeline to enhance security
    original_torch_load = torch.load  # Save the original torch.load function
    torch.load = lambda *args, **kwargs: original_torch_load(*args, weights_only=True, **kwargs)

    try:
        # Generate speech from text
        audio = pipe.generate(text) if hasattr(pipe, 'generate') else pipe(text)
    finally:
        # Restore original torch.load after pipeline execution
        torch.load = original_torch_load

    # Save the generated speech to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        sf.write(temp_audio_file.name, audio, 22050)  # Assuming 22.05 kHz sample rate
        print("TTS audio generated.")
        return temp_audio_file.name
