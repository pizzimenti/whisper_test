#!/usr/bin/env python

import whisper
import torch
import sounddevice as sd
import numpy as np
import pyperclip
import soundfile as sf
import queue
import tempfile
import functools

def list_microphones():
    devices = sd.query_devices()
    input_devices = [dev for dev in devices if dev['max_input_channels'] > 0]
    print("\nAvailable input devices:")
    for idx, dev in enumerate(input_devices):
        print(f"{idx}: {dev['name']}")
    return input_devices

def record_audio(audio_queue, device_index, duration=15, fs=48000):
    """Records audio from the selected microphone and puts the file path into a queue."""
    print(f"Recording started at {fs} Hz...")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', device=device_index)
        sd.wait()
        print("Recording complete.")
        print(f"Audio data shape: {audio.shape}, dtype: {audio.dtype}")

        # Save audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            sf.write(temp_audio_file.name, audio, fs)
            audio_queue.put(temp_audio_file.name)
    except Exception as e:
        print(f"An error occurred during recording: {e}")
        audio_queue.put(None)
    # Signal that recording is done
    audio_queue.put(None)

def play_audio(file_path):
    """Plays the audio file using sounddevice and soundfile."""
    print("Playing back the recorded audio...")
    try:
        data, fs = sf.read(file_path, dtype='float32')
        sd.play(data, fs)
        sd.wait()  # Wait until playback is finished
        print("Playback complete.")
    except Exception as e:
        print(f"An error occurred during playback: {e}")

def transcribe_audio(model_size="tiny", audio_queue=None):
    """Transcribes audio data from the queue using the Whisper model."""
    # Override torch.load to suppress the FutureWarning
    original_torch_load = torch.load
    torch.load = functools.partial(torch.load, weights_only=True)

    model = whisper.load_model(model_size)

    # Restore the original torch.load
    torch.load = original_torch_load

    transcriptions = []
    while True:
        audio_file_path = audio_queue.get()
        if audio_file_path is None:
            break
        print(f"Received audio file: {audio_file_path}")

        # Play back the audio before transcription
        play_audio(audio_file_path)

        # Use Whisper's load_audio function
        audio = whisper.load_audio(audio_file_path)

        # Transcribe the audio
        result = model.transcribe(audio, fp16=False)
        print("Transcription:", result["text"])
        transcriptions.append(result["text"])

    # Copy transcription to clipboard
    full_transcription = "\n".join(transcriptions)
    try:
        pyperclip.copy(full_transcription)
        print("Transcription has been copied to the clipboard.")
    except pyperclip.PyperclipException as e:
        print(f"Failed to copy transcription to clipboard: {e}")

def main():
    # List microphones and allow the user to select one
    input_devices = list_microphones()
    while True:
        try:
            device_index = int(input("\nEnter the device index for your microphone: "))
            if 0 <= device_index < len(input_devices):
                selected_device_index = input_devices[device_index]['index']
                selected_device_name = input_devices[device_index]['name']
                print(f"Selected device: {selected_device_name}")
                break
            else:
                print("Invalid device index. Please try again.")
        except ValueError:
            print("Please enter a valid integer.")

    # Record audio
    audio_queue = queue.Queue()
    record_audio(audio_queue, selected_device_index)

    # Transcribe audio
    transcribe_audio("tiny", audio_queue)

    print("Transcription complete.")

if __name__ == "__main__":
    main()

