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
import threading
import sys
import time

def list_microphones():
    devices = sd.query_devices()
    input_devices = [dev for dev in devices if dev['max_input_channels'] > 0]
    print("\nAvailable input devices:")
    for idx, dev in enumerate(input_devices):
        print(f"{idx}: {dev['name']}")
    return input_devices

def record_audio(audio_queue, device_index, fs=48000):
    """Records audio from the selected microphone until Enter is pressed."""
    print(f"Recording started at {fs} Hz...")
    audio_data = []
    recording = threading.Event()
    recording.set()
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Recording error: {status}", file=sys.stderr)
        if recording.is_set():
            audio_data.append(indata.copy())
        else:
            raise sd.CallbackStop()

    # Start input stream
    stream = sd.InputStream(
        samplerate=fs,
        device=device_index,
        channels=1,
        dtype='float32',
        callback=callback
    )

    # Start the stream in a separate thread
    with stream:
        # Start the timer display in a separate thread
        def display_timer():
            while recording.is_set():
                elapsed_time = time.time() - start_time
                print(f"\rRecording... {elapsed_time:.1f}s", end='')
                time.sleep(0.1)
            print("\nRecording stopped.")

        timer_thread = threading.Thread(target=display_timer)
        timer_thread.start()

        # Wait for user to press Enter
        input("\nPress Enter to stop recording...\n")
        recording.clear()
        timer_thread.join()

    # Concatenate all recorded data
    if audio_data:
        audio = np.concatenate(audio_data, axis=0)
        audio_duration = audio.shape[0] / fs
        print(f"Recording complete. Total duration: {audio_duration:.2f}s")
        # Save audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            sf.write(temp_audio_file.name, audio, fs)
            audio_queue.put((temp_audio_file.name, audio_duration))
    else:
        print("No audio data recorded.")
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

def transcribe_audio(model_sizes, audio_queue=None):
    """Transcribes audio data from the queue using the specified Whisper model(s)."""
    # Override torch.load to suppress the FutureWarning
    original_torch_load = torch.load
    torch.load = functools.partial(torch.load, weights_only=True)

    transcriptions = []
    while True:
        audio_item = audio_queue.get()
        if audio_item is None:
            break
        audio_file_path, audio_duration = audio_item
        print(f"Received audio file: {audio_file_path}")
        # Play back the audio before transcription
        play_audio(audio_file_path)

        # Use Whisper's load_audio function
        audio = whisper.load_audio(audio_file_path)

        for model_size in model_sizes:
            print(f"\nTranscribing with model '{model_size}'...")
            start_time = time.time()
            model = whisper.load_model(model_size)
            transcription_start = time.time()
            result = model.transcribe(audio, fp16=False, language='en')
            transcription_end = time.time()
            transcription_time = transcription_end - transcription_start
            coefficient = audio_duration / transcription_time if transcription_time > 0 else 0
            print(f"Transcription ({model_size}): {result['text']}")
            print(f"Audio Duration: {audio_duration:.2f}s")
            print(f"Transcription Time: {transcription_time:.2f}s")
            print(f"Coefficient of Real-time: {coefficient:.2f}x")
            transcriptions.append((model_size, result['text']))

    # Restore the original torch.load
    torch.load = original_torch_load

    # Copy the last transcription to clipboard
    if transcriptions:
        _, last_transcription = transcriptions[-1]
        try:
            pyperclip.copy(last_transcription)
            print("\nLast transcription has been copied to the clipboard.")
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

    # Prompt the user to select the Whisper model
    # List available models
    available_models = ["tiny", "base", "small", "medium", "large", "all"]
    print("\nAvailable Whisper models:")
    for idx, model_name in enumerate(available_models):
        print(f"{idx}: {model_name}")

    # Prompt user to select a model
    while True:
        try:
            model_index = int(input("\nEnter the model index to use for transcription: "))
            if 0 <= model_index < len(available_models):
                selected_model = available_models[model_index]
                print(f"Selected model: {selected_model}")
                break
            else:
                print("Invalid model index. Please try again.")
        except ValueError:
            print("Please enter a valid integer.")

    # Determine the model sizes to use
    if selected_model == "all":
        model_sizes = ["tiny", "base", "small", "medium", "large"]
    else:
        model_sizes = [selected_model]

    # Record audio
    audio_queue = queue.Queue()
    record_audio(audio_queue, selected_device_index)

    # Transcribe audio using the selected model(s)
    transcribe_audio(model_sizes, audio_queue)

    print("Transcription complete.")

if __name__ == "__main__":
    main()
