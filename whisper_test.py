#!/usr/bin/env python

import whisper
import sounddevice as sd
import numpy as np
import threading
import queue
import pyperclip

def record_audio(audio_queue, duration=15, fs=16000):
    """Records audio from the default microphone and puts it into a queue."""
    print("Recording started...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    audio_queue.put(audio)
    # Signal that recording is done
    audio_queue.put(None)

def transcribe_audio(model_size="tiny", audio_queue=None):
    """Transcribes audio data from the queue using the Whisper model."""
    model = whisper.load_model(model_size)
    transcriptions = []
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        # Transcribe
        result = model.transcribe(audio.flatten(), fp16=False)
        print("Transcription:", result["text"])
        transcriptions.append(result["text"])
    # After processing is done
    full_transcription = "\n".join(transcriptions)
    # Copy to clipboard
    try:
        pyperclip.copy(full_transcription)
        print("Transcription has been copied to the clipboard.")
    except pyperclip.PyperclipException as e:
        print(f"Failed to copy transcription to clipboard: {e}")

def main():
    # Create a queue for audio data
    audio_queue = queue.Queue()

    # Create and start threads
    record_thread = threading.Thread(target=record_audio, args=(audio_queue,))
    transcribe_thread = threading.Thread(target=transcribe_audio, args=("tiny", audio_queue))

    record_thread.start()
    transcribe_thread.start()

    # Wait for threads to finish
    record_thread.join()
    transcribe_thread.join()

    print("Transcription complete.")

if __name__ == "__main__":
    main()
