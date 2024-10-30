#!/usr/bin/env python3

import whisper
import pyaudio
import wave

# Function to list available audio input devices
def list_microphones():
    audio = pyaudio.PyAudio()
    print("Available audio devices:")
    for i in range(audio.get_device_count()):
        device = audio.get_device_info_by_index(i)
        print(f"Device {i}: {device['name']}")
    audio.terminate()

# Function to record audio from the selected device
def record_audio(device_index, output_file="test_audio.wav", record_seconds=15):  # 15 seconds
    CHUNK = 1024  # Buffer size
    FORMAT = pyaudio.paInt16  # Audio format
    CHANNELS = 1  # Mono audio
    RATE = 48000  # Sample rate

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                        input_device_index=device_index, frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    try:
        for _ in range(0, int(RATE / CHUNK * record_seconds)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print("Recording complete.")

# Function to transcribe audio using Whisper
def transcribe_audio(model_size="tiny", input_file="test_audio.wav"):
    model = whisper.load_model(model_size, device="cpu")
    print("Transcribing...")
    result = model.transcribe(input_file, language="en")
    print("Transcription:", result["text"])

# Main script flow
def main():
    list_microphones()
    device_index = int(input("Enter the device index for your microphone: "))
    record_audio(device_index)
    transcribe_audio()

if __name__ == "__main__":
    main()

