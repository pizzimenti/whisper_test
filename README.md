# Whisper Transcription Tool

A Python script that records audio from your microphone, transcribes it using OpenAI's Whisper models, and copies the transcription to your clipboard. This tool supports multiple Whisper model sizes and provides real-time transcription metrics.

## Features

- **Microphone Selection**: Choose from available input devices for recording.
- **Model Selection**: Select from different Whisper models (`tiny`, `base`, `small`, `medium`, `large`, or `all`).
- **Real-time Recording**: Displays recording duration and stops recording upon user input.
- **Audio Playback**: Plays back the recorded audio before transcription.
- **Transcription**: Transcribes audio using the selected Whisper model(s).
- **Performance Metrics**: Displays transcription time and real-time coefficient.
- **Clipboard Integration**: Copies the last transcription result to the clipboard.

## Requirements

- Python 3.7 or higher
- Microphone for audio input

## Installation

Run the installation script to set up the environment and install dependencies:

```bash
./install.sh

Follow the on-screen prompts to:

- Select your microphone device.
- Choose the Whisper model(s) for transcription.

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **[OpenAI Whisper](https://github.com/openai/whisper)**: For the powerful speech recognition models.
- **[PyTorch](https://pytorch.org/)**: For the machine learning framework used to run Whisper models.
- **Community Contributors**: For the development of open-source packages utilized in this project.

