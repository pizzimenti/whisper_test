#!/bin/zsh

# Create and activate a virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv whisper_env
source whisper_env/bin/activate

# Upgrade pip within the virtual environment
echo "Upgrading pip..."
pip install --upgrade pip

# Install required Python packages
echo "Installing required Python packages..."
pip install -r requirements.txt

echo "All Python dependencies have been installed."
echo "The virtual environment 'whisper_env' is activated."
echo "You can now run your script using 'python whisper_test.py'."

# Keep the virtual environment activated and provide instructions to run the script
source whisper_env/bin/activate && zsh
