# Whisper Mic App

A desktop application for real-time speech transcription using OpenAI's Whisper models.

## Features

- Support for all six Whisper V2 models (Tiny to Turbo)
- Push-to-talk transcription
- Meeting recording with automatic transcription (up to 2 hours)
- Live transcription mode
- GPU acceleration support (CUDA)
- Model caching for improved performance
- Meeting transcripts saved by date
- Dark mode interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/psdwizzard/MeetingBuddy.git
cd MeetingBuddy
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install required packages:
```bash
pip install customtkinter
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U openai-whisper
pip install sounddevice
pip install numpy
pip install pyperclip
pip install openai==.028
```

4. Run the application:
```bash
start.bat
```

## Usage

### Transcription Modes

- **Push-to-Talk**: Hold down the "Listen" button while speaking, release to transcribe
- **Meeting Recording**: Click "Start Meeting" to begin recording (max 2 hours). Click "End Meeting" to stop and save
- **Live Transcription**: Click "Live Transcribe" for continuous transcription, "Stop Live" to end

### Settings

- **Model Selection**: Choose from Tiny, Base, Small, Medium, Large, or Turbo models
- **Keep Model Loaded**: Toggle to keep model in VRAM for faster subsequent transcriptions
- **Use CUDA**: Toggle GPU acceleration (requires CUDA-compatible GPU)

## Output

- Meeting transcriptions are saved in the `meetings` folder with timestamp filenames
- Use the "Copy" button to copy transcribed text to clipboard

## System Requirements

- Windows OS
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
