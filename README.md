# MeetingBuddy

A desktop application for real-time speech transcription using OpenAI's Whisper models and advanced LLM-based summarization techniques.

![Whisper Mic App Screenshot](https://raw.githubusercontent.com/psdwizzard/MeetingBuddy/refs/heads/main/Screenshot.png)

## Features

### Whisper Transcription
- Supports all six Whisper V2 models (Tiny to Turbo) for high-quality transcription.

### Push-to-Talk Transcription
- Hold down the "Listen" button to capture and transcribe audio in real time.

### Meeting Recording Mode
- Record meetings (up to 2 hours), automatically transcribe, and save transcripts with timestamped filenames.

### Live Transcription Mode
- Continuous, real-time transcription with timestamps.

### Advanced Meeting Mode with Summarization
In addition to standard transcription, the advanced mode processes the full transcript to generate:

#### Bullet-Point Summaries
- The transcript is split into chunks (based on an approximate token limit of 1900 tokens without splitting lines) and each chunk is summarized into bullet points.

#### Executive Summary
- The bullet points from all chunks are combined and used to generate a concise executive summary.

### Dual Summarization Options

#### Local Summarization via Ollama API
- Summaries are generated using a locally hosted LLM.
- Customizable via settings, including an optional IP override.

#### Cloud Summarization via OpenAI API
- Alternatively, if an API key is provided, the entire transcript is sent to OpenAI for summarization.

### Additional Features
- Resource Management: The app carefully unloads the Whisper model before running summarization (to free up VRAM) and reloads it afterward if required.
- `<think>` Tag Filtering: When summarizing, any text enclosed in `<think>...</think>` tags is removed so that internal notes are not included in the final summary.
- GPU Acceleration: Supports CUDA for improved performance (if a CUDA-compatible GPU is available).
- Persistent Settings: User preferences (default model, API keys, whether to keep models loaded, custom Ollama IP, etc.) are saved in a JSON file so settings persist between sessions.
- Dark Mode Interface: The GUI uses a dark theme via customTkinter.

## Installation

### Clone the Repository
```bash
git clone https://github.com/psdwizzard/MeetingBuddy.git
cd MeetingBuddy
```

### Create and Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Install Required Packages
```bash
pip install customtkinter
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U openai-whisper
pip install sounddevice
pip install numpy
pip install pyperclip
pip install openai==0.028
```

### Run the Application
```bash
start.bat
```

## Usage

### Transcription Modes

#### Push-to-Talk
- Hold down the "Listen" button while speaking; release to transcribe.

#### Meeting Recording
- Click "Start Meeting" to begin recording (max 2 hours)
- Click "End Meeting" to stop recording and automatically transcribe and save the transcript.

#### Live Transcription
- Click "Live Transcribe" for continuous real-time transcription (with timestamps)
- Click "Stop Live" to end.

### Advanced Meeting Mode & Summarization

#### Advanced Meeting Mode
In addition to transcribing the meeting, advanced mode processes the transcript to produce an executive summary:

- The transcript is split into chunks based on an approximate token limit of 1900 tokens (ensuring no line is split in half)
- Each chunk is summarized into bullet points only
- These bullet points are then combined and reprocessed to generate an overall executive summary (which includes a concise paragraph summary along with key takeaways and action items)

#### Summarization Options
You can choose to use either:

**Local Summarization (Ollama):**
- Processes the transcript locally with custom prompts

**Cloud Summarization (OpenAI):**
- If an OpenAI API key is provided, the entire transcript is sent to OpenAI for summarization

#### Resource Management
Before summarization, the app unloads the Whisper model to free up VRAM. For local summarization, the Ollama model is loaded only once for all transcript chunks, then unloaded once the process is complete.

### Settings

#### Model Selection
- Choose from Tiny, Base, Small, Medium, Large, or Turbo

#### Performance Options
- Keep Model Loaded: Toggle to keep the Whisper model loaded in VRAM for faster subsequent transcriptions
- Use CUDA: Enable GPU acceleration (requires a CUDA-compatible NVIDIA GPU)

#### Ollama Settings
- Choose the default local LLM model
- Optionally, specify a custom IP address for the Ollama API

#### OpenAI Settings
- Toggle to use the OpenAI API instead of local summarization
- Enter your OpenAI API key and select the desired model (gpt-3.5-turbo or gpt-4)

### Output

#### Transcripts
- Transcriptions are displayed in real time with timestamps
- Saved in the meetings folder using timestamped filenames

#### Summaries
- Advanced meeting mode produces an executive summary based on bullet-point summaries of transcript segments

#### Copy Functionality
- Use the "Copy" button to copy transcribed text or summaries to the clipboard

## System Requirements
- Operating System: Windows OS
- Python Version: Python 3.8 or higher
- GPU Requirements: NVIDIA GPU with CUDA support (optional for GPU acceleration)
