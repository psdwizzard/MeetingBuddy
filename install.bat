@echo off
REM Clone the Repository
git clone https://github.com/psdwizzard/MeetingBuddy.git
if errorlevel 1 (
    echo Error cloning repository. Exiting.
    pause
    exit /b 1
)
cd MeetingBuddy

REM Create a Virtual Environment named 'venv'
python -m venv venv
if errorlevel 1 (
    echo Error creating virtual environment. Exiting.
    pause
    exit /b 1
)

REM Activate the Virtual Environment
call venv\Scripts\activate

REM Install Required Packages
pip install customtkinter
pip install pillow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U openai-whisper
pip install sounddevice
pip install numpy
pip install pyperclip
pip install openai==0.028
pip install python-docx
pip install fpdf
pip install requests

REM Run the Application
call start.bat

REM Pause to keep the window open in case of errors or to see output
pause
