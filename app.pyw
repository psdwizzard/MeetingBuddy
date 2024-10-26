import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import whisper
import sounddevice as sd
import numpy as np
import pyperclip
import os
import datetime

# Global variables
loaded_model = None  # To store the loaded model
is_listening = False
recording_stream = None
audio_data = []
meeting_running = False
meeting_audio_data = []
meeting_start_time = None
meeting_thread = None

def start_recording(event=None):
    global recording_stream, audio_data, is_listening
    if is_listening:
        return
    is_listening = True
    listen_button.config(text="Listening...")
    app.update()
    fs = 16000  # Sample rate
    audio_data = []

    def callback(indata, frames, time, status):
        audio_data.append(indata.copy())

    recording_stream = sd.InputStream(samplerate=fs, channels=1, callback=callback)
    recording_stream.start()

def stop_recording(event=None):
    global recording_stream, audio_data, is_listening
    if not is_listening:
        return
    recording_stream.stop()
    recording_stream.close()
    listen_button.config(text="Listen")
    app.update()
    is_listening = False

    # Concatenate all recorded data
    audio = np.concatenate(audio_data, axis=0)
    threading.Thread(target=transcribe_audio, args=(audio,)).start()

def transcribe_audio(audio, save_to_file=False, filename=None):
    global loaded_model
    try:
        disable_buttons()
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, "Transcribing...")
        app.update()

        selected_model = model_var.get()

        # Check if the model is loaded and matches the selected model
        if loaded_model and loaded_model['name'] == selected_model:
            model = loaded_model['model']
        else:
            text_output.delete(1.0, tk.END)
            text_output.insert(tk.END, "Loading model...")
            app.update()
            model = whisper.load_model(selected_model)
            # If "Keep Model Loaded" is enabled, store the model
            if keep_model_loaded.get():
                loaded_model = {'name': selected_model, 'model': model}

        # Transcribe the audio
        result = model.transcribe(audio.flatten(), fp16=False)
        transcription = result['text']
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, transcription)
        if save_to_file and filename:
            save_transcription(transcription, filename)
        enable_buttons()
    except Exception as e:
        enable_buttons()
        messagebox.showerror("Error", str(e))

def copy_text():
    text = text_output.get(1.0, tk.END).strip()
    if text:
        pyperclip.copy(text)
        messagebox.showinfo("Copied", "Text copied to clipboard.")
    else:
        messagebox.showwarning("Warning", "No text to copy.")

def disable_buttons():
    listen_button.config(state=tk.DISABLED)
    start_meeting_button.config(state=tk.DISABLED)
    end_meeting_button.config(state=tk.DISABLED)

def enable_buttons():
    listen_button.config(state=tk.NORMAL)
    start_meeting_button.config(state=tk.NORMAL)
    end_meeting_button.config(state=tk.NORMAL)

def start_meeting():
    global meeting_audio_data, meeting_start_time, meeting_thread, meeting_running
    if meeting_running:
        messagebox.showwarning("Warning", "Meeting is already running.")
        return
    meeting_running = True
    meeting_start_time = datetime.datetime.now()
    meeting_audio_data = []
    start_meeting_button.config(state=tk.DISABLED)
    end_meeting_button.config(state=tk.NORMAL)
    text_output.delete(1.0, tk.END)
    text_output.insert(tk.END, "Meeting started...")
    app.update()
    meeting_thread = threading.Thread(target=record_meeting)
    meeting_thread.start()

def end_meeting():
    global meeting_running
    if not meeting_running:
        messagebox.showwarning("Warning", "No meeting is currently running.")
        return
    meeting_running = False
    end_meeting_button.config(state=tk.DISABLED)
    start_meeting_button.config(state=tk.NORMAL)
    text_output.insert(tk.END, "\nMeeting ended. Transcribing...")
    app.update()

def record_meeting():
    global meeting_audio_data, meeting_start_time, meeting_running
    fs = 16000  # Sample rate

    def callback(indata, frames, time, status):
        meeting_audio_data.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        start_time = datetime.datetime.now()
        while meeting_running:
            if (datetime.datetime.now() - start_time).total_seconds() >= 7200:  # 2 hours
                meeting_running = False
                break
            sd.sleep(1000)  # Sleep for 1 second

    # Concatenate all recorded data
    audio = np.concatenate(meeting_audio_data, axis=0)
    # Generate filename
    filename = meeting_start_time.strftime("%m-%d-%Y-%H%M") + ".txt"
    filepath = os.path.join("meetings", filename)
    threading.Thread(target=transcribe_audio, args=(audio, True, filepath)).start()

def save_transcription(transcription, filepath):
    # Ensure the 'meetings' directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(transcription)
    messagebox.showinfo("Saved", f"Transcription saved to {filepath}")

def on_model_change(*args):
    global loaded_model
    selected_model = model_var.get()
    if keep_model_loaded.get():
        # If a model is loaded and the selected model is different, unload it
        if loaded_model and loaded_model['name'] != selected_model:
            loaded_model = None  # Unload the model

def on_keep_model_change(*args):
    global loaded_model
    if not keep_model_loaded.get():
        loaded_model = None  # Unload the model

# Initialize the main application window
app = tk.Tk()
app.title("Whisper Mic App")

# Dropdown for selecting the model
model_var = tk.StringVar(value="base")
models = ["tiny", "base", "small", "medium", "large", "turbo"]  # Added "turbo" to the list
model_dropdown = tk.OptionMenu(app, model_var, *models)
model_dropdown.pack(pady=10)

# Checkbox to keep the model loaded
keep_model_loaded = tk.BooleanVar(value=False)
keep_model_checkbox = tk.Checkbutton(app, text="Keep Model Loaded", variable=keep_model_loaded)
keep_model_checkbox.pack(pady=5)

# Bind the model change event
model_var.trace('w', on_model_change)
keep_model_loaded.trace('w', on_keep_model_change)

# Frame for buttons
button_frame = tk.Frame(app)
button_frame.pack(pady=10)

# Listen button
listen_button = tk.Button(button_frame, text="Listen")
listen_button.grid(row=0, column=0, padx=5)

# Start Meeting button
start_meeting_button = tk.Button(button_frame, text="Start Meeting", command=start_meeting)
start_meeting_button.grid(row=0, column=1, padx=5)

# End Meeting button
end_meeting_button = tk.Button(button_frame, text="End Meeting", command=end_meeting, state=tk.DISABLED)
end_meeting_button.grid(row=0, column=2, padx=5)

# Bind press and release events to the Listen button
listen_button.bind('<ButtonPress-1>', start_recording)
listen_button.bind('<ButtonRelease-1>', stop_recording)

# Text output area
text_output = scrolledtext.ScrolledText(app, wrap=tk.WORD, width=50, height=10)
text_output.pack(padx=10, pady=10)

# Copy button
copy_button = tk.Button(app, text="Copy", command=copy_text)
copy_button.pack(pady=10)

app.mainloop()
