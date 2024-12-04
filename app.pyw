import customtkinter
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import whisper
import sounddevice as sd
import numpy as np
import pyperclip
import os
import datetime
import torch
import queue

# Set appearance mode and color theme
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

# Global variables
recording_stream = None
audio_data = []
is_listening = False
meeting_running = False
meeting_audio_data = []
meeting_start_time = None
meeting_thread = None
preloaded_model = None
live_transcribing = False
live_audio_queue = queue.Queue()
live_transcribe_thread = None
stop_event = threading.Event()

def start_recording(event=None):
    global recording_stream, audio_data, is_listening
    if is_listening:
        return
    is_listening = True
    listen_button.configure(text="Listening...")
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
    listen_button.configure(text="Listen")
    app.update()
    is_listening = False

    # Concatenate all recorded data
    audio = np.concatenate(audio_data, axis=0)
    threading.Thread(target=transcribe_audio, args=(audio,)).start()

def transcribe_audio(audio, save_to_file=False, filename=None):
    try:
        disable_buttons()
        text_output.delete(1.0, tk.END)
        text_output.insert(tk.END, "Transcribing...")
        app.update()

        selected_model = model_var.get()

        # Decide whether to use CUDA or CPU based on the checkbox
        device = 'cuda' if use_cuda_var.get() and torch.cuda.is_available() else 'cpu'

        if keep_model_loaded_var.get() and preloaded_model is not None:
            model = preloaded_model
        else:
            model = whisper.load_model(selected_model)
            model = model.to(device)

        # Convert audio to tensor and move to the selected device
        audio_tensor = torch.from_numpy(audio).float().to(device)

        # Transcribe the audio
        result = model.transcribe(audio_tensor.flatten(), fp16=device=='cuda')
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
    listen_button.configure(state=tk.DISABLED)
    start_meeting_button.configure(state=tk.DISABLED)
    end_meeting_button.configure(state=tk.DISABLED)
    live_transcribe_button.configure(state=tk.DISABLED)
    stop_live_button.configure(state=tk.DISABLED)

def enable_buttons():
    listen_button.configure(state=tk.NORMAL)
    start_meeting_button.configure(state=tk.NORMAL)
    end_meeting_button.configure(state=tk.NORMAL)
    live_transcribe_button.configure(state=tk.NORMAL)
    stop_live_button.configure(state=tk.NORMAL)

def load_model(model_name):
    global preloaded_model
    print(f"Loading model: {model_name}")

    # Decide whether to use CUDA or CPU based on the checkbox
    device = 'cuda' if use_cuda_var.get() and torch.cuda.is_available() else 'cpu'

    preloaded_model = whisper.load_model(model_name)
    preloaded_model = preloaded_model.to(device)

def unload_model():
    global preloaded_model
    print("Unloading model")
    del preloaded_model
    preloaded_model = None
    torch.cuda.empty_cache()

def on_keep_model_loaded_change():
    if keep_model_loaded_var.get():
        selected_model = model_var.get()
        load_model(selected_model)
    else:
        unload_model()

def on_model_change(selected_model):
    if keep_model_loaded_var.get():
        load_model(selected_model)

def on_use_cuda_change():
    # Reload the model to switch between CUDA and CPU
    if keep_model_loaded_var.get():
        selected_model = model_var.get()
        load_model(selected_model)

def start_meeting():
    global meeting_audio_data, meeting_start_time, meeting_thread, meeting_running
    if meeting_running:
        messagebox.showwarning("Warning", "Meeting is already running.")
        return
    meeting_running = True
    meeting_start_time = datetime.datetime.now()
    meeting_audio_data = []
    start_meeting_button.configure(state=tk.DISABLED)
    end_meeting_button.configure(state=tk.NORMAL)
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
    end_meeting_button.configure(state=tk.DISABLED)
    start_meeting_button.configure(state=tk.NORMAL)
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

def start_live_transcription():
    global live_transcribing, live_audio_queue, live_transcribe_thread, recording_stream, stop_event
    if live_transcribing:
        messagebox.showwarning("Warning", "Live transcription is already running.")
        return

    live_transcribing = True
    stop_event.clear()
    live_transcribe_button.configure(state=tk.DISABLED)
    stop_live_button.configure(state=tk.NORMAL)
    text_output.delete(1.0, tk.END)
    text_output.insert(tk.END, "Live transcription started...")
    app.update()

    fs = 16000  # Sample rate

    def audio_callback(indata, frames, time, status):
        live_audio_queue.put(indata.copy())

    recording_stream = sd.InputStream(samplerate=fs, channels=1, callback=audio_callback)
    recording_stream.start()

    live_transcribe_thread = threading.Thread(target=process_live_audio)
    live_transcribe_thread.start()

def stop_live_transcription():
    global live_transcribing, recording_stream, stop_event
    if not live_transcribing:
        messagebox.showwarning("Warning", "No live transcription is running.")
        return

    live_transcribing = False
    stop_event.set()

    # Stop the recording stream safely
    if recording_stream is not None:
        recording_stream.stop()
        recording_stream.close()
        recording_stream = None

    # Wait for the transcription thread to finish
    if live_transcribe_thread is not None:
        live_transcribe_thread.join()
        live_transcribe_thread = None

    live_transcribe_button.configure(state=tk.NORMAL)
    stop_live_button.configure(state=tk.DISABLED)
    text_output.insert(tk.END, "\nLive transcription stopped.")
    app.update()

def process_live_audio():
    global live_transcribing
    buffer_duration = 5  # Seconds
    buffer_size = int(16000 * buffer_duration)
    audio_buffer = np.zeros((0, 1), dtype=np.float32)

    selected_model = model_var.get()
    device = 'cuda' if use_cuda_var.get() and torch.cuda.is_available() else 'cpu'

    if keep_model_loaded_var.get() and preloaded_model is not None:
        model = preloaded_model
    else:
        model = whisper.load_model(selected_model)
        model = model.to(device)

    while not stop_event.is_set():
        try:
            # Collect audio data from the queue
            while not live_audio_queue.empty():
                data = live_audio_queue.get()
                audio_buffer = np.vstack((audio_buffer, data))

            # If buffer is large enough, transcribe
            if len(audio_buffer) >= buffer_size:
                # Prepare audio chunk
                audio_chunk = audio_buffer[:buffer_size]
                audio_buffer = audio_buffer[buffer_size:]

                # Convert audio to tensor and move to device
                audio_tensor = torch.from_numpy(audio_chunk.flatten()).float().to(device)

                # Transcribe audio chunk
                result = model.transcribe(audio_tensor, fp16=device=='cuda')
                transcription = result['text']

                # Update text output in the main thread
                text_output.insert(tk.END, transcription + " ")
                text_output.see(tk.END)
                app.update()
            else:
                # Wait a bit before checking again
                threading.Event().wait(0.1)
        except Exception as e:
            if not stop_event.is_set():
                messagebox.showerror("Error", str(e))
            break

    # Process any remaining audio in the buffer
    if len(audio_buffer) > 0:
        try:
            audio_tensor = torch.from_numpy(audio_buffer.flatten()).float().to(device)
            result = model.transcribe(audio_tensor, fp16=device=='cuda')
            transcription = result['text']
            text_output.insert(tk.END, transcription)
            text_output.see(tk.END)
            app.update()
        except Exception as e:
            if not stop_event.is_set():
                messagebox.showerror("Error", str(e))

# Initialize the main application window
app = customtkinter.CTk()
app.title("Whisper Mic App")

# Model selection dropdown and checkboxes at the top
model_var = tk.StringVar(value="base")
model_dropdown = customtkinter.CTkOptionMenu(
    app, variable=model_var, values=["tiny", "base", "small", "medium", "large", "turbo"], command=on_model_change
)
model_dropdown.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

keep_model_loaded_var = tk.BooleanVar()
keep_model_loaded_var.set(False)
keep_model_loaded_checkbox = customtkinter.CTkCheckBox(
    app, text="Keep model loaded", variable=keep_model_loaded_var, command=on_keep_model_loaded_change
)
keep_model_loaded_checkbox.grid(row=1, column=0, pady=5, padx=10, sticky="w")

use_cuda_var = tk.BooleanVar()
use_cuda_var.set(torch.cuda.is_available())
use_cuda_checkbox = customtkinter.CTkCheckBox(
    app, text="Use CUDA (GPU)", variable=use_cuda_var, command=on_use_cuda_change
)
use_cuda_checkbox.grid(row=1, column=1, pady=5, padx=10, sticky="w")

# Create frames for grid layout
button_frame = customtkinter.CTkFrame(app)
text_frame = customtkinter.CTkFrame(app)

button_frame.grid(row=2, column=0, sticky="ns")
text_frame.grid(row=2, column=1, sticky="nsew")

# Configure grid weights for responsiveness
app.grid_rowconfigure(2, weight=1)
app.grid_columnconfigure(1, weight=1)

# Add buttons to the left frame
listen_button = customtkinter.CTkButton(button_frame, text="Listen")
listen_button.pack(pady=5, padx=10, fill="x")
listen_button.bind('<ButtonPress-1>', start_recording)
listen_button.bind('<ButtonRelease-1>', stop_recording)

start_meeting_button = customtkinter.CTkButton(button_frame, text="Start Meeting", command=start_meeting)
start_meeting_button.pack(pady=5, padx=10, fill="x")

end_meeting_button = customtkinter.CTkButton(button_frame, text="End Meeting", command=end_meeting)
end_meeting_button.pack(pady=5, padx=10, fill="x")

live_transcribe_button = customtkinter.CTkButton(button_frame, text="Live Transcribe", command=start_live_transcription)
live_transcribe_button.pack(pady=5, padx=10, fill="x")

stop_live_button = customtkinter.CTkButton(button_frame, text="Stop Live", command=stop_live_transcription)
stop_live_button.pack(pady=5, padx=10, fill="x")
stop_live_button.configure(state=tk.DISABLED)

copy_button = customtkinter.CTkButton(button_frame, text="Copy", command=copy_text)
copy_button.pack(pady=5, padx=10, fill="x")

# Add text box to the right frame
text_output = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, bg="#333", fg="white")
text_output.pack(fill="both", expand=True, padx=10, pady=10)

# Load model if checkbox is checked at startup
on_keep_model_loaded_change()

app.mainloop()
