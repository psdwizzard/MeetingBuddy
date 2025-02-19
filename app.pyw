import customtkinter
import tkinter as tk
from tkinter import scrolledtext, messagebox
from tkinter import Listbox, END, SINGLE
import threading
import whisper
import sounddevice as sd
import numpy as np
import pyperclip
import os
import datetime
import torch
import queue
import requests
import json  # For persistent settings
import openai  # For OpenAI API usage
import re    # For regex processing

# Set appearance mode and color theme
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

# -------------------- Global Variables --------------------
recording_stream = None
audio_data = []
is_listening = False
meeting_running = False
meeting_audio_data = []
meeting_start_time = None
meeting_thread = None
preloaded_model = None  # For Whisper model (transcription)
live_transcribing = False
live_audio_queue = queue.Queue()
live_transcribe_thread = None
stop_event = threading.Event()

advanced_meeting_running = False
advanced_meeting_audio_data = []
advanced_meeting_start_time = None
advanced_meeting_thread = None

# -------------------- Helper Functions --------------------
def remove_think_tags(text):
    """Remove any text between <think> and </think> tags."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

# -------------------- Ollama API Integration --------------------
def get_ollama_base_url():
    if default_use_openai_api_var.get() and default_openai_api_key_var.get().strip():
        return "http://localhost:11434"
    ip = default_ollama_ip_var.get().strip()
    return f"http://{ip}:11434" if ip else "http://localhost:11434"

def load_ollama_models():
    try:
        url = get_ollama_base_url() + "/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        models = [item["name"] for item in data.get("models", [])]
        return models
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load Ollama models: {e}")
        return []

def load_ollama_model_by_name(model_name):
    try:
        url = get_ollama_base_url() + "/api/generate"
        payload = {"model": model_name, "keep_alive": -1}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Ollama model '{model_name}' loaded into memory.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load Ollama model '{model_name}': {e}")

def unload_ollama_model(model_name):
    try:
        url = get_ollama_base_url() + "/api/generate"
        payload = {"model": model_name, "keep_alive": 0}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Ollama model '{model_name}' unloaded from memory.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to unload Ollama model '{model_name}': {e}")

def ollama_summarize_meeting(transcription):
    try:
        cleaned_text = remove_think_tags(transcription)
        prompt = ("You're the world's best stenographer and note taker for a fortune 500 company. "
                  "Please provide a detailed summary meeting with as much accuracy as possible. Use the following format:\n\n"
                  "Summary:\nOne paragraph overview.\n\n"
                  "Key Take Aways:\n- bullet point\n- bullet point\n\n"
                  "Action Items:\n- bullet point\n- bullet point\n\n"
                  "Transcript:\n" + cleaned_text)
        url = get_ollama_base_url() + "/api/generate"
        payload = {
            "model": ollama_model_var.get(),
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        summary = data.get("response", "No summary provided.")
        return remove_think_tags(summary)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to summarize meeting: {e}")
        return "Error in summarization."

# -------------------- OpenAI API Functions --------------------
def get_openai_summary(text):
    cleaned_text = remove_think_tags(text)
    openai.api_key = default_openai_api_key_var.get()
    prompt = ("You're the world's best stenographer and note taker for a fortune 500 company. "
              "Please provide a detailed summary meeting with as much accuracy as possible. Use the following format:\n\n"
              "Summary:\nOne paragraph overview.\n\n"
              "Key Take Aways:\n- bullet point\n- bullet point\n\n"
              "Action Items:\n- bullet point\n- bullet point\n\n"
              "Transcript:\n" + cleaned_text)
    response = openai.ChatCompletion.create(
        model=default_openai_model_var.get(),
        messages=[{"role": "system", "content": prompt}]
    )
    return remove_think_tags(response.choices[0].message.content)

# -------------------- Transcript Splitting Functions --------------------
def split_transcript_into_segments_with_times(transcript):
    # Using a threshold of 1900 tokens per chunk to stay below LLM limits
    TOKEN_LIMIT = 1900
    lines = transcript.splitlines()
    segments = []
    current_segment = []
    current_word_count = 0
    current_start = None
    current_end = None
    for line in lines:
        if line.startswith('[') and ' - ' in line:
            try:
                ts_range = line.split(']')[0][1:]
                start_str, end_str = ts_range.split('-')
                start_str = start_str.strip()
                end_str = end_str.strip()
                word_count = len(line.split())
                if current_start is None:
                    current_start = start_str
                current_end = end_str
            except Exception:
                word_count = len(line.split())
        else:
            word_count = len(line.split())
        if current_word_count + word_count > TOKEN_LIMIT and current_segment:
            segments.append(("\n".join(current_segment), current_start, current_end))
            current_segment = [line]
            current_word_count = word_count
            if line.startswith('[') and ' - ' in line:
                try:
                    ts_range = line.split(']')[0][1:]
                    start_str, end_str = ts_range.split('-')
                    current_start = start_str.strip()
                    current_end = end_str.strip()
                except Exception:
                    pass
        else:
            current_segment.append(line)
            current_word_count += word_count
    if current_segment:
        segments.append(("\n".join(current_segment), current_start if current_start else "00:00:00", current_end if current_end else ""))
    return segments

# -------------------- Local Summarization Functions --------------------
def summarize_text_local(text):
    segments = split_transcript_into_segments_with_times(text)
    bullet_points_list = []
    model_name = ollama_model_var.get()
    # Load the model once for all chunks.
    load_ollama_model_by_name(model_name)
    chunk_index = 1
    for segment_text, start_time, end_time in segments:
        prompt = (f"Transcript Segment {chunk_index}:\n{segment_text}\n\n"
                  "Provide detailed bullet points summarizing the key takeaways from this segment. "
                  "Return one bullet point per line without any extra headers or numbering.")
        seg_bullets = ollama_summarize_meeting(prompt)
        if seg_bullets.strip():
            bullet_points_list.append(seg_bullets.strip())
        chunk_index += 1
    # Concatenate bullet points from all chunks.
    all_bullet_points = "\n".join(bullet_points_list)
    # Generate an executive summary based on the aggregated bullet points.
    exec_prompt = ("Based on the following bullet points, provide a concise executive summary in one paragraph, "
                   "followed by key action items as bullet points. Do not include any additional headings in your response.\n\n"
                   "Bullet Points:\n" + all_bullet_points)
    executive_summary = ollama_summarize_meeting(exec_prompt)
    unload_ollama_model(model_name)
    final_text = "Key Takeaways:\n" + all_bullet_points + "\n\nExecutive Summary:\n" + executive_summary
    return final_text

# -------------------- Summarization Button Function --------------------
def summarize_current_text():
    text = text_output.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "No transcript available to summarize.")
        return
    try:
        if default_use_openai_api_var.get() and default_openai_api_key_var.get().strip():
            summary = get_openai_summary(text)
        else:
            unload_model()
            summary = summarize_text_local(text)
            if keep_model_loaded_var.get():
                load_model(model_var.get())
        text_output.insert(tk.END, "\n\nSummary:\n" + summary)
    except Exception as e:
        messagebox.showerror("Error", f"Summarization failed: {e}")

# -------------------- Helper Function for Timestamped Transcript --------------------
def format_transcript_with_timestamps(result):
    transcript = ""
    for segment in result.get("segments", []):
        start_time = str(datetime.timedelta(seconds=int(segment["start"])))
        end_time = str(datetime.timedelta(seconds=int(segment["end"])))
        transcript += f"[{start_time} - {end_time}] {segment['text']}\n"
    return transcript

# -------------------- Whisper Model and Transcription Functions --------------------
def load_model(model_name):
    global preloaded_model
    print(f"Loading Whisper model: {model_name}")
    device = 'cuda' if torch.cuda.is_available() and use_cuda_var.get() else 'cpu'
    preloaded_model = whisper.load_model(model_name, device=device)

def unload_model():
    global preloaded_model
    if preloaded_model is not None:
        print("Unloading Whisper model")
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
    if keep_model_loaded_var.get():
        load_model(model_var.get())

def start_recording(event=None):
    global recording_stream, audio_data, is_listening
    if is_listening:
        return
    is_listening = True
    listen_button.configure(text="Listening...")
    meeting_buddy_tab.update()
    fs = 16000
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
    meeting_buddy_tab.update()
    is_listening = False
    audio = np.concatenate(audio_data, axis=0)
    threading.Thread(target=transcribe_audio, args=(audio, False, None, False)).start()

def transcribe_audio(audio, save_to_file=False, filename=None, apply_timestamps=True):
    try:
        disable_buttons()
        text_output.delete("1.0", tk.END)
        text_output.insert("1.0", "Transcribing...")
        meeting_buddy_tab.update()
        selected_model = model_var.get()
        device = 'cuda' if torch.cuda.is_available() and use_cuda_var.get() else 'cpu'
        if keep_model_loaded_var.get() and preloaded_model is not None:
            model = preloaded_model
        else:
            model = whisper.load_model(selected_model, device=device)
        audio_tensor = torch.from_numpy(audio).float().to(device)
        result = model.transcribe(audio_tensor.flatten(), fp16=(device=='cuda'))
        transcription = format_transcript_with_timestamps(result) if apply_timestamps else result['text']
        text_output.delete("1.0", tk.END)
        text_output.insert("1.0", transcription)
        if save_to_file and filename:
            save_transcription(transcription, filename)
        enable_buttons()
    except Exception as e:
        enable_buttons()
        messagebox.showerror("Error", str(e))

def copy_text():
    text = text_output.get("1.0", tk.END).strip()
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
    text_output.delete("1.0", tk.END)
    text_output.insert("1.0", "Meeting started...")
    meeting_buddy_tab.update()
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
    meeting_buddy_tab.update()

def record_meeting():
    global meeting_audio_data, meeting_start_time, meeting_running
    fs = 16000
    def callback(indata, frames, time, status):
        meeting_audio_data.append(indata.copy())
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        start_time = datetime.datetime.now()
        while meeting_running:
            if (datetime.datetime.now() - start_time).total_seconds() >= 7200:
                meeting_running = False
                break
            sd.sleep(1000)
    audio = np.concatenate(meeting_audio_data, axis=0)
    meeting_name = meeting_name_var.get().strip()
    if meeting_name == "":
        filename = meeting_start_time.strftime("%m-%d-%Y-%H%M")
    else:
        filename = meeting_name + "_" + meeting_start_time.strftime("%m-%d-%Y-%H%M")
    filepath = os.path.join("meetings", filename + ".txt")
    threading.Thread(target=transcribe_audio, args=(audio, True, filepath)).start()

def save_transcription(transcription, filepath):
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
    text_output.delete("1.0", tk.END)
    text_output.insert(tk.END, "Live transcription started...")
    meeting_buddy_tab.update()
    fs = 16000
    def audio_callback(indata, frames, time, status):
        live_audio_queue.put(indata.copy())
    recording_stream = sd.InputStream(samplerate=fs, channels=1, callback=audio_callback)
    recording_stream.start()
    live_transcribe_thread = threading.Thread(target=process_live_audio)
    live_transcribe_thread.start()

def stop_live_transcription():
    global live_transcribing, recording_stream, stop_event, live_transcribe_thread
    if not live_transcribing:
        messagebox.showwarning("Warning", "No live transcription is running.")
        return
    live_transcribing = False
    stop_event.set()
    if recording_stream is not None:
        recording_stream.stop()
        recording_stream.close()
        recording_stream = None
    def check_thread():
        if live_transcribe_thread is not None and live_transcribe_thread.is_alive():
            app.after(100, check_thread)
        else:
            live_transcribe_button.configure(state=tk.NORMAL)
            stop_live_button.configure(state=tk.DISABLED)
            text_output.insert(tk.END, "\nLive transcription stopped.")
            meeting_buddy_tab.update()
    check_thread()

def process_live_audio():
    global live_transcribing
    buffer_duration = 5  # seconds
    buffer_size = int(16000 * buffer_duration)
    audio_buffer = np.zeros((0, 1), dtype=np.float32)
    selected_model = model_var.get()
    device = 'cuda' if torch.cuda.is_available() and use_cuda_var.get() else 'cpu'
    if keep_model_loaded_var.get() and preloaded_model is not None:
        model = preloaded_model
    else:
        model = whisper.load_model(selected_model, device=device)
    while not stop_event.is_set():
        try:
            while not live_audio_queue.empty():
                data = live_audio_queue.get()
                audio_buffer = np.vstack((audio_buffer, data))
            if len(audio_buffer) >= buffer_size:
                audio_chunk = audio_buffer[:buffer_size]
                audio_buffer = audio_buffer[buffer_size:]
                audio_tensor = torch.from_numpy(audio_chunk.flatten()).float().to(device)
                result = model.transcribe(audio_tensor, fp16=(device=='cuda'))
                transcription = format_transcript_with_timestamps(result)
                text_output.insert(tk.END, transcription + " ")
                text_output.see(tk.END)
                meeting_buddy_tab.update()
            else:
                threading.Event().wait(0.05)
        except Exception as e:
            if not stop_event.is_set():
                messagebox.showerror("Error", str(e))
            break
    if len(audio_buffer) > 0:
        try:
            audio_tensor = torch.from_numpy(audio_buffer.flatten()).float().to(device)
            result = model.transcribe(audio_tensor, fp16=(device=='cuda'))
            transcription = result['text']
            text_output.insert(tk.END, transcription)
            text_output.see(tk.END)
            meeting_buddy_tab.update()
        except Exception as e:
            if not stop_event.is_set():
                messagebox.showerror("Error", str(e))

def start_advanced_meeting():
    global advanced_meeting_running, advanced_meeting_audio_data, advanced_meeting_start_time, advanced_meeting_thread
    if advanced_meeting_running:
        messagebox.showwarning("Warning", "Advanced meeting is already running.")
        return
    advanced_meeting_running = True
    advanced_meeting_start_time = datetime.datetime.now()
    advanced_meeting_audio_data = []
    start_advanced_button.configure(state=tk.DISABLED)
    end_advanced_button.configure(state=tk.NORMAL)
    text_output.delete("1.0", tk.END)
    text_output.insert("1.0", "Advanced Meeting started...")
    meeting_buddy_tab.update()
    advanced_meeting_thread = threading.Thread(target=record_advanced_meeting)
    advanced_meeting_thread.start()

def end_advanced_meeting():
    global advanced_meeting_running
    if not advanced_meeting_running:
        messagebox.showwarning("Warning", "No advanced meeting is currently running.")
        return
    advanced_meeting_running = False
    end_advanced_button.configure(state=tk.DISABLED)
    start_advanced_button.configure(state=tk.NORMAL)
    text_output.insert(tk.END, "\nAdvanced Meeting ended. Transcribing and summarizing...")
    meeting_buddy_tab.update()

def record_advanced_meeting():
    global advanced_meeting_audio_data, advanced_meeting_running
    fs = 16000
    def callback(indata, frames, time, status):
        advanced_meeting_audio_data.append(indata.copy())
    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        start_time = datetime.datetime.now()
        while advanced_meeting_running:
            if (datetime.datetime.now() - start_time).total_seconds() >= 7200:
                advanced_meeting_running = False
                break
            sd.sleep(1000)
    audio = np.concatenate(advanced_meeting_audio_data, axis=0)
    selected_model = model_var.get()
    device = 'cuda' if torch.cuda.is_available() and use_cuda_var.get() else 'cpu'
    if keep_model_loaded_var.get() and preloaded_model is not None:
        model = preloaded_model
    else:
        model = whisper.load_model(selected_model, device=device)
    audio_tensor = torch.from_numpy(audio).float().to(device)
    result = model.transcribe(audio_tensor.flatten(), fp16=(device=='cuda'))
    transcription = format_transcript_with_timestamps(result)
    if not (default_use_openai_api_var.get() and default_openai_api_key_var.get().strip()):
        if not keep_model_loaded_var.get():
            unload_model()
        else:
            load_model(model_var.get())
    if default_use_openai_api_var.get() and default_openai_api_key_var.get().strip():
        summary = get_openai_summary(transcription)
    else:
        summary = summarize_text_local(transcription)
    meeting_name = meeting_name_var.get().strip()
    if meeting_name == "":
        base_filename = advanced_meeting_start_time.strftime("%m-%d-%Y-%H%M")
    else:
        base_filename = meeting_name + "_" + advanced_meeting_start_time.strftime("%m-%d-%Y-%H%M")
    adv_filename = base_filename + "_advanced.txt"
    adv_filepath = os.path.join("meetings", adv_filename)
    final_text = "Transcript:\n" + transcription + "\n\nSummary:\n" + summary
    save_transcription(final_text, adv_filepath)
    text_output.delete("1.0", tk.END)
    text_output.insert("1.0", final_text)

def process_advanced_meeting():
    try:
        record_advanced_meeting()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# -------------------- Persistent Settings Functions --------------------
def load_persistent_settings():
    if os.path.exists("settings.json"):
        try:
            with open("settings.json", "r") as f:
                data = json.load(f)
            return data
        except Exception as e:
            messagebox.showerror("Error", f"Error loading settings: {e}")
            return {
                "default_model": "base",
                "default_keep_model_loaded": False,
                "default_ollama_model": "Select Ollama Model",
                "default_ollama_ip": "",
                "default_use_openai_api": False,
                "default_openai_api_key": "",
                "default_openai_model": "gpt-3.5-turbo"
            }
    else:
        return {
            "default_model": "base",
            "default_keep_model_loaded": False,
            "default_ollama_model": "Select Ollama Model",
            "default_ollama_ip": "",
            "default_use_openai_api": False,
            "default_openai_api_key": "",
            "default_openai_model": "gpt-3.5-turbo"
        }

def save_persistent_settings():
    settings_data = {
        "default_model": default_model_var.get(),
        "default_keep_model_loaded": default_keep_model_loaded_var.get(),
        "default_ollama_model": default_ollama_model_var.get(),
        "default_ollama_ip": default_ollama_ip_var.get(),
        "default_use_openai_api": default_use_openai_api_var.get(),
        "default_openai_api_key": default_openai_api_key_var.get(),
        "default_openai_model": default_openai_model_var.get()
    }
    try:
        with open("settings.json", "w") as f:
            json.dump(settings_data, f)
    except Exception as e:
        messagebox.showerror("Error", f"Error saving settings: {e}")

def save_settings():
    model_var.set(default_model_var.get())
    keep_model_loaded_var.set(default_keep_model_loaded_var.get())
    ollama_model_var.set(default_ollama_model_var.get())
    save_persistent_settings()
    on_keep_model_loaded_change()
    messagebox.showinfo("Settings Saved", "Default settings have been updated.")

# -------------------- Old Meetings Tab Functions --------------------
def update_old_meetings_list():
    """Load meeting files from the 'meetings' folder into the listbox."""
    old_meetings_listbox.delete(0, END)
    if not os.path.exists("meetings"):
        os.makedirs("meetings", exist_ok=True)
    files = sorted(os.listdir("meetings"))
    for file in files:
        if file.endswith(".txt"):
            old_meetings_listbox.insert(END, file)

def load_selected_meeting(event=None):
    """Load the content of the selected meeting file into the old meetings text area."""
    selected = old_meetings_listbox.curselection()
    if selected:
        filename = old_meetings_listbox.get(selected[0])
        filepath = os.path.join("meetings", filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            old_meeting_text.delete("1.0", tk.END)
            old_meeting_text.insert("1.0", content)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load meeting: {e}")

def rename_meeting():
    """Rename the selected meeting file using the name from the meeting name entry plus its original date suffix."""
    selected = old_meetings_listbox.curselection()
    if not selected:
        messagebox.showwarning("Warning", "No meeting selected for renaming.")
        return
    original_filename = old_meetings_listbox.get(selected[0])
    new_name = old_meeting_name_var.get().strip()
    if new_name == "":
        messagebox.showwarning("Warning", "Please enter a new meeting name in the entry.")
        return
    parts = original_filename.split("_")
    if len(parts) >= 2:
        date_part = parts[-1].replace(".txt", "")
        suffix = ""
        if "advanced" in original_filename:
            suffix = "_advanced"
        new_filename = new_name + "_" + date_part + suffix + ".txt"
    else:
        new_filename = new_name + ".txt"
    original_filepath = os.path.join("meetings", original_filename)
    new_filepath = os.path.join("meetings", new_filename)
    try:
        os.rename(original_filepath, new_filepath)
        update_old_meetings_list()
        messagebox.showinfo("Renamed", f"Meeting renamed to {new_filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to rename meeting: {e}")

def summarize_old_meeting():
    """Append a summary to the content of the old meeting text area."""
    content = old_meeting_text.get("1.0", tk.END).strip()
    if not content:
        messagebox.showwarning("Warning", "No meeting content to summarize.")
        return
    try:
        if default_use_openai_api_var.get() and default_openai_api_key_var.get().strip():
            summary = get_openai_summary(content)
        else:
            unload_model()
            summary = summarize_text_local(content)
            if keep_model_loaded_var.get():
                load_model(model_var.get())
        # Append the summary to the existing content
        old_meeting_text.insert(tk.END, "\n\nSummary:\n" + summary)
    except Exception as e:
        messagebox.showerror("Error", f"Summarization failed: {e}")

def save_old_meeting():
    """Save the changes in the old meeting text area back to the selected file."""
    selected = old_meetings_listbox.curselection()
    if not selected:
        messagebox.showwarning("Warning", "No meeting selected to save.")
        return
    filename = old_meetings_listbox.get(selected[0])
    filepath = os.path.join("meetings", filename)
    content = old_meeting_text.get("1.0", tk.END)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        messagebox.showinfo("Saved", f"Changes saved to {filename}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save meeting: {e}")

# -------------------- Main Application Window and Tab Layout --------------------
app = customtkinter.CTk()
app.title("Whisper Mic App with Ollama and OpenAI Integration")

default_model_var = tk.StringVar(app, value="base")
default_keep_model_loaded_var = tk.BooleanVar(app, value=False)
default_ollama_model_var = tk.StringVar(app, value="Select Ollama Model")
default_ollama_ip_var = tk.StringVar(app, value="")  # For custom IP
default_use_openai_api_var = tk.BooleanVar(app, value=False)
default_openai_api_key_var = tk.StringVar(app, value="")
default_openai_model_var = tk.StringVar(app, value="gpt-3.5-turbo")

persisted = load_persistent_settings()
default_model_var.set(persisted.get("default_model", "base"))
default_keep_model_loaded_var.set(persisted.get("default_keep_model_loaded", False))
default_ollama_model_var.set(persisted.get("default_ollama_model", "Select Ollama Model"))
default_ollama_ip_var.set(persisted.get("default_ollama_ip", ""))
default_use_openai_api_var.set(persisted.get("default_use_openai_api", False))
default_openai_api_key_var.set(persisted.get("default_openai_api_key", ""))
default_openai_model_var.set(persisted.get("default_openai_model", "gpt-3.5-turbo"))

tabview = customtkinter.CTkTabview(app, width=1000, height=600)
tabview.pack(fill="both", expand=True, padx=10, pady=10)
tabview.add("Meeting Buddy")
tabview.add("Old Meetings")
tabview.add("Settings")

# -------------------- Meeting Buddy Tab --------------------
meeting_buddy_tab = tabview.tab("Meeting Buddy")
model_var = tk.StringVar(meeting_buddy_tab, value=default_model_var.get())
model_dropdown = customtkinter.CTkOptionMenu(
    meeting_buddy_tab, variable=model_var,
    values=["tiny", "base", "small", "medium", "large", "turbo"],
    command=on_model_change
)
model_dropdown.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
keep_model_loaded_var = tk.BooleanVar(meeting_buddy_tab, value=default_keep_model_loaded_var.get())
keep_model_loaded_checkbox = customtkinter.CTkCheckBox(
    meeting_buddy_tab, text="Keep model loaded (Whisper)",
    variable=keep_model_loaded_var, command=on_keep_model_loaded_change
)
keep_model_loaded_checkbox.grid(row=1, column=0, pady=5, padx=10, sticky="w")
use_cuda_var = tk.BooleanVar()
use_cuda_var.set(torch.cuda.is_available())
use_cuda_checkbox = customtkinter.CTkCheckBox(
    meeting_buddy_tab, text="Use CUDA (GPU)",
    variable=use_cuda_var, command=on_use_cuda_change
)
use_cuda_checkbox.grid(row=1, column=1, pady=5, padx=10, sticky="w")
load_ollama_models_button = customtkinter.CTkButton(
    meeting_buddy_tab, text="Load Ollama Models", command=lambda: populate_ollama_models()
)
load_ollama_models_button.grid(row=1, column=2, pady=5, padx=10, sticky="w")
ollama_model_var = tk.StringVar(meeting_buddy_tab, value=default_ollama_model_var.get())
ollama_model_dropdown = customtkinter.CTkOptionMenu(
    meeting_buddy_tab, variable=ollama_model_var, values=[]
)
ollama_model_dropdown.grid(row=1, column=3, pady=5, padx=10, sticky="w")

# Frames for Meeting Buddy tab
mb_button_frame = customtkinter.CTkFrame(meeting_buddy_tab)
mb_text_frame = customtkinter.CTkFrame(meeting_buddy_tab)
mb_button_frame.grid(row=2, column=0, sticky="ns")
mb_text_frame.grid(row=2, column=1, sticky="nsew", columnspan=3)
meeting_buddy_tab.grid_rowconfigure(2, weight=1)
meeting_buddy_tab.grid_columnconfigure(1, weight=1)

# Meeting Buddy controls
listen_button = customtkinter.CTkButton(mb_button_frame, text="Listen")
listen_button.pack(pady=5, padx=10, fill="x")
listen_button.bind('<ButtonPress-1>', start_recording)
listen_button.bind('<ButtonRelease-1>', stop_recording)
start_meeting_button = customtkinter.CTkButton(mb_button_frame, text="Start Meeting", command=start_meeting)
start_meeting_button.pack(pady=5, padx=10, fill="x")
end_meeting_button = customtkinter.CTkButton(mb_button_frame, text="End Meeting", command=end_meeting)
end_meeting_button.pack(pady=5, padx=10, fill="x")
live_transcribe_button = customtkinter.CTkButton(mb_button_frame, text="Live Transcribe", command=start_live_transcription)
live_transcribe_button.pack(pady=5, padx=10, fill="x")
stop_live_button = customtkinter.CTkButton(mb_button_frame, text="Stop Live", command=stop_live_transcription)
stop_live_button.pack(pady=5, padx=10, fill="x")
stop_live_button.configure(state=tk.DISABLED)
copy_button = customtkinter.CTkButton(mb_button_frame, text="Copy", command=copy_text)
copy_button.pack(pady=5, padx=10, fill="x")
start_advanced_button = customtkinter.CTkButton(mb_button_frame, text="Start Advanced Meeting", command=start_advanced_meeting)
start_advanced_button.pack(pady=5, padx=10, fill="x")
end_advanced_button = customtkinter.CTkButton(mb_button_frame, text="End Advanced Meeting", command=end_advanced_meeting)
end_advanced_button.pack(pady=5, padx=10, fill="x")
end_advanced_button.configure(state=tk.DISABLED)
summarize_button = customtkinter.CTkButton(mb_button_frame, text="Summarize", command=summarize_current_text)
summarize_button.pack(pady=5, padx=10, fill="x")
# Meeting Name input at the bottom of Meeting Buddy controls
meeting_name_label = customtkinter.CTkLabel(mb_button_frame, text="Meeting Name:")
meeting_name_label.pack(pady=(10, 0), padx=10, fill="x")
meeting_name_var = tk.StringVar(mb_button_frame, value="")  # Default empty
meeting_name_entry = customtkinter.CTkEntry(mb_button_frame, textvariable=meeting_name_var)
meeting_name_entry.pack(pady=(0, 10), padx=10, fill="x")

text_output = scrolledtext.ScrolledText(mb_text_frame, wrap=tk.WORD, bg="#333", fg="white")
text_output.pack(fill="both", expand=True, padx=10, pady=10)

# -------------------- Old Meetings Tab --------------------
old_meetings_tab = tabview.tab("Old Meetings")
# Left frame: List of meetings (narrowed to 200 pixels)
old_list_frame = customtkinter.CTkFrame(old_meetings_tab, width=200)
old_list_frame.pack(side=tk.LEFT, fill="y", padx=10, pady=10)
old_meetings_listbox = tk.Listbox(old_list_frame, selectmode=SINGLE)
old_meetings_listbox.pack(fill="both", expand=True, padx=5, pady=5)
old_meetings_listbox.bind("<<ListboxSelect>>", load_selected_meeting)
# Bottom frame for controls: buttons stacked vertically
old_controls_frame = customtkinter.CTkFrame(old_list_frame)
old_controls_frame.pack(fill="x", padx=5, pady=5)
rename_button = customtkinter.CTkButton(old_controls_frame, text="Rename", command=rename_meeting)
rename_button.pack(fill="x", pady=2, padx=2)
summarize_old_button = customtkinter.CTkButton(old_controls_frame, text="Summarize", command=summarize_old_meeting)
summarize_old_button.pack(fill="x", pady=2, padx=2)
save_old_button = customtkinter.CTkButton(old_controls_frame, text="Save", command=save_old_meeting)
save_old_button.pack(fill="x", pady=2, padx=2)
# Meeting Name entry for renaming in Old Meetings
old_meeting_name_var = tk.StringVar(old_controls_frame, value="")
old_meeting_name_entry = customtkinter.CTkEntry(old_controls_frame, textvariable=old_meeting_name_var)
old_meeting_name_entry.pack(fill="x", pady=2, padx=2)

# Right frame: Text area for viewing/editing the selected meeting
old_text_frame = customtkinter.CTkFrame(old_meetings_tab)
old_text_frame.pack(side=tk.RIGHT, fill="both", expand=True, padx=10, pady=10)
old_meeting_text = scrolledtext.ScrolledText(old_text_frame, wrap=tk.WORD, bg="#333", fg="white")
old_meeting_text.pack(fill="both", expand=True, padx=5, pady=5)
update_old_meetings_list()

# -------------------- Settings Tab --------------------
settings_tab = tabview.tab("Settings")
default_model_label = customtkinter.CTkLabel(settings_tab, text="Default Whisper Model:")
default_model_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
default_model_menu = customtkinter.CTkOptionMenu(
    settings_tab, variable=default_model_var,
    values=["tiny", "base", "small", "medium", "large", "turbo"]
)
default_model_menu.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
default_keep_checkbox = customtkinter.CTkCheckBox(
    settings_tab, text="Keep default model loaded permanently",
    variable=default_keep_model_loaded_var
)
default_keep_checkbox.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="w")
default_ollama_label = customtkinter.CTkLabel(settings_tab, text="Default Ollama Model:")
default_ollama_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
ollama_options = load_ollama_models()
if not ollama_options:
    ollama_options = ["Select Ollama Model"]
default_ollama_model_var.set(default_ollama_model_var.get() if default_ollama_model_var.get() in ollama_options else ollama_options[0])
default_ollama_menu = customtkinter.CTkOptionMenu(
    settings_tab, variable=default_ollama_model_var,
    values=ollama_options
)
default_ollama_menu.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
default_ollama_ip_label = customtkinter.CTkLabel(settings_tab, text="Ollama IP Address (optional):")
default_ollama_ip_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
default_ollama_ip_entry = customtkinter.CTkEntry(settings_tab, textvariable=default_ollama_ip_var)
default_ollama_ip_entry.grid(row=3, column=1, padx=10, pady=10, sticky="ew")
openai_frame = customtkinter.CTkFrame(settings_tab)
openai_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
openai_title = customtkinter.CTkLabel(openai_frame, text="OpenAI Settings", font=("Arial", 16))
openai_title.grid(row=0, column=0, columnspan=2, pady=(0,10))
default_use_openai_checkbox = customtkinter.CTkCheckBox(
    openai_frame, text="Use non-local OpenAI API",
    variable=default_use_openai_api_var
)
default_use_openai_checkbox.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="w")
openai_key_label = customtkinter.CTkLabel(openai_frame, text="OpenAI API Key:")
openai_key_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
openai_key_entry = customtkinter.CTkEntry(openai_frame, textvariable=default_openai_api_key_var)
openai_key_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
openai_model_label = customtkinter.CTkLabel(openai_frame, text="OpenAI Model:")
openai_model_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
openai_model_menu = customtkinter.CTkOptionMenu(
    openai_frame, variable=default_openai_model_var,
    values=["gpt-3.5-turbo", "gpt-4"]
)
openai_model_menu.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
save_settings_button = customtkinter.CTkButton(settings_tab, text="Save Settings", command=save_settings)
save_settings_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

# Preload Whisper model if enabled
on_keep_model_loaded_change()

def populate_ollama_models():
    models = load_ollama_models()
    if models:
        ollama_model_dropdown.configure(values=models)
        if default_ollama_model_var.get() in models:
            ollama_model_var.set(default_ollama_model_var.get())
        else:
            ollama_model_var.set(models[0])
        messagebox.showinfo("Loaded", "Ollama models loaded successfully.")

app.mainloop()
