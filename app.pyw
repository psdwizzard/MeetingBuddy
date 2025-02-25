import customtkinter
import tkinter as tk
from tkinter import scrolledtext, messagebox, Listbox, END, SINGLE, filedialog
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
import json
import openai
import re
import wave
from fpdf import FPDF
from docx import Document

# -------------------- Helper Functions --------------------
def remove_think_tags(text):
    """Remove any text between <think> and </think> tags."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

# -------------------- Audio Saving Function --------------------
def save_audio(audio, filepath, samplerate=16000):
    try:
        # Normalize audio to 16-bit PCM format
        audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bits = 2 bytes
            wf.setframerate(samplerate)
            wf.writeframes(audio_int16.tobytes())
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save audio: {e}")

# -------------------- Ollama API Integration Functions --------------------
def get_ollama_base_url(use_openai, openai_api_key, ollama_ip):
    if use_openai and openai_api_key.strip():
        return "http://localhost:11434"
    ip = ollama_ip.strip()
    return f"http://{ip}:11434" if ip else "http://localhost:11434"

def load_ollama_models(use_openai, openai_api_key, ollama_ip):
    try:
        url = get_ollama_base_url(use_openai, openai_api_key, ollama_ip) + "/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        models = [item["name"] for item in data.get("models", [])]
        return models
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load Ollama models: {e}")
        return []

def load_ollama_model_by_name(model_name, use_openai, openai_api_key, ollama_ip):
    try:
        url = get_ollama_base_url(use_openai, openai_api_key, ollama_ip) + "/api/generate"
        payload = {"model": model_name, "keep_alive": -1}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Ollama model '{model_name}' loaded into memory.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load Ollama model '{model_name}': {e}")

def unload_ollama_model(model_name, use_openai, openai_api_key, ollama_ip):
    try:
        url = get_ollama_base_url(use_openai, openai_api_key, ollama_ip) + "/api/generate"
        payload = {"model": model_name, "keep_alive": 0}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"Ollama model '{model_name}' unloaded from memory.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to unload Ollama model '{model_name}': {e}")

def ollama_summarize_meeting(transcription, ollama_model, use_openai, openai_api_key, ollama_ip):
    try:
        cleaned_text = remove_think_tags(transcription)
        prompt = (
            "You're the world's best stenographer and note taker for a fortune 500 company. "
            "Please provide a detailed summary meeting with as much accuracy as possible. Use the following format:\n\n"
            "Summary:\nOne paragraph overview.\n\n"
            "Key Take Aways:\n- bullet point\n- bullet point\n\n"
            "Action Items:\n- bullet point\n- bullet point\n\n"
            "Transcript:\n" + cleaned_text
        )
        url = get_ollama_base_url(use_openai, openai_api_key, ollama_ip) + "/api/generate"
        payload = {
            "model": ollama_model,
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

def get_openai_summary(text, openai_api_key, openai_model):
    cleaned_text = remove_think_tags(text)
    openai.api_key = openai_api_key
    prompt = (
        "You're the world's best stenographer and note taker for a fortune 500 company. "
        "Please provide a detailed summary meeting with as much accuracy as possible. Use the following format:\n\n"
        "Summary:\nOne paragraph overview.\n\n"
        "Key Take Aways:\n- bullet point\n- bullet point\n\n"
        "Action Items:\n- bullet point\n- bullet point\n\n"
        "Transcript:\n" + cleaned_text
    )
    response = openai.ChatCompletion.create(
        model=openai_model,
        messages=[{"role": "system", "content": prompt}]
    )
    return remove_think_tags(response.choices[0].message.content)

def split_transcript_into_segments_with_times(transcript):
    TOKEN_LIMIT = 1900  # tokens per chunk
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
        segments.append(("\n".join(current_segment),
                         current_start if current_start else "00:00:00",
                         current_end if current_end else ""))
    return segments

def summarize_text_local(text, ollama_model, use_openai, openai_api_key, ollama_ip):
    segments = split_transcript_into_segments_with_times(text)
    bullet_points_list = []
    load_ollama_model_by_name(ollama_model, use_openai, openai_api_key, ollama_ip)
    chunk_index = 1
    for segment_text, start_time, end_time in segments:
        prompt = (
            f"Transcript Segment {chunk_index}:\n{segment_text}\n\n"
            "Provide detailed bullet points summarizing the key takeaways from this segment. "
            "Return one bullet point per line without any extra headers or numbering."
        )
        seg_bullets = ollama_summarize_meeting(prompt, ollama_model, use_openai, openai_api_key, ollama_ip)
        if seg_bullets.strip():
            bullet_points_list.append(seg_bullets.strip())
        chunk_index += 1
    all_bullet_points = "\n".join(bullet_points_list)
    exec_prompt = (
        "Based on the following bullet points, provide a concise executive summary in one paragraph, "
        "followed by key action items as bullet points. Do not include any additional headings in your response.\n\n"
        "Bullet Points:\n" + all_bullet_points
    )
    executive_summary = ollama_summarize_meeting(exec_prompt, ollama_model, use_openai, openai_api_key, ollama_ip)
    unload_ollama_model(ollama_model, use_openai, openai_api_key, ollama_ip)
    final_text = "Key Takeaways:\n" + all_bullet_points + "\n\nExecutive Summary:\n" + executive_summary
    return final_text

def format_transcript_with_timestamps(result):
    transcript = ""
    for segment in result.get("segments", []):
        start_time = str(datetime.timedelta(seconds=int(segment["start"])))
        end_time = str(datetime.timedelta(seconds=int(segment["end"])))
        transcript += f"[{start_time} - {end_time}] {segment['text']}\n"
    return transcript

# -------------------- Export Functions --------------------
def export_to_pdf(text, filepath):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split('\n'):
            pdf.multi_cell(0, 10, txt=line)
        pdf.output(filepath)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export PDF: {e}")

def export_to_docx(text, filepath):
    try:
        document = Document()
        for line in text.split('\n'):
            document.add_paragraph(line)
        document.save(filepath)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to export DOCX: {e}")

# -------------------- Model Manager --------------------
class ModelManager:
    def __init__(self):
        self.model = None

    def load_model(self, model_name, use_cuda):
        device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'
        print(f"Loading Whisper model: {model_name} on device: {device}")
        self.model = whisper.load_model(model_name, device=device)
        return self.model

    def unload_model(self):
        if self.model is not None:
            print("Unloading Whisper model")
            del self.model
            self.model = None
            torch.cuda.empty_cache()

# -------------------- MeetingBuddy Application --------------------
class MeetingBuddyApp:
    def __init__(self):
        self.app = customtkinter.CTk()
        self.app.title("MeetingBuddy")

        # Load persistent settings
        self.settings = self.load_persistent_settings()

        # State variables
        self.model_manager = ModelManager()
        self.is_listening = False
        self.meeting_running = False
        self.live_transcribing = False
        self.advanced_meeting_running = False

        self.audio_data = []
        self.meeting_audio_data = []
        self.advanced_meeting_audio_data = []
        self.live_audio_queue = queue.Queue()
        self.ui_update_queue = queue.Queue()
        self.stop_event = threading.Event()

        # Tkinter Variables (with persisted defaults)
        self.default_model_var = tk.StringVar(value=self.settings.get("default_model", "base"))
        self.default_keep_model_loaded_var = tk.BooleanVar(value=self.settings.get("default_keep_model_loaded", False))
        self.default_ollama_model_var = tk.StringVar(value=self.settings.get("default_ollama_model", "Select Ollama Model"))
        self.default_ollama_ip_var = tk.StringVar(value=self.settings.get("default_ollama_ip", ""))
        self.default_use_openai_api_var = tk.BooleanVar(value=self.settings.get("default_use_openai_api", False))
        self.default_openai_api_key_var = tk.StringVar(value=self.settings.get("default_openai_api_key", ""))
        self.default_openai_model_var = tk.StringVar(value=self.settings.get("default_openai_model", "gpt-3.5-turbo"))
        self.use_cuda_var = tk.BooleanVar(value=torch.cuda.is_available())

        self.model_var = tk.StringVar(value=self.default_model_var.get())
        self.ollama_model_var = tk.StringVar(value=self.default_ollama_model_var.get())
        self.meeting_name_var = tk.StringVar(value="")
        self.old_meeting_name_var = tk.StringVar(value="")

        # Add new theme settings variables
        self.appearance_mode_var = customtkinter.StringVar(value=self.settings.get("appearance_mode", "Dark"))
        self.accent_color_var = customtkinter.StringVar(value=self.settings.get("accent_color", "#1f538d"))
        self.text_color_var = customtkinter.StringVar(value=self.settings.get("text_color", "white"))
        
        # Apply saved theme settings
        customtkinter.set_appearance_mode(self.appearance_mode_var.get())
        customtkinter.set_default_color_theme("blue")  # Base theme before custom colors

        # Add default theme constants
        self.DEFAULT_APPEARANCE_MODE = "Dark"
        self.DEFAULT_ACCENT_COLOR = "#1f538d"

        # Build the UI
        self.create_ui()

        # Preload model if enabled
        if self.default_keep_model_loaded_var.get():
            self.model_manager.load_model(self.model_var.get(), self.use_cuda_var.get())

        # Begin polling the UI update queue
        self.poll_ui_queue()

    # ------------- Persistent Settings -------------
    def load_persistent_settings(self):
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
            "default_openai_model": "gpt-3.5-turbo",
            "appearance_mode": "Dark",
            "accent_color": "#1f538d"
        }

    def save_persistent_settings(self):
        settings_data = {
            "default_model": self.default_model_var.get(),
            "default_keep_model_loaded": self.default_keep_model_loaded_var.get(),
            "default_ollama_model": self.default_ollama_model_var.get(),
            "default_ollama_ip": self.default_ollama_ip_var.get(),
            "default_use_openai_api": self.default_use_openai_api_var.get(),
            "default_openai_api_key": self.default_openai_api_key_var.get(),
            "default_openai_model": self.default_openai_model_var.get(),
            "appearance_mode": self.appearance_mode_var.get(),
            "accent_color": self.accent_color_var.get()
        }
        try:
            with open("settings.json", "w") as f:
                json.dump(settings_data, f)
        except Exception as e:
            messagebox.showerror("Error", f"Error saving settings: {e}")

    # ------------- UI Creation -------------
    def create_ui(self):
        self.tabview = customtkinter.CTkTabview(self.app, width=1000, height=600)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        self.tabview.add("Meeting Buddy")
        self.tabview.add("Old Meetings")
        self.tabview.add("Settings")

        # ---- Meeting Buddy Tab ----
        self.meeting_buddy_tab = self.tabview.tab("Meeting Buddy")
        self.model_dropdown = customtkinter.CTkOptionMenu(
            self.meeting_buddy_tab, variable=self.model_var,
            values=["tiny", "base", "small", "medium", "large", "turbo"],
            command=self.on_model_change
        )
        self.model_dropdown.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

        self.keep_model_loaded_checkbox = customtkinter.CTkCheckBox(
            self.meeting_buddy_tab, text="Keep model loaded (Whisper)",
            variable=self.default_keep_model_loaded_var, command=self.on_keep_model_loaded_change
        )
        self.keep_model_loaded_checkbox.grid(row=1, column=0, pady=5, padx=10, sticky="w")

        self.use_cuda_checkbox = customtkinter.CTkCheckBox(
            self.meeting_buddy_tab, text="Use CUDA (GPU)",
            variable=self.use_cuda_var, command=self.on_use_cuda_change
        )
        self.use_cuda_checkbox.grid(row=1, column=1, pady=5, padx=10, sticky="w")

        self.load_ollama_models_button = customtkinter.CTkButton(
            self.meeting_buddy_tab, text="Load Ollama Models", command=self.populate_ollama_models
        )
        self.load_ollama_models_button.grid(row=1, column=2, pady=5, padx=10, sticky="w")

        self.ollama_model_dropdown = customtkinter.CTkOptionMenu(
            self.meeting_buddy_tab, variable=self.ollama_model_var, values=[]
        )
        self.ollama_model_dropdown.grid(row=1, column=3, pady=5, padx=10, sticky="w")

        # Frames for Meeting Buddy tab
        self.mb_button_frame = customtkinter.CTkFrame(self.meeting_buddy_tab)
        self.mb_text_frame = customtkinter.CTkFrame(self.meeting_buddy_tab)
        self.mb_button_frame.grid(row=2, column=0, sticky="ns")
        self.mb_text_frame.grid(row=2, column=1, sticky="nsew", columnspan=3)
        self.meeting_buddy_tab.grid_rowconfigure(2, weight=1)
        self.meeting_buddy_tab.grid_columnconfigure(1, weight=1)

        # Controls
        self.listen_button = customtkinter.CTkButton(self.mb_button_frame, text="Listen")
        self.listen_button.pack(pady=5, padx=10, fill="x")
        self.listen_button.bind('<ButtonPress-1>', self.start_recording)
        self.listen_button.bind('<ButtonRelease-1>', self.stop_recording)

        self.start_meeting_button = customtkinter.CTkButton(self.mb_button_frame, text="Start Meeting", command=self.start_meeting)
        self.start_meeting_button.pack(pady=5, padx=10, fill="x")
        self.end_meeting_button = customtkinter.CTkButton(self.mb_button_frame, text="End Meeting", command=self.end_meeting)
        self.end_meeting_button.pack(pady=5, padx=10, fill="x")

        self.live_transcribe_button = customtkinter.CTkButton(self.mb_button_frame, text="Live Transcribe", command=self.start_live_transcription)
        self.live_transcribe_button.pack(pady=5, padx=10, fill="x")
        self.stop_live_button = customtkinter.CTkButton(self.mb_button_frame, text="Stop Live", command=self.stop_live_transcription)
        self.stop_live_button.pack(pady=5, padx=10, fill="x")
        self.stop_live_button.configure(state=tk.DISABLED)

        self.copy_button = customtkinter.CTkButton(self.mb_button_frame, text="Copy", command=self.copy_text)
        self.copy_button.pack(pady=5, padx=10, fill="x")

        # New Upload Audio Button
        self.upload_button = customtkinter.CTkButton(self.mb_button_frame, text="Upload Audio", command=self.upload_audio)
        self.upload_button.pack(pady=5, padx=10, fill="x")

        self.start_advanced_button = customtkinter.CTkButton(self.mb_button_frame, text="Start Advanced Meeting", command=self.start_advanced_meeting)
        self.start_advanced_button.pack(pady=5, padx=10, fill="x")
        self.end_advanced_button = customtkinter.CTkButton(self.mb_button_frame, text="End Advanced Meeting", command=self.end_advanced_meeting)
        self.end_advanced_button.pack(pady=5, padx=10, fill="x")
        self.end_advanced_button.configure(state=tk.DISABLED)

        self.summarize_button = customtkinter.CTkButton(self.mb_button_frame, text="Summarize", command=self.summarize_current_text)
        self.summarize_button.pack(pady=5, padx=10, fill="x")

        self.meeting_name_label = customtkinter.CTkLabel(self.mb_button_frame, text="Meeting Name:")
        self.meeting_name_label.pack(pady=(10, 0), padx=10, fill="x")
        self.meeting_name_entry = customtkinter.CTkEntry(self.mb_button_frame, textvariable=self.meeting_name_var)
        self.meeting_name_entry.pack(pady=(0, 10), padx=10, fill="x")

        self.text_output = scrolledtext.ScrolledText(self.mb_text_frame, wrap=tk.WORD, bg="#333", fg="white")
        self.text_output.pack(fill="both", expand=True, padx=10, pady=10)

        # ---- Old Meetings Tab ----
        self.old_meetings_tab = self.tabview.tab("Old Meetings")
        # Search Frame (Dynamic search: key release triggers search)
        self.search_frame = customtkinter.CTkFrame(self.old_meetings_tab)
        self.search_frame.pack(fill="x", padx=10, pady=(10, 0))
        self.search_entry = customtkinter.CTkEntry(self.search_frame, placeholder_text="Search meetings...")
        self.search_entry.pack(side=tk.LEFT, fill="x", expand=True, padx=(0,5))
        self.search_entry.bind("<KeyRelease>", self.search_meetings_event)
        self.clear_search_button = customtkinter.CTkButton(self.search_frame, text="Clear", command=self.update_old_meetings_list)
        self.clear_search_button.pack(side=tk.LEFT, padx=5)

        # Listbox Frame
        self.old_list_frame = customtkinter.CTkFrame(self.old_meetings_tab, width=200)
        self.old_list_frame.pack(side=tk.LEFT, fill="y", padx=10, pady=10)
        self.old_meetings_listbox = Listbox(self.old_list_frame, selectmode=SINGLE)
        self.old_meetings_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.old_meetings_listbox.bind("<<ListboxSelect>>", self.load_selected_meeting)

        # Controls Frame (Rename, Summarize, Save, Export)
        self.old_controls_frame = customtkinter.CTkFrame(self.old_list_frame)
        self.old_controls_frame.pack(fill="x", padx=5, pady=5)
        self.rename_button = customtkinter.CTkButton(self.old_controls_frame, text="Rename", command=self.rename_meeting)
        self.rename_button.pack(fill="x", pady=2, padx=2)
        self.summarize_old_button = customtkinter.CTkButton(self.old_controls_frame, text="Summarize", command=self.summarize_old_meeting)
        self.summarize_old_button.pack(fill="x", pady=2, padx=2)
        self.save_old_button = customtkinter.CTkButton(self.old_controls_frame, text="Save", command=self.save_old_meeting)
        self.save_old_button.pack(fill="x", pady=2, padx=2)
        self.export_pdf_button = customtkinter.CTkButton(self.old_controls_frame, text="Export PDF", command=self.export_current_meeting_pdf)
        self.export_pdf_button.pack(fill="x", pady=2, padx=2)
        self.export_docx_button = customtkinter.CTkButton(self.old_controls_frame, text="Export DOCX", command=self.export_current_meeting_docx)
        self.export_docx_button.pack(fill="x", pady=2, padx=2)
        self.old_meeting_name_entry = customtkinter.CTkEntry(self.old_controls_frame, textvariable=self.old_meeting_name_var)
        self.old_meeting_name_entry.pack(fill="x", pady=2, padx=2)

        self.old_text_frame = customtkinter.CTkFrame(self.old_meetings_tab)
        self.old_text_frame.pack(side=tk.RIGHT, fill="both", expand=True, padx=10, pady=10)
        self.old_meeting_text = scrolledtext.ScrolledText(self.old_text_frame, wrap=tk.WORD, bg="#333", fg="white")
        self.old_meeting_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.update_old_meetings_list()

        # ---- Settings Tab ----
        self.settings_tab = self.tabview.tab("Settings")

        # Main container
        settings_container = customtkinter.CTkFrame(self.settings_tab)
        settings_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Create the four quadrant frames
        top_left_frame = customtkinter.CTkFrame(settings_container)  # UI Settings
        top_right_frame = customtkinter.CTkFrame(settings_container)  # Whisper Model
        bottom_left_frame = customtkinter.CTkFrame(settings_container)  # Local LLMs
        bottom_right_frame = customtkinter.CTkFrame(settings_container)  # API Settings

        # Position the quadrants
        top_left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        top_right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        bottom_left_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        bottom_right_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # Quadrant 1: UI Settings (top left)
        theme_label = customtkinter.CTkLabel(top_left_frame, text="UI Theme Settings", font=("Arial", 16))
        theme_label.pack(pady=(10,5))

        # Appearance Mode (Dark/Light)
        appearance_frame = customtkinter.CTkFrame(top_left_frame)
        appearance_frame.pack(fill="x", padx=10, pady=5)
        appearance_label = customtkinter.CTkLabel(appearance_frame, text="Appearance Mode:")
        appearance_label.pack(side="left", padx=5)
        appearance_menu = customtkinter.CTkOptionMenu(
            appearance_frame,
            variable=self.appearance_mode_var,
            values=["Dark", "Light"],
            command=self.update_appearance_mode
        )
        appearance_menu.pack(side="right", padx=5)

        # Accent Color
        accent_frame = customtkinter.CTkFrame(top_left_frame)
        accent_frame.pack(fill="x", padx=10, pady=5)
        accent_label = customtkinter.CTkLabel(accent_frame, text="Accent Color:")
        accent_label.pack(side="left", padx=5)
        accent_entry = customtkinter.CTkEntry(
            accent_frame,
            textvariable=self.accent_color_var,
            width=100
        )
        accent_entry.pack(side="right", padx=5)

        # Color Preview
        color_preview = self.create_color_preview(top_left_frame)
        color_preview.pack(fill="x", padx=10, pady=5)
        
        # Reset Button
        reset_button = customtkinter.CTkButton(
            top_left_frame,
            text="Reset UI Settings",
            command=self.reset_ui_settings
        )
        reset_button.pack(fill="x", padx=10, pady=10)

        # Quadrant 2: Whisper Model Settings (top right)
        model_label = customtkinter.CTkLabel(top_right_frame, text="Whisper Model Settings", font=("Arial", 16))
        model_label.pack(pady=(10,5))

        self.default_model_label = customtkinter.CTkLabel(top_right_frame, text="Default Whisper Model:")
        self.default_model_label.pack(fill="x", padx=10, pady=5)
        self.default_model_menu = customtkinter.CTkOptionMenu(
            top_right_frame,
            variable=self.default_model_var,
            values=["tiny", "base", "small", "medium", "large", "turbo"]
        )
        self.default_model_menu.pack(fill="x", padx=10, pady=5)

        self.default_keep_checkbox = customtkinter.CTkCheckBox(
            top_right_frame,
            text="Keep default model loaded permanently",
            variable=self.default_keep_model_loaded_var
        )
        self.default_keep_checkbox.pack(fill="x", padx=10, pady=5)

        # Quadrant 3: Local LLM Settings (bottom left)
        llm_label = customtkinter.CTkLabel(bottom_left_frame, text="Local LLM Settings", font=("Arial", 16))
        llm_label.pack(pady=(10,5))

        self.default_ollama_label = customtkinter.CTkLabel(bottom_left_frame, text="Default Ollama Model:")
        self.default_ollama_label.pack(fill="x", padx=10, pady=5)

        ollama_options = load_ollama_models(self.default_use_openai_api_var.get(), 
                                              self.default_openai_api_key_var.get(), 
                                              self.default_ollama_ip_var.get())
        if not ollama_options:
            ollama_options = ["Select Ollama Model"]

        self.default_ollama_menu = customtkinter.CTkOptionMenu(
            bottom_left_frame,
            variable=self.default_ollama_model_var,
            values=ollama_options
        )
        self.default_ollama_menu.pack(fill="x", padx=10, pady=5)

        ip_frame = customtkinter.CTkFrame(bottom_left_frame)
        ip_frame.pack(fill="x", padx=10, pady=5)
        self.default_ollama_ip_label = customtkinter.CTkLabel(ip_frame, text="Ollama IP:")
        self.default_ollama_ip_label.pack(side="left", padx=5)
        self.default_ollama_ip_entry = customtkinter.CTkEntry(
            ip_frame,
            textvariable=self.default_ollama_ip_var
        )
        self.default_ollama_ip_entry.pack(side="right", expand=True, fill="x", padx=5)

        # Quadrant 4: API Settings (bottom right)
        self.openai_frame = customtkinter.CTkFrame(bottom_right_frame)
        self.openai_frame.pack(fill="x", padx=10, pady=10)

        api_label = customtkinter.CTkLabel(self.openai_frame, text="OpenAI API Settings", font=("Arial", 16))
        api_label.pack(pady=(10,5))

        self.default_use_openai_checkbox = customtkinter.CTkCheckBox(
            self.openai_frame,
            text="Use OpenAI API",
            variable=self.default_use_openai_api_var
        )
        self.default_use_openai_checkbox.pack(fill="x", padx=10, pady=5)

        key_frame = customtkinter.CTkFrame(self.openai_frame)
        key_frame.pack(fill="x", padx=10, pady=5)
        self.openai_key_label = customtkinter.CTkLabel(key_frame, text="API Key:")
        self.openai_key_label.pack(side="left", padx=5)
        self.openai_key_entry = customtkinter.CTkEntry(
            key_frame,
            textvariable=self.default_openai_api_key_var,
            show="*"
        )
        self.openai_key_entry.pack(side="right", expand=True, fill="x", padx=5)

        model_frame = customtkinter.CTkFrame(self.openai_frame)
        model_frame.pack(fill="x", padx=10, pady=5)
        self.openai_model_label = customtkinter.CTkLabel(model_frame, text="Model:")
        self.openai_model_label.pack(side="left", padx=5)
        self.openai_model_menu = customtkinter.CTkOptionMenu(
            model_frame,
            variable=self.default_openai_model_var,
            values=["gpt-3.5-turbo", "gpt-4"]
        )
        self.openai_model_menu.pack(side="right", padx=5)

        # Save Button at the bottom
        self.save_settings_button = customtkinter.CTkButton(settings_container, text="Save Settings", command=self.save_settings)
        self.save_settings_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        # Configure grid weights
        settings_container.grid_columnconfigure(0, weight=1)
        settings_container.grid_columnconfigure(1, weight=1)
        settings_container.grid_rowconfigure(0, weight=1)
        settings_container.grid_rowconfigure(1, weight=1)
        self.settings_tab.grid_columnconfigure(0, weight=1)
        self.settings_tab.grid_rowconfigure(0, weight=1)

    # ------------- UI Update Polling -------------
    def poll_ui_queue(self):
        try:
            while True:
                update_func = self.ui_update_queue.get_nowait()
                update_func()
        except queue.Empty:
            pass
        self.app.after(100, self.poll_ui_queue)

    def queue_ui_update(self, func):
        self.ui_update_queue.put(func)

    # ------------- Event Handlers -------------
    def on_model_change(self, selected_model):
        if self.default_keep_model_loaded_var.get():
            self.model_manager.load_model(selected_model, self.use_cuda_var.get())

    def on_keep_model_loaded_change(self):
        if self.default_keep_model_loaded_var.get():
            self.model_manager.load_model(self.model_var.get(), self.use_cuda_var.get())
        else:
            self.model_manager.unload_model()

    def on_use_cuda_change(self):
        if self.default_keep_model_loaded_var.get():
            self.model_manager.load_model(self.model_var.get(), self.use_cuda_var.get())

    def populate_ollama_models(self):
        models = load_ollama_models(self.default_use_openai_api_var.get(), self.default_openai_api_key_var.get(), self.default_ollama_ip_var.get())
        if models:
            self.ollama_model_dropdown.configure(values=models)
            if self.default_ollama_model_var.get() in models:
                self.ollama_model_var.set(self.default_ollama_model_var.get())
            else:
                self.ollama_model_var.set(models[0])
            messagebox.showinfo("Loaded", "Ollama models loaded successfully.")

    # ------------- Recording and Transcription -------------
    def start_recording(self, event=None):
        if self.is_listening:
            return
        self.is_listening = True
        self.queue_ui_update(lambda: self.listen_button.configure(text="Listening..."))
        fs = 16000
        self.audio_data = []
        def callback(indata, frames, time, status):
            self.audio_data.append(indata.copy())
        self.recording_stream = sd.InputStream(samplerate=fs, channels=1, callback=callback)
        self.recording_stream.start()

    def stop_recording(self, event=None):
        if not self.is_listening:
            return
        self.recording_stream.stop()
        self.recording_stream.close()
        self.queue_ui_update(lambda: self.listen_button.configure(text="Listen"))
        self.is_listening = False
        audio = np.concatenate(self.audio_data, axis=0)
        # For "Listen", do not apply timestamps.
        threading.Thread(target=self.transcribe_audio, args=(audio, False, None, False)).start()

    def transcribe_audio(self, audio, save_to_file=False, filename=None, apply_timestamps=True):
        self.disable_buttons()
        self.queue_ui_update(lambda: self.text_output.delete("1.0", tk.END))
        self.queue_ui_update(lambda: self.text_output.insert("1.0", "Transcribing..."))
        selected_model = self.model_var.get()
        device = 'cuda' if torch.cuda.is_available() and self.use_cuda_var.get() else 'cpu'
        if self.default_keep_model_loaded_var.get() and self.model_manager.model is not None:
            model = self.model_manager.model
        else:
            model = whisper.load_model(selected_model, device=device)
        audio_tensor = torch.from_numpy(audio).float().to(device)
        result = model.transcribe(audio_tensor.flatten(), fp16=(device=='cuda'))
        transcription = format_transcript_with_timestamps(result) if apply_timestamps else result['text']
        self.queue_ui_update(lambda: self.text_output.delete("1.0", tk.END))
        self.queue_ui_update(lambda: self.text_output.insert("1.0", transcription))
        if save_to_file and filename:
            self.save_transcription(transcription, filename)
        self.enable_buttons()

    def copy_text(self):
        text = self.text_output.get("1.0", tk.END).strip()
        if text:
            pyperclip.copy(text)

    def disable_buttons(self):
        self.queue_ui_update(lambda: self.listen_button.configure(state=tk.DISABLED))
        self.queue_ui_update(lambda: self.start_meeting_button.configure(state=tk.DISABLED))
        self.queue_ui_update(lambda: self.end_meeting_button.configure(state=tk.DISABLED))
        self.queue_ui_update(lambda: self.live_transcribe_button.configure(state=tk.DISABLED))
        self.queue_ui_update(lambda: self.stop_live_button.configure(state=tk.DISABLED))

    def enable_buttons(self):
        self.queue_ui_update(lambda: self.listen_button.configure(state=tk.NORMAL))
        self.queue_ui_update(lambda: self.start_meeting_button.configure(state=tk.NORMAL))
        self.queue_ui_update(lambda: self.end_meeting_button.configure(state=tk.NORMAL))
        self.queue_ui_update(lambda: self.live_transcribe_button.configure(state=tk.NORMAL))
        self.queue_ui_update(lambda: self.stop_live_button.configure(state=tk.NORMAL))

    # ------------- New Method: Summarize Current Text -------------
    def summarize_current_text(self):
        text = self.text_output.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "No transcript available to summarize.")
            return
        try:
            # Unload Whisper model before summarization
            self.model_manager.unload_model()
            
            if self.default_use_openai_api_var.get() and self.default_openai_api_key_var.get().strip():
                summary = get_openai_summary(text, self.default_openai_api_key_var.get(), self.default_openai_model_var.get())
            else:
                summary = summarize_text_local(text, self.ollama_model_var.get(),
                                               self.default_use_openai_api_var.get(),
                                               self.default_openai_api_key_var.get(),
                                               self.default_ollama_ip_var.get())
            self.queue_ui_update(lambda: self.text_output.insert(tk.END, "\n\nSummary:\n" + summary))
            
            # Reload Whisper model if keep_model_loaded is enabled
            if self.default_keep_model_loaded_var.get():
                self.model_manager.load_model(self.model_var.get(), self.use_cuda_var.get())
            
        except Exception as e:
            messagebox.showerror("Error", f"Summarization failed: {e}")
            # Ensure model is reloaded even if summarization fails
            if self.default_keep_model_loaded_var.get():
                self.model_manager.load_model(self.model_var.get(), self.use_cuda_var.get())

    def start_meeting(self):
        if self.meeting_running:
            messagebox.showwarning("Warning", "Meeting is already running.")
            return
        self.meeting_running = True
        self.meeting_start_time = datetime.datetime.now()
        self.meeting_audio_data = []
        self.queue_ui_update(lambda: self.start_meeting_button.configure(state=tk.DISABLED))
        self.queue_ui_update(lambda: self.end_meeting_button.configure(state=tk.NORMAL))
        self.queue_ui_update(lambda: self.text_output.delete("1.0", tk.END))
        self.queue_ui_update(lambda: self.text_output.insert("1.0", "Meeting started..."))
        threading.Thread(target=self.record_meeting).start()

    def end_meeting(self):
        if not self.meeting_running:
            messagebox.showwarning("Warning", "No meeting is currently running.")
            return
        self.meeting_running = False
        self.queue_ui_update(lambda: self.end_meeting_button.configure(state=tk.DISABLED))
        self.queue_ui_update(lambda: self.start_meeting_button.configure(state=tk.NORMAL))
        self.queue_ui_update(lambda: self.text_output.insert(tk.END, "\nMeeting ended. Transcribing..."))
        # The meeting recording thread will finish and trigger transcription.

    def record_meeting(self):
        fs = 16000
        def callback(indata, frames, time, status):
            self.meeting_audio_data.append(indata.copy())
        with sd.InputStream(samplerate=fs, channels=1, callback=callback):
            start_time = datetime.datetime.now()
            while self.meeting_running:
                if (datetime.datetime.now() - start_time).total_seconds() >= 7200:
                    self.meeting_running = False
                    break
                sd.sleep(1000)
        audio = np.concatenate(self.meeting_audio_data, axis=0)
        meeting_name = self.meeting_name_var.get().strip()
        if meeting_name == "":
            base_name = self.meeting_start_time.strftime("%m-%d-%Y-%H%M")
        else:
            base_name = meeting_name + "_" + self.meeting_start_time.strftime("%m-%d-%Y-%H%M")
        txt_filepath = os.path.join("meetings", base_name + ".txt")
        wav_filepath = os.path.join("meetings", base_name + ".wav")
        # Save audio file
        save_audio(audio, wav_filepath, samplerate=fs)
        threading.Thread(target=self.transcribe_audio, args=(audio, True, txt_filepath, True)).start()

    def save_transcription(self, transcription, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(transcription)
            messagebox.showinfo("Saved", f"Transcription saved to {filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save transcription: {e}")

    def start_live_transcription(self):
        if self.live_transcribing:
            messagebox.showwarning("Warning", "Live transcription is already running.")
            return
        self.live_transcribing = True
        self.stop_event.clear()
        self.queue_ui_update(lambda: self.live_transcribe_button.configure(state=tk.DISABLED))
        self.queue_ui_update(lambda: self.stop_live_button.configure(state=tk.NORMAL))
        self.queue_ui_update(lambda: self.text_output.delete("1.0", tk.END))
        self.queue_ui_update(lambda: self.text_output.insert(tk.END, "Live transcription started..."))
        fs = 16000
        def audio_callback(indata, frames, time, status):
            self.live_audio_queue.put(indata.copy())
        self.recording_stream = sd.InputStream(samplerate=fs, channels=1, callback=audio_callback)
        self.recording_stream.start()
        threading.Thread(target=self.process_live_audio).start()

    def stop_live_transcription(self):
        if not self.live_transcribing:
            messagebox.showwarning("Warning", "No live transcription is running.")
            return
        self.live_transcribing = False
        self.stop_event.set()
        if self.recording_stream is not None:
            self.recording_stream.stop()
            self.recording_stream.close()
            self.recording_stream = None
        def check_thread():
            if threading.active_count() > 1:
                self.app.after(100, check_thread)
            else:
                self.queue_ui_update(lambda: self.live_transcribe_button.configure(state=tk.NORMAL))
                self.queue_ui_update(lambda: self.stop_live_button.configure(state=tk.DISABLED))
                self.queue_ui_update(lambda: self.text_output.insert(tk.END, "\nLive transcription stopped."))
        check_thread()

    def process_live_audio(self):
        buffer_duration = 5  # seconds
        buffer_size = int(16000 * buffer_duration)
        audio_buffer = np.zeros((0, 1), dtype=np.float32)
        selected_model = self.model_var.get()
        device = 'cuda' if torch.cuda.is_available() and self.use_cuda_var.get() else 'cpu'
        if self.default_keep_model_loaded_var.get() and self.model_manager.model is not None:
            model = self.model_manager.model
        else:
            model = whisper.load_model(selected_model, device=device)
        while not self.stop_event.is_set():
            try:
                while not self.live_audio_queue.empty():
                    data = self.live_audio_queue.get()
                    audio_buffer = np.vstack((audio_buffer, data))
                if len(audio_buffer) >= buffer_size:
                    audio_chunk = audio_buffer[:buffer_size]
                    audio_buffer = audio_buffer[buffer_size:]
                    audio_tensor = torch.from_numpy(audio_chunk.flatten()).float().to(device)
                    result = model.transcribe(audio_tensor, fp16=(device=='cuda'))
                    transcription = format_transcript_with_timestamps(result)
                    self.queue_ui_update(lambda: self.text_output.insert(tk.END, transcription + " "))
                    self.queue_ui_update(lambda: self.text_output.see(tk.END))
                else:
                    threading.Event().wait(0.05)
            except Exception as e:
                if not self.stop_event.is_set():
                    messagebox.showerror("Error", str(e))
                break
        if len(audio_buffer) > 0:
            try:
                audio_tensor = torch.from_numpy(audio_buffer.flatten()).float().to(device)
                result = model.transcribe(audio_tensor, fp16=(device=='cuda'))
                transcription = result['text']
                self.queue_ui_update(lambda: self.text_output.insert(tk.END, transcription))
                self.queue_ui_update(lambda: self.text_output.see(tk.END))
            except Exception as e:
                if not self.stop_event.is_set():
                    messagebox.showerror("Error", str(e))

    def start_advanced_meeting(self):
        if self.advanced_meeting_running:
            messagebox.showwarning("Warning", "Advanced meeting is already running.")
            return
        self.advanced_meeting_running = True
        self.advanced_meeting_start_time = datetime.datetime.now()
        self.advanced_meeting_audio_data = []
        self.queue_ui_update(lambda: self.start_advanced_button.configure(state=tk.DISABLED))
        self.queue_ui_update(lambda: self.end_advanced_button.configure(state=tk.NORMAL))
        self.queue_ui_update(lambda: self.text_output.delete("1.0", tk.END))
        self.queue_ui_update(lambda: self.text_output.insert("1.0", "Advanced Meeting started..."))
        threading.Thread(target=self.record_advanced_meeting).start()

    def end_advanced_meeting(self):
        if not self.advanced_meeting_running:
            messagebox.showwarning("Warning", "No advanced meeting is currently running.")
            return
        self.advanced_meeting_running = False
        self.queue_ui_update(lambda: self.end_advanced_button.configure(state=tk.DISABLED))
        self.queue_ui_update(lambda: self.start_advanced_button.configure(state=tk.NORMAL))
        self.queue_ui_update(lambda: self.text_output.insert(tk.END, "\nAdvanced Meeting ended. Transcribing and summarizing..."))
        # The recording thread will handle transcription and summarization.

    def record_advanced_meeting(self):
        fs = 16000
        def callback(indata, frames, time, status):
            self.advanced_meeting_audio_data.append(indata.copy())
        with sd.InputStream(samplerate=fs, channels=1, callback=callback):
            start_time = datetime.datetime.now()
            while self.advanced_meeting_running:
                if (datetime.datetime.now() - start_time).total_seconds() >= 7200:
                    self.advanced_meeting_running = False
                    break
                sd.sleep(1000)
        audio = np.concatenate(self.advanced_meeting_audio_data, axis=0)
        selected_model = self.model_var.get()
        device = 'cuda' if torch.cuda.is_available() and self.use_cuda_var.get() else 'cpu'
        if self.default_keep_model_loaded_var.get() and self.model_manager.model is not None:
            model = self.model_manager.model
        else:
            model = whisper.load_model(selected_model, device=device)
        audio_tensor = torch.from_numpy(audio).float().to(device)
        result = model.transcribe(audio_tensor.flatten(), fp16=(device=='cuda'))
        transcription = format_transcript_with_timestamps(result)
        if not (self.default_use_openai_api_var.get() and self.default_openai_api_key_var.get().strip()):
            if not self.default_keep_model_loaded_var.get():
                self.model_manager.unload_model()
            else:
                self.model_manager.load_model(self.model_var.get(), self.use_cuda_var.get())
        if self.default_use_openai_api_var.get() and self.default_openai_api_key_var.get().strip():
            summary = get_openai_summary(transcription, self.default_openai_api_key_var.get(), self.default_openai_model_var.get())
        else:
            summary = summarize_text_local(transcription, self.ollama_model_var.get(),
                                           self.default_use_openai_api_var.get(),
                                           self.default_openai_api_key_var.get(),
                                           self.default_ollama_ip_var.get())
        meeting_name = self.meeting_name_var.get().strip()
        if meeting_name == "":
            base_name = self.advanced_meeting_start_time.strftime("%m-%d-%Y-%H%M")
        else:
            base_name = meeting_name + "_" + self.advanced_meeting_start_time.strftime("%m-%d-%Y-%H%M")
        txt_filepath = os.path.join("meetings", base_name + "_advanced.txt")
        wav_filepath = os.path.join("meetings", base_name + "_advanced.wav")
        # Save audio file
        save_audio(audio, wav_filepath, samplerate=fs)
        final_text = "Transcript:\n" + transcription + "\n\nSummary:\n" + summary
        self.save_transcription(final_text, txt_filepath)
        self.queue_ui_update(lambda: self.text_output.delete("1.0", tk.END))
        self.queue_ui_update(lambda: self.text_output.insert("1.0", final_text))

    # ------------- Old Meetings Tab Functions -------------
    def update_old_meetings_list(self):
        self.old_meetings_listbox.delete(0, END)
        if not os.path.exists("meetings"):
            os.makedirs("meetings", exist_ok=True)
        files = sorted(os.listdir("meetings"))
        for file in files:
            if file.endswith(".txt"):
                self.old_meetings_listbox.insert(END, file)

    def load_selected_meeting(self, event=None):
        selected = self.old_meetings_listbox.curselection()
        if selected:
            filename = self.old_meetings_listbox.get(selected[0])
            filepath = os.path.join("meetings", filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                self.old_meeting_text.delete("1.0", tk.END)
                self.old_meeting_text.insert("1.0", content)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load meeting: {e}")

    def rename_meeting(self):
        selected = self.old_meetings_listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "No meeting selected for renaming.")
            return
        original_filename = self.old_meetings_listbox.get(selected[0])
        new_name = self.old_meeting_name_var.get().strip()
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
            self.update_old_meetings_list()
            messagebox.showinfo("Renamed", f"Meeting renamed to {new_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rename meeting: {e}")

    def summarize_old_meeting(self):
        content = self.old_meeting_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No meeting content to summarize.")
            return
        try:
            # Unload Whisper model before summarization
            self.model_manager.unload_model()
            
            if self.default_use_openai_api_var.get() and self.default_openai_api_key_var.get().strip():
                summary = get_openai_summary(content, self.default_openai_api_key_var.get(), self.default_openai_model_var.get())
            else:
                summary = summarize_text_local(content, self.ollama_model_var.get(),
                                               self.default_use_openai_api_var.get(),
                                               self.default_openai_api_key_var.get(),
                                               self.default_ollama_ip_var.get())
            self.old_meeting_text.insert(tk.END, "\n\nSummary:\n" + summary)
            
            # Reload Whisper model if keep_model_loaded is enabled
            if self.default_keep_model_loaded_var.get():
                self.model_manager.load_model(self.model_var.get(), self.use_cuda_var.get())
            
        except Exception as e:
            messagebox.showerror("Error", f"Summarization failed: {e}")
            # Ensure model is reloaded even if summarization fails
            if self.default_keep_model_loaded_var.get():
                self.model_manager.load_model(self.model_var.get(), self.use_cuda_var.get())

    def save_old_meeting(self):
        selected = self.old_meetings_listbox.curselection()
        if not selected:
            messagebox.showwarning("Warning", "No meeting selected to save.")
            return
        filename = self.old_meetings_listbox.get(selected[0])
        filepath = os.path.join("meetings", filename)
        content = self.old_meeting_text.get("1.0", tk.END)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Saved", f"Changes saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save meeting: {e}")

    # ------------- Export Functions (Old Meetings) -------------
    def export_current_meeting_pdf(self):
        content = self.old_meeting_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No meeting content to export.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
        if filepath:
            export_to_pdf(content, filepath)
            messagebox.showinfo("Exported", f"Meeting exported as PDF to {filepath}")

    def export_current_meeting_docx(self):
        content = self.old_meeting_text.get("1.0", tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No meeting content to export.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".docx", filetypes=[("Word Documents", "*.docx")])
        if filepath:
            export_to_docx(content, filepath)
            messagebox.showinfo("Exported", f"Meeting exported as DOCX to {filepath}")

    # ------------- Search Functions (Old Meetings) -------------
    def search_meetings_event(self, event=None):
        self.search_meetings()

    def search_meetings(self):
        term = self.search_entry.get().strip().lower()
        self.old_meetings_listbox.delete(0, END)
        if not os.path.exists("meetings"):
            os.makedirs("meetings", exist_ok=True)
        for file in sorted(os.listdir("meetings")):
            if file.endswith(".txt"):
                filepath = os.path.join("meetings", file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read().lower()
                    if term in content or term in file.lower():
                        self.old_meetings_listbox.insert(END, file)
                except Exception:
                    continue

    def save_settings(self):
        self.default_model_var.set(self.default_model_var.get())
        self.default_keep_model_loaded_var.set(self.default_keep_model_loaded_var.get())
        self.default_ollama_model_var.set(self.default_ollama_model_var.get())
        self.settings["appearance_mode"] = self.appearance_mode_var.get()
        self.settings["accent_color"] = self.accent_color_var.get()
        self.settings["text_color"] = self.text_color_var.get()
        
        self.save_persistent_settings()
        self.apply_theme_settings()
        messagebox.showinfo("Settings Saved", "Settings have been updated.")

    # ------------- New Method: Upload Audio -------------
    def upload_audio(self):
        filepath = filedialog.askopenfilename(
            title="Select an Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                # Use whisper.load_audio to load the file as a numpy array
                audio = whisper.load_audio(filepath)
                # Optionally, you might want to resample or trim the audio here.
                threading.Thread(target=self.transcribe_audio, args=(audio, False, None, True)).start()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process uploaded audio: {e}")

    # ------------- Run Application -------------
    def run(self):
        self.app.mainloop()

    # ------------- New Method: Create Color Preview -------------
    def create_color_preview(self, parent):
        self.color_preview = customtkinter.CTkFrame(parent)
        self.color_buttons = []  # Store color buttons
        colors = [
            "#1f538d", "#2FA572", "#9D3BE1", "#E12B3B", 
            "#E17B2B", "#E1D92B", "#3BE136", "#2BC4E1"
        ]
        for i, color in enumerate(colors):
            color_button = customtkinter.CTkButton(
                self.color_preview, 
                text="", 
                width=30,
                height=30,
                fg_color=color,
                hover_color=color,
                command=lambda c=color: self.accent_color_var.set(c)
            )
            color_button.grid(row=0, column=i, padx=2, pady=2)
            self.color_buttons.append((color_button, color))  # Store button and its color
        return self.color_preview

    # ------------- New Method: Update Appearance Mode -------------
    def update_appearance_mode(self, new_mode):
        customtkinter.set_appearance_mode(new_mode)

    # ------------- New Method: Apply Theme Settings -------------
    def apply_theme_settings(self):
        customtkinter.set_appearance_mode(self.appearance_mode_var.get())
        # Update all widgets with new colors
        self.update_widget_colors(self.app)

    # ------------- New Method: Update Widget Colors -------------
    def update_widget_colors(self, widget):
        try:
            # Skip color preview buttons by checking if they're in the color preview frame
            if isinstance(widget, customtkinter.CTkButton):
                # Get widget's parent
                parent = widget.winfo_parent()
                if parent:
                    parent_widget = widget.nametowidget(parent)
                    # Check if this button is in the color preview frame
                    is_color_preview_button = False
                    try:
                        is_color_preview_button = widget.winfo_width() == 30 and widget.winfo_height() == 30
                    except:
                        pass
                    
                    if not is_color_preview_button:
                        widget.configure(fg_color=self.accent_color_var.get())
            elif isinstance(widget, customtkinter.CTkOptionMenu):
                widget.configure(fg_color=self.accent_color_var.get())
        except:
            pass
        for child in widget.winfo_children():
            self.update_widget_colors(child)

    # ------------- New Method: Reset UI Settings -------------
    def reset_ui_settings(self):
        self.appearance_mode_var.set(self.DEFAULT_APPEARANCE_MODE)
        self.accent_color_var.set(self.DEFAULT_ACCENT_COLOR)
        customtkinter.set_appearance_mode(self.DEFAULT_APPEARANCE_MODE)
        
        # Restore color preview buttons to their original colors
        for button, color in self.color_buttons:
            button.configure(fg_color=color, hover_color=color)
        
        self.update_widget_colors(self.app)

if __name__ == "__main__":
    app = MeetingBuddyApp()
    app.run()
