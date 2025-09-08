# 🎙️ TARS Voicebot

A local voice assistant inspired by **TARS from Interstellar**, running on Linux with:

- **AIRHUG 01 USB mic/speaker** via ALSA
- **Wakeword detection** (“hey Tars”) using VAD + Whisper
- **Speech-to-text** with [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- **LLM responses** served by [Ollama](https://ollama.ai)
- **Text-to-speech** with [Piper](https://github.com/rhasspy/piper) (custom TARS voice)
- Optional **MQTT presence trigger** for smart-home integration
- Optional **conversation memory** (summaries stored as JSON)

---

## 🚀 Features

- 🎤 Capture audio via ALSA (`plughw:2,0`) — no PortAudio required  
- 🔊 Spoken replies via Piper TTS → AIRHUG speaker  
- 🗣️ Wakeword: “hey Tars” (configurable in `.env`)  
- 🤖 Chat backend: Ollama (`gpt-oss:20b`, `llama3:8b-instruct`, or your choice)  
- 📡 Optional MQTT presence topic (`vision/frontdesk/presence`)  
- 🧠 Memory: saves summaries of each conversation to `memories.json`

---

## 📦 Requirements

- Python 3.10+ with [uv](https://docs.astral.sh/uv/)  
- Working ALSA setup (USB mic + speaker detected in `arecord -l` and `aplay -l`)  
- [Ollama](https://ollama.ai) running locally or on your network  
- Piper TTS model (`.onnx` + `.onnx.json`)  
- MQTT broker (optional)

---

## 🔧 Setup

1. Clone the repo and enter:
   ```bash
   git clone https://github.com/YOUR_USERNAME/voicebot.git
   cd voicebot

2. 
