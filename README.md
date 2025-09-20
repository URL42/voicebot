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
   ```

2.  Configure Git (one-time per machine):
   ```bash
   git config --global user.name "URL42"
   git config --global user.email "your@email.com"
   git config --global init.defaultBranch main
   ```

3. Initialize the local repository:
   ```bash
   cd ~/voicebot
   git init
   git add .
   git commit -m "Initial commit: working TARS voicebot"
   ```

4. Create the GitHub repository:

- Go to [https://github.com/new](https://github.com/new)  
- Set **Repository name**: `voicebot`  
- Leave it empty (no README, .gitignore, or license)  
- Click **Create repository**  
- Copy the HTTPS URL (e.g., `https://github.com/URL42/voicebot.git`)

5. Add the remote:

```bash
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/URL42/voicebot.git
git remote -v
```

6. 
