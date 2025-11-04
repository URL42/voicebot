# TARS Voicebot

A local-first voice assistant tuned for the AIRHUG microphone/speaker stack. It listens for a wake phrase, streams speech into Faster-Whisper for transcription, sends text to an Ollama-served LLM, and speaks answers through Piper TTS.

## Highlights
- ALSA-first capture with automatic `hw→plughw` fallback and optional PortAudio.
- Always-on wakeword windows with barge-in handling and follow-up grace period.
- Faster-Whisper streaming STT + Piper TTS (custom voices under `voices/`).
- Optional MQTT presence trigger and lightweight memory summaries.
- Bluetooth speaker auto-connect support when `AUTO_BT_CONNECT=1`.

## Requirements
- Linux with functional ALSA drivers (`arecord -l`, `aplay -l`).
- Python 3.12+ and [uv](https://docs.astral.sh/uv/) for dependency management.
- [Ollama](https://ollama.ai) serving your chosen chat model.
- [Piper](https://github.com/rhasspy/piper) model files (examples in `voices/`).
- Optional: MQTT broker publishing presence payloads.

## Setup
1. Install dependencies:
   ```bash
   uv sync
   ```
2. Copy and edit the environment file:
   ```bash
   cp example.env .env
   $EDITOR .env
   ```
   Key knobs:
   - `REQUIRE_WAKEWORD`, `WAKE_ALWAYS_ON`, `WAKE_WINDOW_SEC`, `POST_TTS_ACCEPT_SEC`.
   - `ALSA_DEVICE` set to `auto` to detect AIRHUG, or override with `plughw:x,y`.
   - `PIPER_VOICE` pointing at the voice you want Piper to use.
   - `ENABLE_PRESENCE=1` if you want MQTT to arm/disarm the mic.
3. Start Ollama and Piper so they can accept requests.

## Running Locally
Launch an interactive session in the project root:
```bash
uv run python voicebot.py
```
The assistant now auto-restarts if ALSA, Whisper, or other subsystems exit unexpectedly. Wakeword mode remains armed indefinitely, so you can say “hey Tars” at any time without restarting the process.

## Systemd Service
A template unit is provided at `systemd/voicebot.service`. Adjust `User`, `Group`, `WorkingDirectory`, and the `EnvironmentFile` path for your system, then install it:
```bash
sudo cp systemd/voicebot.service /etc/systemd/system/voicebot.service
sudo cp .env /etc/voicebot.env   # or craft a dedicated env file
sudo systemctl daemon-reload
sudo systemctl enable --now voicebot.service
```
Logs are available via `journalctl -u voicebot.service -f`.

## Wakeword & Barge-In Tips
- Use `WAKE_SENSITIVITY=high` if background noise is low, otherwise stick to `medium`.
- `POST_TTS_ACCEPT_SEC` keeps the wake window open after Piper finishes speaking so you can jump back in without repeating the wakeword.
- If the wakeword seems too eager, reduce `WAKE_WINDOW_SEC` or switch sensitivity to `low`.
- Set `ALLOW_BARGE_IN=1` to interrupt TTS when you start speaking; tune `BARGE_IN_FRAMES` (≈30 ms per frame).

## Maintenance & Troubleshooting
- Clear the generated `memories.json` if you want to reset conversation summaries.
- Use `WAKE_DEBUG_WAV=1` to dump wakeword audio snippets under `/tmp` for tuning.
- For Bluetooth speakers, populate `BT_SPEAKER_MAC` and set `AUTO_BT_CONNECT=1`.
- If running headless, keep Ollama and Piper managed by their own services so this unit can reconnect.
