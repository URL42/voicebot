# TARS Voicebot

A local-first voice assistant tuned for the AIRHUG microphone/speaker stack. It listens for a wake phrase, streams speech into Faster-Whisper for transcription, sends text to an Ollama-served LLM, and speaks answers through Piper TTS.

## Highlights
- ALSA-first capture with automatic `hw→plughw` fallback and optional PortAudio.
- Always-on wakeword windows with barge-in handling and follow-up grace period.
- Faster-Whisper streaming STT + Piper TTS (custom voices under `voices/`).
- Optional MQTT presence trigger and lightweight memory summaries.
- Bluetooth speaker auto-connect support when `AUTO_BT_CONNECT=1`.
- Structured JSON logging with wake/STT metrics so you can tail/ingest easily.
- Optional acoustic wakeword detector powered by [openWakeWord](https://github.com/dscripka/openwakeword).

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

### Developer Shortcuts
- `make run` – start the assistant.
- `make calibrate` – guided VAD/noise calibration (records short samples).
- `make lint` / `make format` – run Ruff.

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

### Acoustic Wakeword Detector
Flip on a dedicated detector when you have an [openWakeWord](https://github.com/dscripka/openwakeword) `.tflite` model for your phrase:

```env
USE_ACOUSTIC_WAKE=1
ACOUSTIC_WAKE_MODEL=/home/anthony/models/hey_tars.tflite
ACOUSTIC_WAKE_THRESHOLD=0.65
```

The detector runs before Whisper and opens the accept window when the score passes the threshold. Normal transcript-based matching (with better fuzzy heuristics) still runs afterward for follow-up utterances.

### Wake Debugging & Calibration
- Set `WAKE_DEBUG_WAV=1` to dump both legacy wakeword attempts and near-miss clips from always-on mode. Files land in `/tmp/wake_*.wav` with the transcript score metadata.
- Run `make calibrate` (or `uv run python voicebot.py --calibrate-vad`) to capture a few seconds of room noise + a spoken wakephrase. The tool prints RMS/SNR plus suggested `VAD_AGGRESSIVENESS`, `MAX_SILENCE_SECONDS`, and `WAKE_WINDOW_SEC` values.
- If you swap to a new phrase (e.g., “hey Jarvis”), mirror those variants in `WAKE_ALIASES` so they’re stripped before we forward the text to the LLM.
- `PRE_SPEECH_PAD_MS` buffers ~120 ms of audio ahead of VAD detection, which keeps the first syllable of your command from getting clipped.

### Standalone TTS Tester
Need to sanity-check Piper without running the whole assistant?

```bash
uv run python tts_test.py "hello from TARS"
# or launch an interactive loop:
uv run python tts_test.py
```

The tester loads the same `.env` voice/settings, plays through the configured ALSA device, and can save WAV files via `--save out.wav`.

## Maintenance & Troubleshooting
- Clear the generated `memories.json` if you want to reset conversation summaries.
- Use `WAKE_DEBUG_WAV=1` to dump wakeword audio snippets under `/tmp` for tuning.
- For Bluetooth speakers, populate `BT_SPEAKER_MAC` and set `AUTO_BT_CONNECT=1`.
- If running headless, keep Ollama and Piper managed by their own services so this unit can reconnect.
- JSON logs are emitted to stdout; pipe them into `jq` or Loki/Grafana for long-term tuning (`uv run python voicebot.py | jq`).
