# Repository Guidelines

## Project Structure & Module Organization
Core runtime lives in `voicebot.py`, orchestrating ALSA capture, Faster-Whisper STT, Ollama replies, and Piper TTS. Persona definitions now live under `personas/` and link prompts to the voices in `voices/`; add new JSON files there when crafting agents. Deployment templates stay in `systemd/`. Copy `example.env` to `.env` for local tuning; secrets never belong in Git. Session memories persist in `memories.json`, and `tts_test.py` provides a quick Piper-only sanity check before touching the full stack.

## Build, Test, and Development Commands
Run `uv sync` once to install dependencies. `make run` (or `uv run python voicebot.py`) starts the assistant with automatic recovery. `make calibrate` invokes `voicebot.py --calibrate-vad` for mic noise measurements. Quality gates rely on Ruff: `make lint` for `ruff check .` and `make format` for `ruff format .`. Use `uv run pytest` for automated tests and `uv run python tts_test.py "hi tars"` to validate Piper audio output.

## Coding Style & Naming Conventions
Target Python 3.12, 4-space indents, and Black-like wrapping. Functions, modules, and files use snake_case; env constants stay SCREAMING_SNAKE_CASE. Keep streaming logic composable—factor helpers once a function exceeds one screen. Logging should remain structured JSON (respecting `LOG_JSON`) and must not leak secrets. Always run Ruff before pushing; configuration lives in `pyproject.toml`.

## Testing Guidelines
Prefer `pytest` with suites under `tests/`, mirroring module names (e.g., `tests/test_voicebot.py`). Name cases `test_<behavior>` and mock ALSA, Ollama, and Piper boundaries so tests stay offline. Capture timing thresholds, wake windows, and MQTT flows in assertions. For manual smoke tests, stream logs via `uv run python voicebot.py | jq` and toggle `WAKE_DEBUG_WAV=1` when tuning wake detection.

## Commit & Pull Request Guidelines
Git history favors short, imperative subjects (“better stt buffer”), so keep future commits ≤72 chars and expand context in the body only when necessary. Pull requests should summarize motivation, link issues, list validation commands (`make lint`, `uv run pytest`), and note calibration or env changes. Include concise audio/log evidence whenever you adjust wakeword, VAD, or TTS heuristics.

## Environment & Deployment Notes
Never commit `.env` or Piper binaries; rely on `example.env` plus local overrides. Update `systemd/voicebot.service` with the proper `User`, `WorkingDirectory`, and `EnvironmentFile` before `systemctl enable --now`. Confirm Ollama and Piper services respond before launching the agent, and clear `memories.json` before demos if prior context is sensitive.
