PY = uv run python

.PHONY: run calibrate lint format

run:
	uv run python voicebot.py

calibrate:
	uv run python voicebot.py --calibrate-vad

lint:
	uv run ruff check .

format:
	uv run ruff format .
