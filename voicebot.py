#!/usr/bin/env python3
"""
Voicebot with:
- ALSA capture (no PortAudio needed), robust hw→plughw fallback
- Auto-detect AIRHUG (if ALSA_DEVICE=auto or unset)
- Always-on wakeword option + sensitivity (low/medium/high)
- VAD-based end-of-utterance
- Ollama chat, faster-whisper STT
- Piper TTS spoken replies via ALSA playback
- Optional MQTT presence trigger & memory
- Barge-in (interrupt TTS on user speech; can be disabled via env)
- <think>…</think> sanitizer to hide chain-of-thought
- Keeps wake accept window open briefly **after TTS** for natural follow-ups
"""

import argparse
import logging
import math
import os
import re
import sys
import json
import time
import wave
import queue
import threading
import subprocess
import datetime
import tempfile
import shutil
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
import webrtcvad
import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import paho.mqtt.client as mqtt

try:
    from openwakeword.model import Model as WakewordModel
except ImportError:  # pragma: no cover - optional dependency
    WakewordModel = None

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz
    from rapidfuzz.distance import Levenshtein as rapidfuzz_lev
except ImportError:  # pragma: no cover - optional dependency
    rapidfuzz_fuzz = None
    rapidfuzz_lev = None

# ---------------- Config ----------------
load_dotenv()

LOG_LEVEL            = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_JSON             = os.getenv("LOG_JSON", "1") == "1"
ENABLE_PRESENCE       = os.getenv("ENABLE_PRESENCE", "1") == "1"
MQTT_HOST             = os.getenv("MQTT_HOST", "127.0.0.1")
MQTT_PORT             = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER             = os.getenv("MQTT_USER") or None
MQTT_PASS             = os.getenv("MQTT_PASS") or None
MQTT_PRESENCE_TOPIC   = os.getenv("MQTT_PRESENCE_TOPIC", "vision/frontdesk/presence")

INPUT_DEVICE_INDEX    = int(os.getenv("INPUT_DEVICE_INDEX", "-1"))  # PortAudio fallback only
VAD_AGGRESSIVENESS    = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
MAX_SILENCE_SECONDS   = float(os.getenv("MAX_SILENCE_SECONDS", "0.7"))
SESSION_IDLE_TIMEOUT  = float(os.getenv("SESSION_IDLE_TIMEOUT", "45"))
MAX_HISTORY_TURNS     = int(os.getenv("MAX_HISTORY_TURNS", "12"))  # user+assistant pairs kept in prompt
SAMPLE_RATE           = 16000
FRAME_MS              = 30
FRAME_BYTES           = int(SAMPLE_RATE * (FRAME_MS / 1000.0) * 2)  # 16-bit mono
PRE_SPEECH_PAD_MS     = int(os.getenv("PRE_SPEECH_PAD_MS", "120"))
PRE_SPEECH_FRAMES     = max(0, int(round(PRE_SPEECH_PAD_MS / FRAME_MS))) if PRE_SPEECH_PAD_MS > 0 else 0

USE_ALSA_CAPTURE      = os.getenv("USE_ALSA_CAPTURE", "1") == "1"
ALSA_DEVICE           = os.getenv("ALSA_DEVICE", "").strip()        # e.g. "plughw:2,0" or "auto"
VOICEBOT_PLAY_DEVICE  = os.getenv("VOICEBOT_PLAY_DEVICE", "").strip()

# ---- Wakeword & window ----
REQUIRE_WAKEWORD      = os.getenv("REQUIRE_WAKEWORD", "0") == "1"
WAKE_ALWAYS_ON        = os.getenv("WAKE_ALWAYS_ON", "0") == "1"
WAKE_SENSITIVITY      = os.getenv("WAKE_SENSITIVITY", "medium").strip().lower()  # low|medium|high
WAKE_WINDOW_SEC       = float(os.getenv("WAKE_WINDOW_SEC", "10"))
POST_TTS_ACCEPT_SEC   = float(os.getenv("POST_TTS_ACCEPT_SEC", "4"))  # keep window open after TTS
WAKEWORD              = os.getenv("WAKEWORD", "hey tars").strip()
WAKE_ALIASES          = [a.strip() for a in os.getenv("WAKE_ALIASES", "hey tars,tars").split(",") if a.strip()]
WAKEWORD_TIMEOUT      = float(os.getenv("WAKEWORD_TIMEOUT", "45"))
WAKE_DEBUG_WAV        = os.getenv("WAKE_DEBUG_WAV", "0") == "1"  # <-- fixed
WAKE_RATIO_THRESHOLD  = int(os.getenv("WAKE_RATIO_THRESHOLD", "80"))
WAKE_NEAR_MISS_DELTA  = int(os.getenv("WAKE_NEAR_MISS_DELTA", "5"))
USE_ACOUSTIC_WAKE     = os.getenv("USE_ACOUSTIC_WAKE", "0") == "1"
ACOUSTIC_WAKE_MODEL   = os.getenv("ACOUSTIC_WAKE_MODEL", "").strip()
ACOUSTIC_WAKE_THRESHOLD = float(os.getenv("ACOUSTIC_WAKE_THRESHOLD", "0.65"))

# Ollama / LLM
OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL          = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct")
SYSTEM_PROMPT         = os.getenv(
    "SYSTEM_PROMPT",
    "You are TARS. Reply concisely and only with the final answer. "
    "NEVER include chain-of-thought, analysis, or <think> blocks."
)

# LLM decoding/latency controls
LLM_MAX_TOKENS        = int(os.getenv("LLM_MAX_TOKENS", "100"))
LLM_TEMP              = float(os.getenv("LLM_TEMP", "0.4"))
LLM_TOP_P             = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_REPEAT_PENALTY    = float(os.getenv("LLM_REPEAT_PENALTY", "1.05"))
LLM_CTX               = int(os.getenv("LLM_CTX", "2048"))

# Whisper / STT
WHISPER_SIZE          = os.getenv("WHISPER_SIZE", "tiny.en")      # tiny.en, base.en, small.en...
WHISPER_DEVICE        = os.getenv("WHISPER_DEVICE", "auto")       # "auto", "cuda", "cpu"
WHISPER_COMPUTE       = os.getenv("WHISPER_COMPUTE", "int8")      # "int8", "int8_float32", "float16", "float32"
ASR_LANGUAGE          = os.getenv("ASR_LANGUAGE", "en")

# Barge-in (defaults OFF per your preference now)
ALLOW_BARGE_IN        = os.getenv("ALLOW_BARGE_IN", "0") == "1"
BARGE_IN_FRAMES       = int(os.getenv("BARGE_IN_FRAMES", "10"))   # only used if barge-in enabled

# Memory
ENABLE_MEMORY         = os.getenv("ENABLE_MEMORY", "1") == "1"
MEMORY_FILE           = os.getenv("MEMORY_FILE", "memories.json")

# Ops
OFFLINE_MODE          = os.getenv("OFFLINE_MODE", "0") == "1"

# Bluetooth / Speaker
AUTO_BT_CONNECT       = os.getenv("AUTO_BT_CONNECT", "0") == "1"
BT_SPEAKER_MAC        = os.getenv("BT_SPEAKER_MAC", "").strip()
BT_CONNECT_RETRIES    = int(os.getenv("BT_CONNECT_RETRIES", "5"))
BT_CONNECT_DELAY      = float(os.getenv("BT_CONNECT_DELAY", "2"))
BT_PROFILE            = os.getenv("BT_PROFILE", "a2dp_sink").strip()

# Piper TTS
PIPER_BINARY          = os.getenv("PIPER_BINARY", "piper")
PIPER_VOICE           = os.getenv("PIPER_VOICE", "").strip()  # e.g., /home/anthony/voices/TARS.onnx
PIPER_LENGTH_SCALE    = os.getenv("PIPER_LENGTH_SCALE", "0.9")
PIPER_NOISE_SCALE     = os.getenv("PIPER_NOISE_SCALE", "0.667")
PIPER_NOISE_W         = os.getenv("PIPER_NOISE_W", "0.8")

# ---------------- State ----------------
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(message)s")
logger = logging.getLogger("voicebot")


def log_event(event: str, **fields) -> None:
    payload = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event": event,
    }
    payload.update(fields)
    if LOG_JSON:
        logger.info(json.dumps(payload, ensure_ascii=False))
    else:
        meta = ", ".join(f"{k}={v}" for k, v in payload.items() if k != "event")
        logger.info("%s | %s", event, meta)


session_active: bool = False
last_activity_ts: float = 0.0
audio_q: "queue.Queue[bytes]" = queue.Queue()
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
acoustic_wake = None  # initialized after class definitions

# ---- Output sanitization: remove chain-of-thought / <think> blocks ----
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _drain_queue(q: "queue.Queue[bytes]") -> None:
    """Remove any buffered frames so new sessions start cleanly."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return

def clean_response(text: str) -> str:
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    # Remove obvious reasoning prefixes at line starts
    lines = [ln for ln in text.splitlines() if not ln.strip().lower().startswith(("thought", "reason", "analysis"))]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pcm_stats(pcm: bytes) -> Dict[str, float]:
    if not pcm:
        return {"samples": 0, "rms": 0.0, "peak": 0, "sec": 0.0}
    arr = np.frombuffer(pcm, dtype=np.int16)
    if arr.size == 0:
        return {"samples": 0, "rms": 0.0, "peak": 0, "sec": 0.0}
    float_arr = arr.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(float_arr ** 2)))
    peak = int(np.max(np.abs(arr)))
    sec = float(len(pcm) / (2 * SAMPLE_RATE))
    return {"samples": int(arr.size), "rms": rms, "peak": peak, "sec": sec}


def _dump_wav(prefix: str, pcm: bytes) -> Optional[str]:
    if not pcm:
        return None
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"/tmp/{prefix}_{ts}.wav"
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    log_event("debug_wav_dump", path=path, prefix=prefix, **_pcm_stats(pcm))
    return path


def _maybe_dump_near_miss(tag: str, pcm: bytes, ratio: int, threshold: int) -> None:
    if not WAKE_DEBUG_WAV or not pcm or threshold <= 0:
        return
    if ratio >= max(0, threshold - WAKE_NEAR_MISS_DELTA):
        _dump_wav(tag, pcm)


def _resolve_exec(binary: str) -> Optional[str]:
    if not binary:
        return None
    if os.path.isabs(binary):
        return binary if os.path.exists(binary) else None
    return shutil.which(binary)

print("Loading Whisper model...")
whisper_model = WhisperModel(WHISPER_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)

# ---------------- Helpers ----------------
def sh(cmd: list[str], check: bool = False):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)

def bt_mac_to_sink_name(mac: str, profile: str) -> str:
    return f"bluez_output.{mac.replace(':','_').upper()}.{profile}"

def bt_autoconnect() -> None:
    if not AUTO_BT_CONNECT or not BT_SPEAKER_MAC:
        return
    print(f"[BT] Auto-connecting {BT_SPEAKER_MAC} ...")
    try:
        script = (
            "power on\nagent NoInputNoOutput\ndefault-agent\n"
            f"trust {BT_SPEAKER_MAC}\nconnect {BT_SPEAKER_MAC}\nquit\n"
        )
        sh(["bash", "-lc", f"printf '%s' \"{script}\" | bluetoothctl"])
        sink_name = bt_mac_to_sink_name(BT_SPEAKER_MAC, BT_PROFILE)
        for _ in range(BT_CONNECT_RETRIES):
            time.sleep(BT_CONNECT_DELAY)
            sinks = sh(["bash", "-lc", "pactl list short sinks"]).stdout
            if sink_name in sinks:
                sh(["bash", "-lc", f"pactl set-default-sink {sink_name}"])
                print(f"[BT] Connected sink: {sink_name}")
                return
            sh(["bash", "-lc", f"bluetoothctl connect {BT_SPEAKER_MAC} >/dev/null 2>&1 || true"])
        print("[BT] Warning: sink not found")
    except Exception as e:
        print(f"[BT] Error: {e}")

def _ollama_chat(msgs: List[Dict[str, str]]) -> str:
    """
    Use concise decoding options and strip any <think> leakage.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": msgs,
        "stream": False,
        "options": {
            "num_predict": LLM_MAX_TOKENS,
            "temperature": LLM_TEMP,
            "top_p": LLM_TOP_P,
            "repeat_penalty": LLM_REPEAT_PENALTY,
            "num_ctx": LLM_CTX,
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        raw = (r.json().get("message", {}) or {}).get("content", "").strip()
        return clean_response(raw)
    except Exception as e:
        return f"[error] Ollama API: {e}"

def _ollama_summarize(text: str) -> str:
    msgs = [
        {"role": "system", "content": "You produce concise factual summaries."},
        {"role": "user",   "content": "Summarize in 2–4 bullet points:\n\n" + text},
    ]
    return _ollama_chat(msgs)

def _append_memory(summary: str):
    if not ENABLE_MEMORY or not summary:
        return
    entry = {"ts": int(time.time()), "summary": summary}
    try:
        data = []
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        data.append(entry)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[memory] write error: {e}")

# ---------------- Audio device detection ----------------
_AIRHUG_REGEX = re.compile(r"card\s+(\d+):\s.*AIRHUG.*device\s+(\d+):", re.IGNORECASE)

def find_airhug_device() -> Optional[str]:
    """Return 'hw:X,Y' for AIRHUG, or None."""
    try:
        out = subprocess.check_output(["arecord", "-l"], text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        out = e.output
    m = _AIRHUG_REGEX.search(out or "")
    if m:
        return f"hw:{m.group(1)},{m.group(2)}"
    return None

def _to_plug(dev: Optional[str]) -> Optional[str]:
    if not dev:
        return None
    return dev if dev.startswith("plughw:") else dev.replace("hw:", "plughw:", 1)

def resolve_alsa_device_for_capture(dev_env: str) -> Optional[str]:
    """
    Capture device resolution:
      1) If env set and not 'auto' → honor it.
      2) Else try AIRHUG and wrap as plughw for resampling.
      3) Else None (arecord default).
    """
    if dev_env and dev_env.lower() != "auto":
        return dev_env
    hw = find_airhug_device()
    if hw:
        return _to_plug(hw)  # prefer plug to ensure 16k mono works
    return None

def resolve_play_device() -> Optional[str]:
    """
    Playback priority:
      1) VOICEBOT_PLAY_DEVICE if provided and not 'auto'
      2) ALSA capture device (resolved) if present
      3) None → system default
    """
    if VOICEBOT_PLAY_DEVICE and VOICEBOT_PLAY_DEVICE.lower() != "auto":
        return VOICEBOT_PLAY_DEVICE
    cap = resolve_alsa_device_for_capture(ALSA_DEVICE or "auto")
    if cap:
        return cap
    return None

# ---------------- ALSA Capture (with fallback to plughw) ----------------
class ALSACapture:
    def __init__(self, device: Optional[str], rate: int = SAMPLE_RATE):
        self.configured = resolve_alsa_device_for_capture(device or "auto")  # prefer plughw if auto
        self.device = self.configured
        self.rate = rate
        self.proc: Optional[subprocess.Popen] = None
        self.running = False

    def _spawn(self, dev: Optional[str]) -> subprocess.Popen:
        base = ["arecord", "-q", "-f", "S16_LE", "-c", "1", "-r", str(self.rate)]
        if dev:
            base += ["-D", dev]
        print(f"[ALSA] Using capture device: {dev or '(default)'}")
        return subprocess.Popen(base, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _fallback_target(self) -> Optional[str]:
        # Try plug version of configured → plug of detected AIRHUG → None
        for candidate in [self.configured, find_airhug_device()]:
            if candidate:
                plug = _to_plug(candidate)
                if plug != self.device:
                    return plug
        return None

    def start(self):
        self.running = True
        self.proc = self._spawn(self.device)

        def reader():
            buf = FRAME_BYTES * 10
            last_bytes_time = time.time()
            fallback_done = False

            while self.running:
                if not self.proc or not self.proc.stdout:
                    break

                chunk = self.proc.stdout.read(buf)
                if chunk:
                    audio_q.put(chunk)
                    last_bytes_time = time.time()
                    continue

                # If arecord exited, show stderr and try fallback once
                if self.proc.poll() is not None:
                    err = (self.proc.stderr.read() or "").strip()
                    if err:
                        print(f"[ALSA/arecord stderr]\n{err}", file=sys.stderr)
                    if not fallback_done:
                        fallback_done = True
                        tgt = self._fallback_target()
                        if tgt:
                            try:
                                print(f"[ALSA] Retrying with {tgt} …")
                                self.proc = self._spawn(tgt)
                                self.device = tgt
                                continue
                            except Exception as e:
                                print(f"[ALSA] Fallback spawn failed: {e}", file=sys.stderr)
                    break

                # Process alive but no data for >1s → attempt fallback to plughw once
                if (time.time() - last_bytes_time) > 1.0 and not fallback_done:
                    tgt = self._fallback_target()
                    if tgt:
                        fallback_done = True
                        try:
                            print(f"[ALSA] No data at {self.device}; switching to {tgt} …")
                            try:
                                self.proc.terminate()
                            except Exception:
                                pass
                            self.proc = self._spawn(tgt)
                            self.device = tgt
                            last_bytes_time = time.time()
                            continue
                        except Exception as e:
                            print(f"[ALSA] Fallback error: {e}", file=sys.stderr)

                time.sleep(0.02)

        threading.Thread(target=reader, daemon=True).start()

    def stop(self):
        self.running = False
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass


class AcousticWakeDetector:
    def __init__(self, model_path: str, threshold: float):
        self.enabled = False
        self.threshold = threshold
        self.model_path = model_path
        self._model = None
        if not USE_ACOUSTIC_WAKE:
            return
        if not model_path:
            print("[Wakeword] USE_ACOUSTIC_WAKE=1 but no ACOUSTIC_WAKE_MODEL provided", file=sys.stderr)
            return
        if not Path(model_path).exists():
            print(f"[Wakeword] Model not found at {model_path}", file=sys.stderr)
            return
        if WakewordModel is None:
            print("[Wakeword] openwakeword not installed; run `uv add openwakeword`", file=sys.stderr)
            return
        try:
            self._model = WakewordModel(wakeword_models=[model_path])
            self.enabled = True
            log_event("wake_acoustic_ready", model=os.path.basename(model_path), threshold=threshold)
        except Exception as exc:
            print(f"[Wakeword] Failed to load acoustic model: {exc}", file=sys.stderr)

    def detect(self, pcm: bytes) -> Tuple[bool, float]:
        if not self.enabled or not pcm:
            return False, 0.0
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            result = self._model.predict(audio)
        except Exception as exc:
            log_event("wake_acoustic_error", error=str(exc))
            return False, 0.0
        score = 0.0
        if isinstance(result, dict):
            score = float(max(result.values()) if result else 0.0)
        elif isinstance(result, (list, tuple, np.ndarray)) and len(result):
            score = float(np.max(result))
        triggered = score >= self.threshold
        log_event("wake_acoustic_score", score=score, triggered=triggered)
        return triggered, score


# Instantiate after class definition so Python has the symbol
acoustic_wake = AcousticWakeDetector(ACOUSTIC_WAKE_MODEL, ACOUSTIC_WAKE_THRESHOLD)

# ---------------- Audio Logic ----------------
def record_until_silence(min_ms: int = 0, pad_ms: int = 0) -> bytes:
    """
    Capture until VAD says we've hit silence for MAX_SILENCE_SECONDS.
    Ensures at least `min_ms` of voiced audio and appends `pad_ms` of
    extra audio after the last voiced frame to give STT more context.
    """
    global last_activity_ts
    voiced = bytearray()
    silence_start = None
    min_frames = max(0, int(min_ms // FRAME_MS))
    pad_frames = max(0, int(pad_ms // FRAME_MS))
    voiced_frames = 0
    trailing_frames = 0
    pad_buffer = bytearray()
    max_pad_bytes = pad_frames * FRAME_BYTES
    pre_buffer = deque(maxlen=PRE_SPEECH_FRAMES) if PRE_SPEECH_FRAMES else None
    pulls = 0

    while session_active:
        try:
            chunk = audio_q.get(timeout=1)
        except queue.Empty:
            pulls += 1
            if pulls % 5 == 0:
                print(f"[VAD] waiting for audio... ({pulls})")
            if time.time() - last_activity_ts > SESSION_IDLE_TIMEOUT:
                break
            continue

        for i in range(0, len(chunk), FRAME_BYTES):
            frame = chunk[i:i + FRAME_BYTES]
            if len(frame) < FRAME_BYTES:
                break

            if vad.is_speech(frame, SAMPLE_RATE):
                if pre_buffer:
                    while pre_buffer:
                        voiced.extend(pre_buffer.popleft())
                voiced.extend(frame)
                voiced_frames += 1
                trailing_frames = 0
                silence_start = None
                if pad_buffer:
                    pad_buffer.clear()
                last_activity_ts = time.time()
            else:
                if voiced:
                    trailing_frames += 1
                    if max_pad_bytes and len(pad_buffer) < max_pad_bytes:
                        pad_buffer.extend(frame)
                    enough_audio = (voiced_frames >= min_frames)
                    if enough_audio and trailing_frames >= pad_frames:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= MAX_SILENCE_SECONDS:
                            return bytes(voiced + pad_buffer)
                elif pre_buffer is not None:
                    pre_buffer.append(frame)

        if time.time() - last_activity_ts > SESSION_IDLE_TIMEOUT:
            break

    if not voiced_frames:
        print("[VAD] no voiced audio captured.")
    return bytes(voiced + pad_buffer)

def capture_fixed(duration_sec: float = 3.0) -> bytes:
    """Raw grab via arecord for debugging/calibration."""
    dev = resolve_alsa_device_for_capture(ALSA_DEVICE or "auto")
    seconds = max(1, int(round(duration_sec)))
    cmd = ["arecord", "-q", "-f", "S16_LE", "-c", "1", "-r", str(SAMPLE_RATE), "-d", str(seconds), "-"]
    if dev:
        cmd[1:1] = ["-D", dev]
    try:
        data = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        log_event("capture_fixed", seconds=seconds, **_pcm_stats(data))
        return data
    except subprocess.CalledProcessError as e:
        print(f"[Debug] capture_fixed error: {e.output}", file=sys.stderr)
        return b""


def fixed_capture_3s() -> bytes:
    return capture_fixed(3)

def transcribe(pcm16: bytes) -> str:
    if not pcm16:
        return ""
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    started = time.time()
    segments, _ = whisper_model.transcribe(
        audio,
        vad_filter=False,                 # we already gate with WebRTC VAD
        language=ASR_LANGUAGE,            # lock language for speed/accuracy
        beam_size=1,                      # greedy is fastest
        condition_on_previous_text=False, # don't spend time linking segments
        without_timestamps=True,
    )
    text = "".join(seg.text for seg in segments).strip()
    log_event("asr_transcript", latency_ms=int((time.time() - started) * 1000), text=text, **_pcm_stats(pcm16))
    return text

# ---------------- TTS (Piper) ----------------
def synthesize_tts_piper(text: str) -> Optional[str]:
    if not PIPER_VOICE:
        print("[TTS] PIPER_VOICE not set; skipping TTS.", file=sys.stderr)
        return None
    piper_exec = _resolve_exec(PIPER_BINARY)
    if not piper_exec:
        msg = f"Piper binary '{PIPER_BINARY}' not found in PATH"
        print(f"[TTS] {msg}", file=sys.stderr)
        log_event("tts_error", error=msg)
        return None
    fd, out_path = tempfile.mkstemp(prefix="voicebot_tts_", suffix=".wav")
    os.close(fd)
    cmd = [
        piper_exec,
        "--model", PIPER_VOICE,
        "--length_scale", str(PIPER_LENGTH_SCALE),
        "--noise_scale", str(PIPER_NOISE_SCALE),
        "--noise_w", str(PIPER_NOISE_W),
        "--output_file", out_path,
    ]
    try:
        print(f"[TTS] Piper synth → {out_path}")
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = p.communicate(text, timeout=60)
        if p.returncode != 0:
            print(f"[TTS] Piper failed: {stderr}", file=sys.stderr)
            os.unlink(out_path)
            log_event("tts_error", error=stderr.strip(), code=p.returncode)
            return None
        log_event("tts_synth", path=out_path, chars=len(text))
        return out_path
    except Exception as e:
        print(f"[TTS] Piper error: {e}", file=sys.stderr)
        try:
            os.unlink(out_path)
        except Exception:
            pass
        log_event("tts_error", error=str(e))
        return None

def play_wav(path: str):
    dev = resolve_play_device()
    cmd = ["aplay", path]
    if dev:
        cmd[1:1] = ["-D", dev]
    print("🔊 Playing:", " ".join(cmd))
    log_event("tts_play", device=dev or "default", path=path)

    proc = subprocess.Popen(cmd)
    if not ALLOW_BARGE_IN:
        proc.wait()
        return

    # Interrupt if we detect speech for BARGE_IN_FRAMES consecutive frames
    consecutive = 0
    try:
        while proc.poll() is None:
            time.sleep(FRAME_MS / 1000.0)
            # Non-blocking check of mic queue for frames; we'll consume a few here
            try:
                chunk = audio_q.get_nowait()
            except queue.Empty:
                continue

            for i in range(0, len(chunk), FRAME_BYTES):
                frame = chunk[i:i + FRAME_BYTES]
                if len(frame) < FRAME_BYTES:
                    break
                if vad.is_speech(frame, SAMPLE_RATE):
                    consecutive += 1
                    if consecutive >= BARGE_IN_FRAMES:
                        print("[BargeIn] User speaking → interrupting TTS")
                        proc.terminate()
                        return
                else:
                    consecutive = 0
    finally:
        if proc.poll() is None:
            proc.wait()

def speak(text: str):
    wav = synthesize_tts_piper(text)
    if wav:
        try:
            play_wav(wav)
        finally:
            try:
                os.unlink(wav)
            except Exception:
                pass
    else:
        print("[TTS] (spoken reply skipped) " + text)

# ---------------- Wakeword Matching ----------------
_WAKE_FILLERS = ("hey", "hi", "uh", "um", "ok", "okay", "yo", "hello", "alright")
_SENS_RATIO_DELTA = {"low": -5, "medium": 0, "high": 5}  # positive → stricter, negative → looser


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def _wake_variants(overrides: Optional[List[str]] = None) -> List[str]:
    """
    Return the set of phrases that should count as a wakeword.
    Includes the configured WAKEWORD, env-provided aliases, and the final token.
    """
    base_list = overrides if overrides is not None else WAKE_ALIASES
    candidates = [WAKEWORD] + (base_list or [])
    seen = set()
    variants: List[str] = []
    for cand in candidates:
        word = cand.strip()
        if not word:
            continue
        key = word.lower()
        if key in seen:
            continue
        variants.append(word)
        seen.add(key)
    if WAKEWORD:
        last = WAKEWORD.split()[-1]
        if last and last.lower() not in seen:
            variants.append(last)
    return variants


def _lev(a: str, b: str) -> int:
    if rapidfuzz_lev:
        return int(rapidfuzz_lev.distance(a, b))
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[n]


def _ratio(a: str, b: str) -> int:
    if rapidfuzz_fuzz:
        return int(rapidfuzz_fuzz.partial_ratio(a, b))
    if not a and not b:
        return 100
    dist = _lev(a, b)
    denom = max(len(a), len(b), 1)
    return max(0, int((1 - (dist / denom)) * 100))


def _strip_fillers(text: str) -> str:
    tokens = text.split()
    while tokens and tokens[0] in _WAKE_FILLERS:
        tokens.pop(0)
    return " ".join(tokens)


def _match_wake(text: str, wake: str, sensitivity: str, aliases: List[str] | None = None) -> Tuple[bool, int, int]:
    nt = _strip_fillers(_norm(text))
    nw = _norm(wake)
    alias_src = aliases if aliases is not None else _wake_variants()
    cands = [nw] + ([_norm(a) for a in alias_src])

    best_ratio = 0
    for w in cands:
        if not w:
            continue
        if w in nt:
            log_event("wake_transcript_match", method="substring", candidate=w)
            return True, 100, 0
        ratio = _ratio(nt, w)
        best_ratio = max(best_ratio, ratio)

    last = nw.split()[-1] if nw else ""
    if sensitivity != "low" and last and last in nt:
        log_event("wake_transcript_match", method="last_token", candidate=last)
        return True, 90, 0

    threshold = max(50, WAKE_RATIO_THRESHOLD + _SENS_RATIO_DELTA.get(sensitivity, 0))
    if last:
        best_ratio = max(best_ratio, _ratio(nt, last))
    matched = best_ratio >= threshold
    log_event("wake_transcript_ratio", ratio=best_ratio, threshold=threshold, matched=matched)
    return matched, best_ratio, threshold


def _strip_wake(text: str, wake: str) -> str:
    t = text.strip()
    variants = _wake_variants()
    if wake and wake not in variants:
        variants.append(wake)
    for variant in variants:
        if not variant:
            continue
        pattern = re.compile(rf"^\s*{re.escape(variant)}[\s,.:;!?-]*", re.IGNORECASE)
        stripped = re.sub(pattern, "", t, count=1).strip()
        if stripped != t:
            return stripped
    return t

def _prune_history(msgs: List[Dict[str, str]]) -> None:
    """
    Keep the prompt bounded so Ollama does not reprocess unlimited turns.
    """
    if MAX_HISTORY_TURNS <= 0:
        return
    max_messages = MAX_HISTORY_TURNS * 2  # user+assistant pairs (system prompt excluded)
    excess = len(msgs) - 1 - max_messages
    if excess > 0:
        del msgs[1:1 + excess]

# ---------------- Session ----------------
def voice_session():
    """
    Always-on state machine with wake windows and continuous conversation modes.
    """
    global session_active, last_activity_ts
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    convo_log: List[str] = []
    last_activity_ts = time.time()
    print("I'm listening.")
    alsa = None

    try:
        if USE_ALSA_CAPTURE:
            alsa = ALSACapture(ALSA_DEVICE, SAMPLE_RATE)
            alsa.start()
            _drain_queue(audio_q)
            _run_dialog(messages, convo_log)
        else:
            # Optional PortAudio fallback
            import sounddevice as sd
            def cb(indata, frames, time_info, status):
                if status:
                    print(f"[sounddevice] {status}", file=sys.stderr)
                audio_q.put((indata[:, 0] * 32767).astype(np.int16).tobytes())
            _drain_queue(audio_q)
            with sd.InputStream(device=INPUT_DEVICE_INDEX, channels=1, samplerate=SAMPLE_RATE,
                                dtype="float32", callback=cb):
                _run_dialog(messages, convo_log)
    finally:
        if ENABLE_MEMORY and convo_log:
            try:
                _append_memory(_ollama_summarize("\n".join(convo_log)))
            except Exception as e:
                print(f"[memory] summarize error: {e}")
        session_active = False
        if alsa:
            alsa.stop()

def _answer_and_speak(messages: List[Dict[str, str]], convo_log: List[str], on_done: Optional[Callable[[], None]] = None):
    reply = _ollama_chat(messages)
    print(f"[Bot] {reply}")
    convo_log.append(f"Bot: {reply}")
    messages.append({"role": "assistant", "content": reply})
    _prune_history(messages)
    speak(reply)
    if on_done:
        try:
            on_done()
        except Exception as e:
            print(f"[post-tts] callback error: {e}", file=sys.stderr)

def _run_dialog(messages: List[Dict[str, str]], convo_log: List[str]):
    global session_active

    # --- Mode/state for always-on ---
    accepting = False
    accept_until = 0.0

    def begin_accept_window(now: float):
        nonlocal accepting, accept_until
        accepting = True
        accept_until = now + WAKE_WINDOW_SEC
        print(f"[Wake] WAKE detected → accepting for {WAKE_WINDOW_SEC}s")

    def maybe_extend_window(now: float, extend_sec: float = None):
        nonlocal accept_until
        extend = WAKE_WINDOW_SEC if extend_sec is None else extend_sec
        accept_until = max(accept_until, now + extend)

    def post_tts_hold():
        # keep accept window open a bit longer after bot finishes talking
        if REQUIRE_WAKEWORD and WAKE_ALWAYS_ON:
            now = time.time()
            maybe_extend_window(now, POST_TTS_ACCEPT_SEC)
            print(f"[Wake] extended {POST_TTS_ACCEPT_SEC}s after TTS")

    # --- Legacy single-shot wake mode ---
    if REQUIRE_WAKEWORD and not WAKE_ALWAYS_ON:
        print(f"[Wakeword] Say: '{WAKEWORD}'")
        start_ts = time.time()
        while session_active:
            if time.time() - start_ts > WAKEWORD_TIMEOUT:
                print("[Wakeword] Timeout; no wakeword heard")
                session_active = False
                return

            pcm = record_until_silence(min_ms=500, pad_ms=150)
            if not session_active or not pcm:
                continue

            if WAKE_DEBUG_WAV:
                ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                p = f"/tmp/wake_{ts}.wav"
                with wave.open(p, "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
                    w.writeframes(pcm)
                print(f"[WakeDebug] wrote {p}")

            acoustic_triggered = False
            acoustic_score = 0.0
            if acoustic_wake.enabled:
                acoustic_triggered, acoustic_score = acoustic_wake.detect(pcm)
                if acoustic_triggered:
                    log_event("wake_detected", method="acoustic", score=acoustic_score)

            text = transcribe(pcm)
            if not text:
                continue

            print(f"[Wakeword->ASR] {text}")
            matched, ratio, threshold = _match_wake(
                text,
                WAKEWORD,
                WAKE_SENSITIVITY,
            )
            if matched or acoustic_triggered:
                cleaned = _strip_wake(text, WAKEWORD)
                if cleaned:
                    print(f"[User] {cleaned}")
                    convo_log.append(f"User: {cleaned}")
                    messages.append({"role": "user", "content": cleaned})
                    _prune_history(messages)
                    _answer_and_speak(messages, convo_log, on_done=post_tts_hold)
                break
            else:
                _maybe_dump_near_miss("wake_near", pcm, ratio, threshold)

    # --- Always-on or normal conversation loop ---
    idle_tries = 0
    while session_active:
        pcm = record_until_silence()

        if not session_active:
            break

        if not pcm:
            idle_tries += 1
            if idle_tries >= 3:
                idle_tries = 0
                print("[Debug] capturing fixed 3s window…")
                pcm = fixed_capture_3s()
                if not pcm:
                    continue
            else:
                continue

        idle_tries = 0

        now = time.time()
        acoustic_triggered = False
        acoustic_score = 0.0
        if REQUIRE_WAKEWORD and WAKE_ALWAYS_ON and not accepting and acoustic_wake.enabled:
            acoustic_triggered, acoustic_score = acoustic_wake.detect(pcm)

        text = transcribe(pcm)
        if not text:
            stats = _pcm_stats(pcm)
            print(f"[ASR] empty transcript (samples={stats['samples']}, rms={stats['rms']:.6f}, peak={stats['peak']})")
            log_event("asr_empty", **stats)
            if WAKE_DEBUG_WAV and stats["samples"]:
                _dump_wav("asr_empty", pcm)
            if acoustic_triggered:
                begin_accept_window(now)
                log_event("wake_detected", method="acoustic", score=acoustic_score)
                continue
            continue

        # If wakeword is required, decide whether this utterance should trigger or be treated as a command
        if REQUIRE_WAKEWORD:
            if WAKE_ALWAYS_ON:
                if not accepting:
                    # Look for a wake in this utterance
                    matched, ratio, threshold = _match_wake(
                        text,
                        WAKEWORD,
                        WAKE_SENSITIVITY,
                    )
                    if matched or acoustic_triggered:
                        if acoustic_triggered:
                            begin_accept_window(now)
                            log_event("wake_detected", method="acoustic", score=acoustic_score)
                        else:
                            begin_accept_window(now)
                            log_event("wake_detected", method="transcript", score=ratio)
                        cleaned = _strip_wake(text, WAKEWORD)
                        # If user said "hey tars, <command>" handle immediately
                        if cleaned:
                            print(f"[User] {cleaned}")
                            convo_log.append(f"User: {cleaned}")
                            messages.append({"role": "user", "content": cleaned})
                            _prune_history(messages)
                            _answer_and_speak(messages, convo_log, on_done=post_tts_hold)
                            maybe_extend_window(now)  # keep window open for follow-up
                        continue  # go back to listening (window is open)
                    else:
                        # Not a wake; ignore in wake-wait state
                        print(f"[WakeWait] heard non-wake utterance: {text}")
                        _maybe_dump_near_miss("wake_near", pcm, ratio, threshold)
                        continue
                else:
                    # We are within the accept window
                    if now > accept_until:
                        accepting = False
                        print("[Wake] window expired → waiting for wake")
                        continue
                    # Treat the utterance as a command (strip wake if user repeats it)
                    cleaned = _strip_wake(text, WAKEWORD)
                    user_text = cleaned if cleaned else text
                    print(f"[User] {user_text}")
                    log_event("wake_command", text=user_text)
                    convo_log.append(f"User: {user_text}")
                    messages.append({"role": "user", "content": user_text})
                    _prune_history(messages)
                    _answer_and_speak(messages, convo_log, on_done=post_tts_hold)
                    maybe_extend_window(now)  # extend the window with each command
                    continue
            # else: legacy single-shot path already handled above

        # If wakeword not required, or legacy mode after initial wake:
        print(f"[User] {text}")
        convo_log.append(f"User: {text}")
        messages.append({"role": "user", "content": text})
        _prune_history(messages)
        _answer_and_speak(messages, convo_log, on_done=post_tts_hold)


def _snr_db(speech_rms: float, noise_rms: float) -> float:
    noise = max(noise_rms, 1e-6)
    speech = max(speech_rms, 1e-6)
    return 20.0 * math.log10(speech / noise)


def calibrate_vad(noise_sec: int = 5, speech_sec: int = 4) -> None:
    print(f"[Calibrate] Recording {noise_sec}s of background noise. Stay quiet…")
    time.sleep(1)
    noise = capture_fixed(noise_sec)
    noise_stats = _pcm_stats(noise)

    input("Press Enter, then speak your wakephrase for a few seconds…")
    speech = capture_fixed(speech_sec)
    speech_stats = _pcm_stats(speech)

    snr = _snr_db(speech_stats["rms"], noise_stats["rms"])
    if snr < 6:
        suggested_vad = 3
    elif snr < 12:
        suggested_vad = 2
    else:
        suggested_vad = 1
    max_silence = round(min(1.5, max(0.6, 1.2 - (snr / 40.0))), 2)
    wake_window = round(max(4.0, min(15.0, (speech_stats["sec"] or 1.5) * 3)), 1)

    print("\n[Calibrate] Results:")
    print(f"  Noise RMS:   {noise_stats['rms']:.5f} (samples={noise_stats['samples']})")
    print(f"  Speech RMS:  {speech_stats['rms']:.5f} (samples={speech_stats['samples']})")
    print(f"  SNR (dB):    {snr:.1f}")
    print("Suggested settings:")
    print(f"  VAD_AGGRESSIVENESS={suggested_vad}")
    print(f"  MAX_SILENCE_SECONDS={max_silence}")
    print(f"  WAKE_WINDOW_SEC={wake_window}")
    log_event(
        "calibration",
        noise_rms=noise_stats["rms"],
        speech_rms=speech_stats["rms"],
        snr_db=snr,
        suggested_vad=suggested_vad,
        suggested_max_silence=max_silence,
        suggested_wake_window=wake_window,
    )

# ---------------- MQTT ----------------
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"[MQTT] Connected rc={reason_code}")
    client.subscribe(MQTT_PRESENCE_TOPIC, qos=1)

def on_message(client, userdata, msg):
    global session_active
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError:
        return
    present = bool(payload.get("present", False))
    score = float(payload.get("score", 0.0))
    if present and score >= 0.6 and not session_active:
        print("[Session] Presence detected; arming mic")
        session_active = True
        threading.Thread(target=voice_session, daemon=True).start()
    elif not present and session_active:
        print("[Session] Ending session")
        session_active = False

def start_mqtt_loop():
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="voicebot", protocol=mqtt.MQTTv311)
    if MQTT_USER:
        c.username_pw_set(MQTT_USER, MQTT_PASS)
    c.on_connect = on_connect
    c.on_message = on_message
    c.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    c.loop_start()
    return c

# ---------------- Main ----------------
def main():
    bt_autoconnect()  # obeys AUTO_BT_CONNECT; harmless when 0
    print("[Voicebot] Ready.")
    mqtt_client = None
    run_locally = (not ENABLE_PRESENCE) or OFFLINE_MODE

    if not run_locally:
        try:
            mqtt_client = start_mqtt_loop()
            print(f"[Voicebot] Subscribed to {MQTT_PRESENCE_TOPIC}")
        except Exception as e:
            print(f"[MQTT] Connect failed: {e}")
            run_locally = True

    try:
        if run_locally:
            restart_delay = 1.0
            while True:
                global session_active
                session_active = True
                started = time.time()
                try:
                    voice_session()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"[Voicebot] Session error: {e}", file=sys.stderr)
                if not run_locally:
                    break
                ran_for = time.time() - started
                if ran_for > 30:
                    restart_delay = 1.0
                else:
                    restart_delay = min(restart_delay * 2, 10.0)
                print(f"[Voicebot] Session ended; restarting in {restart_delay:.1f}s")
                time.sleep(restart_delay)
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if mqtt_client:
            mqtt_client.loop_stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TARS Voicebot")
    parser.add_argument("--calibrate-vad", action="store_true", help="run the calibration helper instead of the bot")
    parser.add_argument("--noise-sec", type=int, default=5, help="seconds of background noise to sample")
    parser.add_argument("--speech-sec", type=int, default=4, help="seconds to sample while speaking")
    return parser.parse_args()


def cli():
    args = parse_args()
    if args.calibrate_vad:
        calibrate_vad(noise_sec=args.noise_sec, speech_sec=args.speech_sec)
    else:
        main()


if __name__ == "__main__":
    cli()
