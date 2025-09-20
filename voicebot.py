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
from typing import List, Dict, Optional, Callable

import numpy as np
import webrtcvad
import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import paho.mqtt.client as mqtt

# ---------------- Config ----------------
load_dotenv()

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
SAMPLE_RATE           = 16000
FRAME_MS              = 30
FRAME_BYTES           = int(SAMPLE_RATE * (FRAME_MS / 1000.0) * 2)  # 16-bit mono

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
WAKEWORD_TIMEOUT      = float(os.getenv("WAKEWORD_TIMEOUT", "45"))
WAKE_DEBUG_WAV        = os.getenv("WAKE_DEBUG_WAV", "0") == "1"  # <-- fixed

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
session_active: bool = False
last_activity_ts: float = 0.0
audio_q: "queue.Queue[bytes]" = queue.Queue()
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

# ---- Output sanitization: remove chain-of-thought / <think> blocks ----
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def clean_response(text: str) -> str:
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    # Remove obvious reasoning prefixes at line starts
    lines = [ln for ln in text.splitlines() if not ln.strip().lower().startswith(("thought", "reason", "analysis"))]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text).strip()

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
                voiced.extend(frame)
                voiced_frames += 1
                trailing_frames = 0
                silence_start = None
                last_activity_ts = time.time()
            else:
                if voiced:
                    trailing_frames += 1
                    enough_audio = (voiced_frames >= min_frames)
                    if enough_audio and trailing_frames >= pad_frames:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= MAX_SILENCE_SECONDS:
                            return bytes(voiced)

        if time.time() - last_activity_ts > SESSION_IDLE_TIMEOUT:
            break

    if not voiced_frames:
        print("[VAD] no voiced audio captured.")
    return bytes(voiced)

def fixed_capture_3s() -> bytes:
    """Fallback capture: 3s raw grab regardless of VAD (for debugging)."""
    dev = resolve_alsa_device_for_capture(ALSA_DEVICE or "auto")
    cmd = ["arecord", "-q", "-f", "S16_LE", "-c", "1", "-r", str(SAMPLE_RATE), "-d", "3", "-"]
    if dev:
        cmd[1:1] = ["-D", dev]
    try:
        data = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return data
    except subprocess.CalledProcessError as e:
        print(f"[Debug] fixed_capture_3s error: {e.output}", file=sys.stderr)
        return b""

def transcribe(pcm16: bytes) -> str:
    if not pcm16:
        return ""
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(
        audio,
        vad_filter=False,                 # we already gate with WebRTC VAD
        language=ASR_LANGUAGE,            # lock language for speed/accuracy
        beam_size=1,                      # greedy is fastest
        condition_on_previous_text=False, # don't spend time linking segments
        without_timestamps=True,
    )
    return "".join(seg.text for seg in segments).strip()

# ---------------- TTS (Piper) ----------------
def synthesize_tts_piper(text: str) -> Optional[str]:
    if not PIPER_VOICE:
        print("[TTS] PIPER_VOICE not set; skipping TTS.", file=sys.stderr)
        return None
    fd, out_path = tempfile.mkstemp(prefix="voicebot_tts_", suffix=".wav")
    os.close(fd)
    cmd = [
        PIPER_BINARY,
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
            return None
        return out_path
    except Exception as e:
        print(f"[TTS] Piper error: {e}", file=sys.stderr)
        try:
            os.unlink(out_path)
        except Exception:
            pass
        return None

def play_wav(path: str):
    dev = resolve_play_device()
    cmd = ["aplay", path]
    if dev:
        cmd[1:1] = ["-D", dev]
    print("🔊 Playing:", " ".join(cmd))

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
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()

def _lev(a: str, b: str) -> int:
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

_SENS_FUZZ = {"low": 1, "medium": 2, "high": 3}

def _match_wake(text: str, wake: str, sensitivity: str, aliases: List[str] | None = None) -> bool:
    nt = _norm(text)
    nw = _norm(wake)
    fuzz = _SENS_FUZZ.get(sensitivity, 2)
    cands = [nw] + ([_norm(a) for a in (aliases or [])])

    # Exact/substring match first
    for w in cands:
        if w and w in nt:
            return True

    # Allow last token alone for medium/high only
    last = nw.split()[-1] if nw else ""
    if sensitivity != "low" and last and last in nt:
        return True

    # Fuzzy edit-distance match
    for w in cands + ([last] if last else []):
        if not w:
            continue
        if _lev(nt, w) <= fuzz:
            return True

    return False

def _strip_wake(text: str, wake: str) -> str:
    t = text.strip()
    lw = _norm(wake)
    nt = _norm(t)
    if nt.startswith(lw):
        return t[len(wake):].lstrip(",. :;!?").strip()
    last = lw.split()[-1] if lw else ""
    if last and nt.startswith(last):
        return t[len(last):].lstrip(",. :;!?").strip()
    return t

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
            _run_dialog(messages, convo_log)
        else:
            # Optional PortAudio fallback
            import sounddevice as sd
            def cb(indata, frames, time_info, status):
                if status:
                    print(f"[sounddevice] {status}", file=sys.stderr)
                audio_q.put((indata[:, 0] * 32767).astype(np.int16).tobytes())
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

            text = transcribe(pcm)
            if not text:
                continue

            print(f"[Wakeword->ASR] {text}")
            if _match_wake(text, WAKEWORD, WAKE_SENSITIVITY,
                           aliases=["hey tars", "tars", "hey, tars", "hey-tars"]):
                cleaned = _strip_wake(text, WAKEWORD)
                if cleaned:
                    print(f"[User] {cleaned}")
                    convo_log.append(f"User: {cleaned}")
                    messages.append({"role": "user", "content": cleaned})
                    _answer_and_speak(messages, convo_log, on_done=post_tts_hold)
                break

    # --- Always-on or normal conversation loop ---
    idle_tries = 0
    while session_active:
        pcm = record_until_silence()
        if not pcm:
            idle_tries += 1
            if idle_tries >= 3:
                idle_tries = 0
                print("[Debug] capturing fixed 3s window…")
                pcm = fixed_capture_3s()
        if not session_active or not pcm:
            break

        text = transcribe(pcm)
        if not text:
            print("[ASR] empty transcript")
            continue

        now = time.time()

        # If wakeword is required, decide whether this utterance should trigger or be treated as a command
        if REQUIRE_WAKEWORD:
            if WAKE_ALWAYS_ON:
                if not accepting:
                    # Look for a wake in this utterance
                    if _match_wake(text, WAKEWORD, WAKE_SENSITIVITY,
                                   aliases=["hey tars", "tars", "hey, tars", "hey-tars"]):
                        begin_accept_window(now)
                        cleaned = _strip_wake(text, WAKEWORD)
                        # If user said "hey tars, <command>" handle immediately
                        if cleaned:
                            print(f"[User] {cleaned}")
                            convo_log.append(f"User: {cleaned}")
                            messages.append({"role": "user", "content": cleaned})
                            _answer_and_speak(messages, convo_log, on_done=post_tts_hold)
                            maybe_extend_window(now)  # keep window open for follow-up
                        continue  # go back to listening (window is open)
                    else:
                        # Not a wake; ignore in wake-wait state
                        print(f"[WakeWait] heard non-wake utterance: {text}")
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
                    convo_log.append(f"User: {user_text}")
                    messages.append({"role": "user", "content": user_text})
                    _answer_and_speak(messages, convo_log, on_done=post_tts_hold)
                    maybe_extend_window(now)  # extend the window with each command
                    continue
            # else: legacy single-shot path already handled above

        # If wakeword not required, or legacy mode after initial wake:
        print(f"[User] {text}")
        convo_log.append(f"User: {text}")
        messages.append({"role": "user", "content": text})
        _answer_and_speak(messages, convo_log, on_done=post_tts_hold)

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

    if run_locally:
        global session_active
        session_active = True
        voice_session()

    try:
        while not run_locally:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if mqtt_client:
            mqtt_client.loop_stop()

if __name__ == "__main__":
    main()
