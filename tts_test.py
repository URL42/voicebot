#!/usr/bin/env python3
"""Quick Piper TTS test utility.

Reads the same `.env` configuration that `voicebot.py` uses, lets you type a
phrase (or pass it via CLI), synthesizes with Piper, and plays the audio through
the configured ALSA device.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PIPER_BINARY = os.getenv("PIPER_BINARY", "piper").strip() or "piper"
PIPER_VOICE = os.getenv("PIPER_VOICE", "").strip()
PIPER_LENGTH_SCALE = os.getenv("PIPER_LENGTH_SCALE", "0.9")
PIPER_NOISE_SCALE = os.getenv("PIPER_NOISE_SCALE", "0.667")
PIPER_NOISE_W = os.getenv("PIPER_NOISE_W", "0.8")
VOICEBOT_PLAY_DEVICE = os.getenv("VOICEBOT_PLAY_DEVICE", "").strip()


def _resolve_exec(binary: str) -> str:
    if os.path.isabs(binary):
        if Path(binary).exists():
            return binary
        raise FileNotFoundError(binary)
    path = shutil.which(binary)
    if not path:
        raise FileNotFoundError(binary)
    return path


def synthesize(text: str, output: str | None = None) -> str:
    if not PIPER_VOICE:
        raise RuntimeError("PIPER_VOICE is not set; update your .env")

    exec_path = _resolve_exec(PIPER_BINARY)
    cleanup = False
    if output:
        out_path = output
    else:
        fd, out_path = tempfile.mkstemp(prefix="tts_test_", suffix=".wav")
        os.close(fd)
        cleanup = True

    cmd = [
        exec_path,
        "--model",
        PIPER_VOICE,
        "--length_scale",
        str(PIPER_LENGTH_SCALE),
        "--noise_scale",
        str(PIPER_NOISE_SCALE),
        "--noise_w",
        str(PIPER_NOISE_W),
        "--output_file",
        out_path,
    ]

    proc = subprocess.run(cmd, input=text, text=True, capture_output=True)
    if proc.returncode != 0:
        if cleanup:
            try:
                os.unlink(out_path)
            except OSError:
                pass
        raise RuntimeError(proc.stderr.strip() or f"piper exit {proc.returncode}")

    return out_path


def play(path: str, device: str | None = None) -> None:
    cmd = ["aplay", path]
    if device and device.lower() != "auto":
        cmd[1:1] = ["-D", device]
    subprocess.run(cmd, check=False)


def interactive_loop(device: str | None, keep: bool) -> None:
    print("Type text to speak; blank line quits.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            return
        path = synthesize(text)
        try:
            play(path, device)
        finally:
            if not keep:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            else:
                print(f"Saved audio to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Piper TTS with project settings")
    parser.add_argument("text", nargs="?", help="Text to synthesize once; omit for interactive mode")
    parser.add_argument("--save", metavar="PATH", help="Write audio to PATH instead of a temp file")
    parser.add_argument("--device", help="ALSA playback device (defaults to VOICEBOT_PLAY_DEVICE)")
    parser.add_argument("--keep", action="store_true", help="Keep temp wav files in interactive mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or VOICEBOT_PLAY_DEVICE or None

    if args.text:
        path = synthesize(args.text, output=args.save)
        if not args.save:
            try:
                play(path, device)
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        else:
            play(path, device)
            print(f"Saved audio to {args.save}")
    else:
        interactive_loop(device, keep=args.keep)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"[tts_test] error: {exc}", file=sys.stderr)
        sys.exit(1)
