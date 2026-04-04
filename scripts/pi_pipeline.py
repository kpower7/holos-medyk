"""
Holos Medyk — Raspberry Pi Voice Pipeline
Mic → Gemma 4 E4B (native audio input via LiteRT-LM) → Piper TTS → Speaker

Usage:
    python3 pi_pipeline.py
    python3 pi_pipeline.py --prompt "text input"
    python3 pi_pipeline.py --duration 10
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import litert_lm

MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--litert-community--gemma-4-E4B-it-litert-lm/"
    "snapshots/afca9a55ba2848faee6588e46b47c3164411a903/gemma-4-E4B-it.litertlm"
)
VOICE_PATH = os.path.expanduser(
    "~/voices/uk/uk_UA/ukrainian_tts/medium/uk_UA-ukrainian_tts-medium.onnx"
)

SYSTEM_PROMPT = (
    "You are Holos Medyk, an emergency medical assistant for Ukrainian civilians. "
    "Give clear step-by-step first aid instructions in Ukrainian. "
    "Be concise and direct. Respond only in Ukrainian."
)

SAMPLE_RATE = 16000


def record_audio(duration=7.0):
    """Record from mic."""
    print("\n🎤 Listening... (speak now)")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print(f"   [Recorded {duration}s]")
    return audio.flatten()


def save_audio(audio, path):
    """Save audio to WAV file."""
    sf.write(path, audio, SAMPLE_RATE)


def speak(text):
    """Synthesize and play speech via Piper TTS."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    try:
        result = subprocess.run(
            ["piper", "--model", VOICE_PATH, "--output_file", tmp.name],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )

        if result.returncode == 0 and os.path.exists(tmp.name):
            data, sr = sf.read(tmp.name)
            duration = len(data) / sr
            print(f"🔊 Speaking ({duration:.1f}s)...")
            sd.play(data, sr)
            sd.wait()
    finally:
        os.unlink(tmp.name)


def main():
    parser = argparse.ArgumentParser(description="Holos Medyk Pi Pipeline")
    parser.add_argument("--prompt", "-p", default=None, help="Text input (skip mic)")
    parser.add_argument("--duration", "-d", type=float, default=7.0, help="Recording duration")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS output")
    args = parser.parse_args()

    print("=" * 50)
    print("  ГОЛОС МЕДИК — Holos Medyk")
    print("  Raspberry Pi Voice Assistant")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("[Loading model...]")
    engine = litert_lm.Engine(MODEL_PATH, audio_backend=litert_lm.Backend.CPU)
    conversation = engine.create_conversation()
    print("[Model ready]")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            if args.prompt:
                # Text-only mode
                print(f"📝 Input: {args.prompt}")
                print("🧠 Generating response...")
                start = time.time()
                response = conversation.send_message({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{SYSTEM_PROMPT}\n\nUser: {args.prompt}"},
                    ],
                })
                elapsed = time.time() - start
                text = response["content"][0]["text"]
                print(f"   [{elapsed:.1f}s]")
                print(f"\n💬 Response:\n{text}\n")

                if not args.no_tts:
                    speak(text)
                break

            else:
                # Voice mode — record and send audio directly to Gemma 4
                audio = record_audio(args.duration)

                # Save to temp WAV
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                save_audio(audio, tmp.name)
                tmp.close()

                print("🧠 Processing audio + generating response...")
                start = time.time()
                try:
                    response = conversation.send_message({
                        "role": "user",
                        "content": [
                            {"type": "audio", "path": tmp.name},
                            {"type": "text", "text": SYSTEM_PROMPT},
                        ],
                    })
                    elapsed = time.time() - start
                    text = response["content"][0]["text"]
                    print(f"   [{elapsed:.1f}s]")
                    print(f"\n💬 Response:\n{text}\n")

                    if not args.no_tts:
                        speak(text)
                finally:
                    os.unlink(tmp.name)

    except KeyboardInterrupt:
        print("\n\nExiting. Stay safe. 🇺🇦")
    finally:
        conversation.close()
        engine.close()


if __name__ == "__main__":
    main()
