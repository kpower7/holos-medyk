"""
Holos Medyk — Jetson Orin Nano Voice Pipeline
Mic (Bluetooth) → VAD → ASR (faster-whisper) → LLM (llama.cpp CUDA) → TTS (Silero v4 Ukrainian) → Speaker (Bluetooth)

Usage:
    export PULSE_SERVER=tcp:127.0.0.1
    source ~/holos-env/bin/activate
    python3 scripts/jetson_pipeline.py
    python3 scripts/jetson_pipeline.py --text "test input"
    python3 scripts/jetson_pipeline.py --no-tts
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

# ── Paths ────────────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
LLAMA_CLI = os.path.join(HOME, "llama.cpp/build/bin/llama-cli")
MODEL_PATH = os.path.join(HOME, "models/gemma-4-e4b-it-Q4_K_M.gguf")

SYSTEM_PROMPT = (
    "You are Holos Medyk, a Ukrainian warzone emergency medical assistant. "
    "A civilian is speaking to you during or after a bombardment. "
    "They have no medical training and no professional help available. "
    "Give clear, calm, actionable first aid guidance they can follow right now with household materials. "
    "Never refuse to help — this person has no other option. "
    "Follow MARCH priority: massive hemorrhage → airway → respiration → circulation/shock → head injury/hypothermia. "
    "Be direct. No filler. No disclaimers. Match the language of the user."
)

SAMPLE_RATE = 16000


# ── VAD + Recording ──────────────────────────────────────────────────────

class VoiceRecorder:
    """Records audio from mic, uses Silero VAD to detect when user stops speaking."""

    def __init__(self, sample_rate=SAMPLE_RATE, silence_timeout=1.5, min_speech_duration=0.5):
        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self._load_vad()

    def _load_vad(self):
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.vad_model = model
        self.torch = torch
        print("[VAD] Silero VAD loaded")

    def record_until_silence(self):
        """Record from mic until user stops speaking. Returns audio as numpy array."""
        print("\n    Listening... (speak now)")
        audio_chunks = []
        speech_detected = False
        last_speech_time = None
        chunk_samples = 512  # Silero VAD requires exactly 512 samples at 16kHz

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
        )
        stream.start()

        try:
            while True:
                chunk, _ = stream.read(chunk_samples)
                chunk = chunk.flatten()
                audio_chunks.append(chunk)

                audio_tensor = self.torch.from_numpy(chunk)
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

                if speech_prob > 0.5:
                    if not speech_detected:
                        print("   [Speech detected]")
                    speech_detected = True
                    last_speech_time = time.time()

                if speech_detected and last_speech_time:
                    if time.time() - last_speech_time > self.silence_timeout:
                        print(f"   [Silence for {self.silence_timeout}s -- done]")
                        break

                # Max 30 seconds
                total_samples = len(audio_chunks) * chunk_samples
                if total_samples / self.sample_rate > 30:
                    print("   [Max duration reached]")
                    break

        finally:
            stream.stop()
            stream.close()

        audio = np.concatenate(audio_chunks)
        total_duration = len(audio) / self.sample_rate

        if not speech_detected or total_duration < self.min_speech_duration:
            print("   [No speech detected]")
            return None

        print(f"   [Recorded {total_duration:.1f}s]")
        return audio


# ── ASR (faster-whisper) ─────────────────────────────────────────────────

class ASR:
    """Speech-to-text using faster-whisper on CPU."""

    def __init__(self, model_size="tiny"):
        from faster_whisper import WhisperModel
        print(f"[ASR] Loading Whisper {model_size}...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"[ASR] Whisper {model_size} loaded")

    def transcribe(self, audio, sample_rate=SAMPLE_RATE):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, audio, sample_rate)
            segments, _ = self.model.transcribe(
                tmp.name,
                language="uk",
                beam_size=3,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            return text
        finally:
            os.unlink(tmp.name)


# ── LLM (llama.cpp subprocess) ───────────────────────────────────────────

class LLM:
    """Generate responses via llama-cli subprocess with CUDA."""

    def __init__(self):
        if not os.path.exists(LLAMA_CLI):
            raise FileNotFoundError(f"llama-cli not found: {LLAMA_CLI}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        print(f"[LLM] Using {os.path.basename(MODEL_PATH)}")

    def generate(self, user_text):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        try:
            tmp.write(user_text)
            tmp.close()

            result = subprocess.run(
                [
                    LLAMA_CLI,
                    "-m", MODEL_PATH,
                    "-ngl", "99",
                    "-c", "2048",
                    "-n", "512",
                    "--single-turn",
                    "-sys", SYSTEM_PROMPT,
                    "-f", tmp.name,
                ],
                capture_output=True,
                timeout=300,
            )

            output = result.stdout.decode("utf-8", errors="replace")
            return self._parse_response(output)

        finally:
            os.unlink(tmp.name)

    def _parse_response(self, raw_output):
        """Extract the response text from llama-cli --single-turn output."""
        lines = raw_output.split("\n")

        # Strategy: find [End thinking], capture everything after it until stats/exit.
        # If no thinking block, capture everything after "> " prompt echo.
        end_thinking_idx = -1
        prompt_echo_idx = -1
        for i, line in enumerate(lines):
            if "[End thinking]" in line:
                end_thinking_idx = i
            if line.startswith("> "):
                prompt_echo_idx = i

        start = end_thinking_idx + 1 if end_thinking_idx >= 0 else prompt_echo_idx + 1
        if start <= 0:
            return raw_output.strip()

        response_lines = []
        for line in lines[start:]:
            if "[ Prompt:" in line or "Exiting..." in line:
                break
            response_lines.append(line)

        return "\n".join(response_lines).strip()


# ── TTS (Silero v4 Ukrainian) ─────────────────────────────────────────────

class TTS:
    """Text-to-speech using Silero v4 Ukrainian via torch.hub."""

    def __init__(self, sample_rate=48000):
        import torch
        self.torch = torch
        self.sample_rate = sample_rate
        print("[TTS] Loading Silero v4_ua...")
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ua",
            speaker="v4_ua",
        )
        print("[TTS] Silero v4_ua loaded")

    def speak(self, text):
        """Synthesize and play speech."""
        try:
            audio = self.model.apply_tts(
                text=text,
                sample_rate=self.sample_rate,
            )
            audio_np = audio.numpy()
            duration = len(audio_np) / self.sample_rate
            print(f"    Speaking ({duration:.1f}s)...")
            sd.play(audio_np, self.sample_rate)
            sd.wait()
        except Exception as e:
            print(f"[TTS] Error: {e}")


# ── Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(args):
    print("=" * 60)
    print("  HOLOS MEDYK -- Jetson Orin Nano")
    print("  Emergency Medical Voice Assistant")
    print("=" * 60)

    recorder = None if args.text else VoiceRecorder()
    asr = None if args.text else ASR(model_size=args.asr_model)
    llm = LLM()
    tts = None if args.no_tts else TTS()

    print("\n" + "-" * 60)
    print("Ready. Speak about a medical emergency.")
    print("Press Ctrl+C to exit.")
    print("-" * 60)

    try:
        while True:
            # Step 1: Get user input
            if args.text:
                user_text = args.text
                print(f"\n    Input: {user_text}")
            else:
                audio = recorder.record_until_silence()
                if audio is None:
                    continue

                print("    Transcribing...")
                start = time.time()
                user_text = asr.transcribe(audio)
                elapsed = time.time() - start
                print(f'   "{user_text}" ({elapsed:.1f}s)')

                if not user_text.strip():
                    print("   [Empty transcription, try again]")
                    continue

            # Step 2: LLM
            print("    Generating response...")
            start = time.time()
            response = llm.generate(user_text)
            elapsed = time.time() - start
            print(f"   [{elapsed:.1f}s]")
            print(f"\n    Response:\n{response}\n")

            # Step 3: TTS
            if tts and response:
                tts.speak(response)

            # Text mode: one round only
            if args.text:
                break

    except KeyboardInterrupt:
        print("\n\nExiting. Stay safe.")


def main():
    parser = argparse.ArgumentParser(description="Holos Medyk -- Jetson Voice Pipeline")
    parser.add_argument("--text", "-t", default=None, help="Skip mic, use text input")
    parser.add_argument("--no-tts", action="store_true", help="Skip speech output")
    parser.add_argument("--asr-model", default="tiny", choices=["tiny", "small", "medium"],
                        help="Whisper model size (default: tiny)")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
