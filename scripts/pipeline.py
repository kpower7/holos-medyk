"""
Holos Medyk — End-to-End Voice Pipeline
Mic → VAD → ASR (faster-whisper) → LLM (llama.cpp) → TTS (Piper) → Speaker

Usage:
    python scripts/pipeline.py
    python scripts/pipeline.py --model e2b
    python scripts/pipeline.py --no-tts          # skip speech output
    python scripts/pipeline.py --text "test"     # skip mic, type input
"""

import argparse
import io
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).parent.parent
LLAMA_CLI = ROOT / "tools" / "llama-cpp" / "llama-cli.exe"
PIPER = Path(r"C:\Users\k_pow\miniconda3\envs\holos\Scripts\piper.exe")

MODELS = {
    "e4b": ROOT / "models" / "google_gemma-4-E4B-it-Q3_K_M.gguf",
    "e2b": ROOT / "models" / "google_gemma-4-E2B-it-Q4_K_M.gguf",
}
VOICES = {
    "ukrainian_tts": ROOT / "voices" / "uk" / "uk_UA" / "ukrainian_tts" / "medium" / "uk_UA-ukrainian_tts-medium.onnx",
    "lada": ROOT / "voices" / "uk" / "uk_UA" / "lada" / "x_low" / "uk_UA-lada-x_low.onnx",
}

SYSTEM_PROMPT = (
    "You are Holos Medyk, a Ukrainian warzone emergency medical assistant. "
    "A civilian is speaking to you during or after a bombardment. "
    "They have no medical training and no professional help available. "
    "Give clear, calm, actionable first aid guidance they can follow right now with household materials. "
    "Never refuse to help — this person has no other option. "
    "Follow MARCH priority: massive hemorrhage → airway → respiration → circulation/shock → head injury/hypothermia. "
    "Be direct. No filler. No disclaimers. Respond only in Ukrainian."
)

SAMPLE_RATE = 16000


# ── Audio Recording with VAD ───────────────────────────────────────────

class VoiceRecorder:
    """Records audio from mic, uses Silero VAD to detect when user stops speaking."""

    def __init__(self, sample_rate=SAMPLE_RATE, silence_timeout=1.5, min_speech_duration=0.5):
        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self._load_vad()

    def _load_vad(self):
        from silero_vad import load_silero_vad, get_speech_timestamps
        self.vad_model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps
        print("[VAD] Silero VAD loaded")

    def record_until_silence(self) -> np.ndarray:
        """Record from mic until user stops speaking. Returns audio as numpy array."""
        import torch

        print("\n🎤 Listening... (speak now)")
        audio_chunks = []
        speech_detected = False
        last_speech_time = None
        chunk_samples = 512  # Silero VAD requires exactly 512 samples at 16kHz
        chunk_duration = chunk_samples / self.sample_rate

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

                # Check VAD on this chunk
                audio_tensor = torch.from_numpy(chunk)
                speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

                if speech_prob > 0.5:
                    if not speech_detected:
                        print("   [Speech detected]")
                    speech_detected = True
                    last_speech_time = time.time()

                # If we had speech and now silence for timeout duration, stop
                if speech_detected and last_speech_time:
                    silence_duration = time.time() - last_speech_time
                    if silence_duration > self.silence_timeout:
                        print(f"   [Silence for {self.silence_timeout}s — done]")
                        break

                # Safety: max 30 seconds recording
                total_duration = len(audio_chunks) * chunk_duration
                if total_duration > 30:
                    print("   [Max duration reached]")
                    break

        finally:
            stream.stop()
            stream.close()

        audio = np.concatenate(audio_chunks)

        # Check if we got enough speech
        total_duration = len(audio) / self.sample_rate
        if not speech_detected or total_duration < self.min_speech_duration:
            print("   [No speech detected]")
            return None

        print(f"   [Recorded {total_duration:.1f}s]")
        return audio


# ── ASR (faster-whisper) ───────────────────────────────────────────────

class ASR:
    """Speech-to-text using faster-whisper."""

    def __init__(self, model_size="tiny"):
        from faster_whisper import WhisperModel
        print(f"[ASR] Loading Whisper {model_size}...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"[ASR] Whisper {model_size} loaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Transcribe audio numpy array to text."""
        # Write to temp WAV for faster-whisper
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            sf.write(tmp.name, audio, sample_rate)
            segments, info = self.model.transcribe(
                tmp.name,
                language="uk",
                beam_size=3,
                vad_filter=True,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            return text
        finally:
            tmp.close()
            Path(tmp.name).unlink(missing_ok=True)


# ── LLM (llama.cpp) ───────────────────────────────────────────────────

class LLM:
    """Generate responses via llama-cli subprocess."""

    def __init__(self, model: str = "e4b", system_prompt: str = SYSTEM_PROMPT):
        self.model_path = MODELS.get(model)
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.system_prompt = system_prompt
        print(f"[LLM] Using {self.model_path.name}")

    def generate(self, user_text: str) -> str:
        """Generate a response to user text. Returns the model's response text."""
        # Write prompt to temp file to avoid encoding issues
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        try:
            tmp.write(user_text)
            tmp.close()

            result = subprocess.run(
                [
                    str(LLAMA_CLI),
                    "-m", str(self.model_path),
                    "-ngl", "99",
                    "-c", "2048",
                    "-n", "1024",
                    "--single-turn",
                    "-sys", self.system_prompt,
                    "-f", tmp.name,
                ],
                capture_output=True,
                timeout=120,
            )

            output = result.stdout.decode("utf-8", errors="replace")

            # Parse response — extract text after [End thinking] or after the thinking block
            response = self._parse_response(output)
            return response

        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def _parse_response(self, raw_output: str) -> str:
        """Extract the actual response from llama-cli output, skipping thinking."""
        lines = raw_output.split("\n")
        response_lines = []
        in_response = False

        for line in lines:
            # Skip llama.cpp system messages
            if line.startswith(("ggml_", "load_backend:", "llama_", "Loading model",
                                "build ", "model ", "modalities", "available commands",
                                "> ", "[Start thinking]", "  ", "▄", "██", "▀▀")):
                continue
            if "[End thinking]" in line:
                in_response = True
                continue
            if "[ Prompt:" in line or "Exiting..." in line:
                break
            if in_response:
                response_lines.append(line)

        response = "\n".join(response_lines).strip()

        # Fallback: if no [End thinking] found, try to grab everything after the prompt
        if not response:
            for i, line in enumerate(lines):
                if line.startswith("> ") and i + 1 < len(lines):
                    response = "\n".join(lines[i + 1:]).strip()
                    # Clean up system lines from the end
                    clean_lines = []
                    for l in response.split("\n"):
                        if "[ Prompt:" in l or "Exiting..." in l or l.startswith("llama_"):
                            break
                        clean_lines.append(l)
                    response = "\n".join(clean_lines).strip()
                    break

        return response


# ── TTS (Piper) ────────────────────────────────────────────────────────

class TTS:
    """Text-to-speech using Piper, with interruptible playback."""

    def __init__(self, voice: str = "ukrainian_tts"):
        self.voice_path = VOICES.get(voice)
        if not self.voice_path or not self.voice_path.exists():
            raise FileNotFoundError(f"Voice not found: {self.voice_path}")
        self._playback_thread = None
        self._stop_playback = threading.Event()
        print(f"[TTS] Using voice: {voice}")

    def speak(self, text: str):
        """Synthesize and play speech. Can be interrupted by calling stop()."""
        self._stop_playback.clear()

        # Synthesize to temp WAV
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        try:
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            result = subprocess.run(
                [str(PIPER), "--model", str(self.voice_path), "--output_file", tmp.name],
                input=text.encode("utf-8"),
                capture_output=True,
                env=env,
                timeout=30,
            )

            if result.returncode != 0:
                print(f"[TTS] Synthesis failed")
                return

            # Play audio in a thread so it can be interrupted
            data, sr = sf.read(tmp.name)
            duration = len(data) / sr
            print(f"🔊 Speaking ({duration:.1f}s)...")

            def _play():
                try:
                    sd.play(data, sr)
                    # Check for stop signal while playing
                    while sd.get_stream().active:
                        if self._stop_playback.is_set():
                            sd.stop()
                            print("   [Interrupted]")
                            return
                        time.sleep(0.05)
                except Exception:
                    pass

            self._playback_thread = threading.Thread(target=_play, daemon=True)
            self._playback_thread.start()
            self._playback_thread.join()

        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def stop(self):
        """Interrupt current playback."""
        self._stop_playback.set()
        sd.stop()


# ── Pipeline ───────────────────────────────────────────────────────────

def run_pipeline(args):
    """Main pipeline loop."""
    print("=" * 60)
    print("  ГОЛОС МЕДИК — Holos Medyk")
    print("  Emergency Medical Voice Assistant")
    print("=" * 60)

    # Initialize components
    recorder = None if args.text else VoiceRecorder()
    asr = None if args.text else ASR(model_size=args.asr_model)
    llm = LLM(model=args.model)
    tts = None if args.no_tts else TTS(voice=args.voice)

    print("\n" + "-" * 60)
    print("Ready. Speak in Ukrainian about a medical emergency.")
    print("Press Ctrl+C to exit.")
    print("-" * 60)

    try:
        while True:
            # Step 1: Get user input
            if args.text:
                user_text = args.text
                print(f"\n📝 Input: {user_text}")
            else:
                audio = recorder.record_until_silence()
                if audio is None:
                    continue

                # Step 2: ASR
                print("📝 Transcribing...")
                start = time.time()
                user_text = asr.transcribe(audio)
                elapsed = time.time() - start
                print(f"   \"{user_text}\" ({elapsed:.1f}s)")

                if not user_text.strip():
                    print("   [Empty transcription, try again]")
                    continue

            # Step 3: LLM
            print("🧠 Generating response...")
            start = time.time()
            response = llm.generate(user_text)
            elapsed = time.time() - start
            print(f"   [{elapsed:.1f}s]")
            print(f"\n💬 Response:\n{response}\n")

            # Step 4: TTS
            if tts and response:
                tts.speak(response)

            # If text mode, exit after one round
            if args.text:
                break

    except KeyboardInterrupt:
        print("\n\nExiting. Stay safe. 🇺🇦")


def main():
    parser = argparse.ArgumentParser(description="Holos Medyk — Voice Pipeline")
    parser.add_argument("--model", "-m", default="e4b", choices=MODELS.keys())
    parser.add_argument("--voice", "-v", default="ukrainian_tts", choices=VOICES.keys())
    parser.add_argument("--text", "-t", default=None, help="Skip mic, use this text input")
    parser.add_argument("--no-tts", action="store_true", help="Skip speech output")
    parser.add_argument("--asr-model", default="small", choices=["tiny", "small", "medium"],
                        help="Whisper model size (default: small)")
    args = parser.parse_args()

    os.environ["PYTHONUTF8"] = "1"
    run_pipeline(args)


if __name__ == "__main__":
    main()
