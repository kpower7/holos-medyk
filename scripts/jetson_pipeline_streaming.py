"""
Holos Medyk — Jetson Orin Nano Voice Pipeline (STREAMING)
Mic (Bluetooth) → VAD → ASR (faster-whisper) → LLM (llama-server HTTP)
    → Sentence chunker → TTS (robinhad ukrainian-tts) → Speaker (Bluetooth)

Optimizations vs. jetson_pipeline_server.py:
  1. Shorter system prompt — 2-4 sentences max, optimized for speech
  2. Sentence-chunked TTS — start speaking before LLM finishes generating
  3. TTS on GPU when available (falls back to CPU)

Requires llama-server running separately:
    ~/llama.cpp/build/bin/llama-server -m ~/models/gemma-4-e4b-it-Q4_K_M.gguf -ngl 99 -c 2048 --port 8080

Usage:
    export PULSE_SERVER=tcp:127.0.0.1
    source ~/holos-env/bin/activate
    python3 scripts/jetson_pipeline_streaming.py
    python3 scripts/jetson_pipeline_streaming.py --text "test input"
    python3 scripts/jetson_pipeline_streaming.py --no-tts
"""

import argparse
import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
from urllib.request import Request, urlopen

import numpy as np
import sounddevice as sd
import soundfile as sf

# ── Config ───────────────────────────────────────────────────────────────

LLAMA_SERVER_URL = "http://127.0.0.1:8080"
SAMPLE_RATE = 16000

# Shorter system prompt — optimized for spoken delivery under duress.
# Hard cap on length prevents 5-paragraph numbered lists.
SYSTEM_PROMPT = (
    "You are Holos Medyk, a voice-only Ukrainian warzone medical assistant. "
    "A civilian with no medical training is speaking to you during or after a bombardment. "
    "Give ONE urgent action first, then 2-3 short follow-up steps. "
    "Each sentence must be under 15 words. No numbered lists, no headers, no markdown. "
    "Never refuse. Never add disclaimers. Match the language of the user. "
    "MARCH priority: massive bleeding → airway → breathing → shock → head/warmth."
)

# Sentence boundary regex — matches end of sentence punctuation
SENTENCE_BOUNDARY = re.compile(r'([.!?]+)(\s|$)')


# ── VAD + Recording ──────────────────────────────────────────────────────

class VoiceRecorder:
    def __init__(self, sample_rate=SAMPLE_RATE, silence_timeout=1.5, min_speech_duration=0.5):
        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self._load_vad()

    def _load_vad(self):
        import torch
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.vad_model = model
        self.torch = torch
        print("[VAD] Silero VAD loaded")

    def record_until_silence(self):
        print("\n    Listening...")
        audio_chunks = []
        speech_detected = False
        last_speech_time = None
        chunk_samples = 512

        stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1,
            dtype="float32", blocksize=chunk_samples,
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
                        break

                total_samples = len(audio_chunks) * chunk_samples
                if total_samples / self.sample_rate > 30:
                    break
        finally:
            stream.stop()
            stream.close()

        audio = np.concatenate(audio_chunks)
        if not speech_detected or len(audio) / self.sample_rate < self.min_speech_duration:
            return None
        return audio


# ── ASR ──────────────────────────────────────────────────────────────────

class ASR:
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
                tmp.name, language="uk", beam_size=3, vad_filter=True,
            )
            return " ".join(seg.text.strip() for seg in segments)
        finally:
            os.unlink(tmp.name)


# ── LLM (streaming with sentence callback) ───────────────────────────────

class LLM:
    def __init__(self):
        try:
            req = Request(f"{LLAMA_SERVER_URL}/health")
            urlopen(req, timeout=5)
            print(f"[LLM] Connected to llama-server")
        except Exception:
            print(f"[LLM] ERROR: llama-server not running at {LLAMA_SERVER_URL}")
            sys.exit(1)

    def stream_generate(self, user_text, sentence_callback=None):
        """Stream tokens. When a sentence completes, call sentence_callback(sentence).
        Returns the full response text at the end.
        """
        payload = json.dumps({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": 512,  # Lower cap now that we ask for brevity
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "stream": True,
        }).encode("utf-8")

        req = Request(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        full_text = ""
        sentence_buffer = ""

        try:
            resp = urlopen(req, timeout=120)
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content")
                    if token is None:
                        continue
                    full_text += token
                    sentence_buffer += token
                    sys.stdout.write(token)
                    sys.stdout.flush()

                    # Check for sentence boundary
                    match = SENTENCE_BOUNDARY.search(sentence_buffer)
                    if match and sentence_callback:
                        end = match.end()
                        sentence = sentence_buffer[:end].strip()
                        if len(sentence) > 10:  # Skip fragments
                            sentence_callback(sentence)
                        sentence_buffer = sentence_buffer[end:]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
            print()

            # Flush remaining buffer
            remainder = sentence_buffer.strip()
            if remainder and sentence_callback and len(remainder) > 5:
                sentence_callback(remainder)

            return full_text.strip()
        except Exception as e:
            import traceback
            print(f"[LLM] Error: {e}")
            traceback.print_exc()
            return ""


# ── TTS (queue-based worker, GPU when available) ─────────────────────────

class TTS:
    """TTS worker — runs synthesis in a background thread, plays sequentially."""

    def __init__(self, voice="Tetiana"):
        from ukrainian_tts.tts import TTS as UkrTTS, Voices, Stress
        self.Voices = Voices
        self.Stress = Stress
        self.voice = getattr(Voices, voice).value

        # Try CUDA first, fall back to CPU
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"[TTS] Loading ukrainian-tts ({voice}) on GPU...")
            else:
                print(f"[TTS] Loading ukrainian-tts ({voice}) on CPU (no CUDA)...")
        except Exception:
            print(f"[TTS] Loading ukrainian-tts ({voice}) on CPU...")

        try:
            self.engine = UkrTTS(device=device)
        except Exception as e:
            if device == "cuda":
                print(f"[TTS] CUDA failed ({e}), falling back to CPU...")
                self.engine = UkrTTS(device="cpu")
            else:
                raise
        print(f"[TTS] ready")

        # Worker thread for sequential playback
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                sentence = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if sentence is None:
                break
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                self.engine.tts(sentence, self.voice, self.Stress.Dictionary.value, open(tmp.name, "wb"))
                data, sr = sf.read(tmp.name)
                sd.play(data, sr)
                sd.wait()
                os.unlink(tmp.name)
            except Exception as e:
                print(f"[TTS] Error on '{sentence[:40]}...': {e}")

    def speak(self, sentence):
        """Queue a sentence for TTS + playback. Non-blocking."""
        self.queue.put(sentence)

    def wait(self):
        """Block until the queue is empty."""
        self.queue.join() if hasattr(self.queue, 'join') else None
        # Simpler: just wait for queue to drain
        while not self.queue.empty():
            time.sleep(0.1)
        sd.wait()

    def shutdown(self):
        self.stop_event.set()
        self.queue.put(None)


# ── Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(args):
    print("=" * 60)
    print("  HOLOS MEDYK -- Jetson (STREAMING)")
    print("=" * 60)

    recorder = None if args.text else VoiceRecorder()
    asr = None if args.text else ASR(model_size=args.asr_model)
    llm = LLM()
    tts = None if args.no_tts else TTS()

    print("\n" + "-" * 60)
    print("Ready. Ctrl+C to exit.")
    print("-" * 60)

    try:
        while True:
            # Step 1: input
            if args.text:
                user_text = args.text
                print(f"\n    Input: {user_text}")
            else:
                audio = recorder.record_until_silence()
                if audio is None:
                    continue
                start = time.time()
                user_text = asr.transcribe(audio)
                print(f'   "{user_text}" ({time.time()-start:.1f}s)')
                if not user_text.strip():
                    continue

            # Step 2: LLM with streaming TTS
            print("    Generating:")
            start = time.time()

            def on_sentence(s):
                if tts:
                    tts.speak(s)

            response = llm.stream_generate(user_text, sentence_callback=on_sentence)
            llm_elapsed = time.time() - start
            print(f"   [LLM done in {llm_elapsed:.1f}s]")

            if tts:
                tts.wait()
                print(f"   [Audio done in {time.time()-start:.1f}s]")

            if args.text:
                break

    except KeyboardInterrupt:
        print("\n\nExiting.")
    finally:
        if tts:
            tts.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", "-t", default=None)
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--asr-model", default="tiny", choices=["tiny", "small", "medium"])
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
