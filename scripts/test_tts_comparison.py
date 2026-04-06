"""
Holos Medyk — Ukrainian TTS Comparison Test
Generates audio from multiple TTS engines for native speaker evaluation.

Engines tested:
  1. Piper TTS (uk_UA-ukrainian_tts-medium) — baseline (known broken)
  2. Silero v5 cis_base nostress (ukr_igor, ukr_roman) — MIT, ONNX
  3. Silero v4_ua (mykyta) — small, ONNX
  4. robinhad/ukrainian-tts (all 5 voices) — ESPnet/VITS
  5. eSpeak-ng — robotic fallback

Usage:
    pip install torch torchaudio soundfile sounddevice num2words
    pip install ukrainian-tts  # for robinhad
    python scripts/test_tts_comparison.py
    python scripts/test_tts_comparison.py --play
    python scripts/test_tts_comparison.py --engines silero_v5 espeak
"""

import argparse
import os
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "test_output" / "tts_comparison"


def _save_wav(audio_tensor, path: Path, sample_rate: int):
    """Save a 1D torch tensor to WAV using scipy (avoids torchaudio codec issues)."""
    import numpy as np
    from scipy.io import wavfile
    audio_np = audio_tensor.detach().cpu().numpy()
    if audio_np.dtype != np.int16:
        audio_np = (audio_np * 32767).astype(np.int16)
    wavfile.write(str(path), sample_rate, audio_np)

# Medical emergency phrases for testing — same ones Diana heard before, plus new ones
TEST_PHRASES = [
    "Спокійно. Я допоможу вам крок за кроком.",
    "Притисніть чисту тканину до рани і тисніть сильно.",
    "Не рухайте постраждалого. Чекайте на допомогу.",
    "Накладіть джгут вище рани. Затягніть міцно.",
    "Перевірте, чи людина дихає. Нахиліть голову назад.",
    "Зателефонуйте в швидку допомогу. Номер сто три.",
    # Longer, more conversational — tests natural flow
    "Зараз найважливіше зупинити кровотечу. Візьміть будь-яку чисту тканину, притисніть до рани обома руками і не відпускайте.",
    "Якщо ви бачите, що кров б'є фонтаном, накладіть джгут якомога вище на кінцівку і закрутіть палицею до повної зупинки кровотечі.",
]


def play_audio(path: Path):
    try:
        import sounddevice as sd
        import soundfile as sf
        data, sr = sf.read(str(path))
        duration = len(data) / sr
        print(f"      Playing ({duration:.1f}s)...")
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"      Could not play: {e}")


# ── Engine 1: Piper TTS ──────────────────────────────────────────────────────

def test_piper(output_dir: Path, do_play: bool):
    """Piper TTS — known broken for Ukrainian, included as baseline."""
    piper_exe = Path(r"C:\Users\k_pow\miniconda3\envs\holos\Scripts\piper.exe")
    voices = {
        "ukrainian_tts_medium": ROOT / "voices" / "uk" / "uk_UA" / "ukrainian_tts" / "medium" / "uk_UA-ukrainian_tts-medium.onnx",
        "lada_x_low": ROOT / "voices" / "uk" / "uk_UA" / "lada" / "x_low" / "uk_UA-lada-x_low.onnx",
    }

    if not piper_exe.exists():
        print("  [SKIP] piper.exe not found")
        return

    for voice_name, voice_path in voices.items():
        if not voice_path.exists():
            print(f"  [SKIP] Piper voice {voice_name} not found at {voice_path}")
            continue

        eng_dir = output_dir / f"piper_{voice_name}"
        eng_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n    Voice: {voice_name}")

        for i, phrase in enumerate(TEST_PHRASES):
            out = eng_dir / f"phrase_{i+1}.wav"
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            result = subprocess.run(
                [str(piper_exe), "--model", str(voice_path), "--output_file", str(out)],
                input=phrase.encode("utf-8"),
                capture_output=True, env=env, timeout=30,
            )
            status = "OK" if result.returncode == 0 and out.exists() else "FAIL"
            print(f"      [{i+1}] {status} -- {phrase[:50]}...")
            if do_play and out.exists():
                play_audio(out)


# ── Engine 2: Silero TTS ─────────────────────────────────────────────────────

def test_silero_v5(output_dir: Path, do_play: bool):
    """Silero v5 cis_base nostress — MIT license, auto-stress for Ukrainian."""
    try:
        import torch
    except ImportError:
        print("  [SKIP] torch not installed")
        return

    eng_dir = output_dir / "silero_v5_nostress"
    eng_dir.mkdir(parents=True, exist_ok=True)

    print("    Loading Silero v5 cis_base_nostress model...")
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v5_cis_base_nostress",
        trust_repo=True,
    )

    voices = ["ukr_igor", "ukr_roman"]
    for voice in voices:
        voice_dir = eng_dir / voice
        voice_dir.mkdir(exist_ok=True)
        print(f"\n    Voice: {voice}")
        for i, phrase in enumerate(TEST_PHRASES):
            out = voice_dir / f"phrase_{i+1}.wav"
            try:
                audio = model.apply_tts(
                    text=phrase, speaker=voice, sample_rate=48000
                )
                _save_wav(audio, out, 48000)
                print(f"      [{i+1}] OK -- {phrase[:50]}...")
                if do_play:
                    play_audio(out)
            except Exception as e:
                print(f"      [{i+1}] FAIL -- {e}")


def test_silero_v4(output_dir: Path, do_play: bool):
    """Silero v4_ua — smallest model (34MB), mykyta voice."""
    try:
        import torch
    except ImportError:
        print("  [SKIP] torch not installed")
        return

    eng_dir = output_dir / "silero_v4_ua"
    eng_dir.mkdir(parents=True, exist_ok=True)

    print("    Loading Silero v4_ua model...")
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ua",
        speaker="v4_ua",
        trust_repo=True,
    )

    for i, phrase in enumerate(TEST_PHRASES):
        out = eng_dir / f"phrase_{i+1}.wav"
        try:
            audio = model.apply_tts(
                text=phrase, speaker="mykyta", sample_rate=48000
            )
            _save_wav(audio, out, 48000)
            print(f"    [{i+1}] OK -- {phrase[:50]}...")
            if do_play:
                play_audio(out)
        except Exception as e:
            print(f"    [{i+1}] FAIL -- {e}")


# ── Engine 3: robinhad/ukrainian-tts ──────────────────────────────────────────

def test_robinhad(output_dir: Path, do_play: bool):
    """robinhad/ukrainian-tts — ESPnet VITS, 5 voices, auto-stress."""
    try:
        from ukrainian_tts.tts import TTS as UkrTTS
    except ImportError:
        print("  [SKIP] ukrainian-tts not installed (pip install ukrainian-tts)")
        return

    eng_dir = output_dir / "robinhad_ukrainian_tts"
    eng_dir.mkdir(parents=True, exist_ok=True)

    print("    Loading robinhad/ukrainian-tts engine...")
    try:
        tts = UkrTTS(device="cpu")
    except Exception as e:
        print(f"  [ERROR] Could not init TTS: {e}")
        return

    voices = ["oleksa", "tetiana", "dmytro", "lada", "mykyta"]

    for voice_name in voices:
        voice_dir = eng_dir / voice_name
        voice_dir.mkdir(exist_ok=True)
        print(f"\n    Voice: {voice_name}")

        for i, phrase in enumerate(TEST_PHRASES):
            out = voice_dir / f"phrase_{i+1}.wav"
            try:
                with open(str(out), "wb") as f:
                    tts.tts(phrase, voice=voice_name, stress="dictionary", output_fp=f)
                print(f"      [{i+1}] OK -- {phrase[:50]}...")
                if do_play:
                    play_audio(out)
            except Exception as e:
                print(f"      [{i+1}] FAIL -- {e}")


# ── Engine 4: eSpeak-ng ──────────────────────────────────────────────────────

def test_espeak(output_dir: Path, do_play: bool):
    """eSpeak-ng — robotic but guaranteed intelligible. Tiny footprint."""
    eng_dir = output_dir / "espeak_ng"
    eng_dir.mkdir(parents=True, exist_ok=True)

    # Check if espeak-ng is available
    try:
        subprocess.run(["espeak-ng", "--version"], capture_output=True, timeout=5)
    except FileNotFoundError:
        # Try espeak as fallback name
        try:
            subprocess.run(["espeak", "--version"], capture_output=True, timeout=5)
            espeak_cmd = "espeak"
        except FileNotFoundError:
            print("  [SKIP] espeak-ng not installed")
            print("         Install: https://github.com/espeak-ng/espeak-ng/releases")
            return
    else:
        espeak_cmd = "espeak-ng"

    for i, phrase in enumerate(TEST_PHRASES):
        out = eng_dir / f"phrase_{i+1}.wav"
        result = subprocess.run(
            [espeak_cmd, "-v", "uk", "-w", str(out), phrase],
            capture_output=True, timeout=30,
        )
        status = "OK" if result.returncode == 0 and out.exists() else "FAIL"
        print(f"    [{i+1}] {status} — {phrase[:50]}...")
        if do_play and out.exists():
            play_audio(out)


# ── Main ──────────────────────────────────────────────────────────────────────

ENGINES = {
    "piper": ("Piper TTS (baseline, known broken)", test_piper),
    "silero_v5": ("Silero v5 cis_base nostress (MIT, ONNX-ready)", test_silero_v5),
    "silero_v4": ("Silero v4_ua (34MB, smallest)", test_silero_v4),
    "robinhad": ("robinhad/ukrainian-tts (ESPnet, 5 voices)", test_robinhad),
    "espeak": ("eSpeak-ng (robotic fallback)", test_espeak),
}


def main():
    parser = argparse.ArgumentParser(description="Compare Ukrainian TTS engines")
    parser.add_argument("--play", action="store_true", help="Play audio after generating")
    parser.add_argument(
        "--engines", nargs="+", default=list(ENGINES.keys()),
        choices=list(ENGINES.keys()),
        help="Which engines to test (default: all)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Holos Medyk -- Ukrainian TTS Comparison")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Phrases: {len(TEST_PHRASES)}")
    print(f"Engines: {', '.join(args.engines)}")
    print("=" * 70)

    for engine_key in args.engines:
        desc, func = ENGINES[engine_key]
        print(f"\n{'-' * 70}")
        print(f"Engine: {desc}")
        print(f"{'-' * 70}")
        try:
            func(OUTPUT_DIR, args.play)
        except Exception as e:
            print(f"  [ERROR] {e}")

    # Print summary for Diana
    print(f"\n{'=' * 70}")
    print("DONE — Files for Diana to review:")
    print(f"{'=' * 70}")
    engine_dirs = {
        "piper": ["piper_ukrainian_tts_medium", "piper_lada_x_low"],
        "silero_v5": ["silero_v5_nostress"],
        "silero_v4": ["silero_v4_ua"],
        "robinhad": ["robinhad_ukrainian_tts"],
        "espeak": ["espeak_ng"],
    }
    for engine_key in args.engines:
        for dirname in engine_dirs[engine_key]:
            eng_path = OUTPUT_DIR / dirname
            if eng_path.exists():
                wav_count = len(list(eng_path.rglob("*.wav")))
                print(f"  {eng_path.relative_to(ROOT)} -- {wav_count} files")

    print(f"\nAsk Diana to listen to each folder and rank:")
    print(f"  1. Can you understand what is being said?")
    print(f"  2. Does the pronunciation sound natural?")
    print(f"  3. Would you trust this voice in an emergency?")
    print(f"  4. Rank all voices from best to worst.")


if __name__ == "__main__":
    main()
