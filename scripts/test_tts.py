"""
Holos Medyk — Test Piper TTS Ukrainian Voices
Synthesizes emergency medical phrases and plays them.

Usage:
    python scripts/test_tts.py
    python scripts/test_tts.py --no-play
    python scripts/test_tts.py --voice lada
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
VOICES = {
    "ukrainian_tts": ROOT / "voices" / "uk" / "uk_UA" / "ukrainian_tts" / "medium" / "uk_UA-ukrainian_tts-medium.onnx",
    "lada": ROOT / "voices" / "uk" / "uk_UA" / "lada" / "x_low" / "uk_UA-lada-x_low.onnx",
}
OUTPUT_DIR = ROOT / "test_output"
PIPER = Path(r"C:\Users\k_pow\miniconda3\envs\holos\Scripts\piper.exe")

TEST_PHRASES = [
    "Спокійно. Я допоможу вам крок за кроком.",
    "Притисніть чисту тканину до рани і тисніть сильно.",
    "Не рухайте постраждалого. Чекайте на допомогу.",
    "Накладіть джгут вище рани. Затягніть міцно.",
    "Перевірте, чи людина дихає. Нахиліть голову назад.",
    "Зателефонуйте в швидку допомогу. Номер сто три.",
]


def synthesize(text: str, voice_path: Path, output_path: Path) -> bool:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [str(PIPER), "--model", str(voice_path), "--output_file", str(output_path)],
        input=text.encode("utf-8"),
        capture_output=True,
        env=env,
        timeout=30,
    )
    return result.returncode == 0 and output_path.exists()


def play_audio(path: Path):
    try:
        import sounddevice as sd
        import soundfile as sf
        data, sr = sf.read(str(path))
        duration = len(data) / sr
        print(f"    Playing ({duration:.1f}s)...")
        sd.play(data, sr)
        sd.wait()
    except Exception as e:
        print(f"    Could not play: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test Piper TTS Ukrainian voices")
    parser.add_argument("--no-play", action="store_true", help="Don't play audio")
    parser.add_argument("--voice", default="all", choices=list(VOICES.keys()) + ["all"])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    voices_to_test = VOICES if args.voice == "all" else {args.voice: VOICES[args.voice]}

    for voice_name, voice_path in voices_to_test.items():
        if not voice_path.exists():
            print(f"\nVoice not found: {voice_name} ({voice_path})")
            continue

        print(f"\n{'='*60}")
        print(f"Voice: {voice_name}")
        print(f"{'='*60}")

        for i, phrase in enumerate(TEST_PHRASES):
            output_path = OUTPUT_DIR / f"tts_{voice_name}_{i+1}.wav"
            print(f"\n  [{i+1}] {phrase}")

            if synthesize(phrase, voice_path, output_path):
                import soundfile as sf
                data, sr = sf.read(str(output_path))
                print(f"    Saved: {output_path.name} ({len(data)/sr:.1f}s)")
                if not args.no_play:
                    play_audio(output_path)
            else:
                print(f"    FAILED")

    print("\nDone!")


if __name__ == "__main__":
    main()
