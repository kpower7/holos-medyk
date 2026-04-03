# Голос Медик — Holos Medyk

An offline, voice-interactive emergency medical assistant for Ukrainian civilians under bombardment.

**Gemma 4 Good Hackathon Entry**

## What It Does

Speak in Ukrainian about a medical emergency. Gemma 4 E4B talks you through saving a life — step by step, in Ukrainian, completely offline.

- **Tier 1**: Free Android app (~4GB) — works on any modern smartphone
- **Tier 2**: $75 standalone device (Raspberry Pi 5) — for shelters and aid stations

## Tech Stack

- **LLM**: Gemma 4 E4B (4.5B params), fine-tuned on trauma surgeon protocols
- **ASR**: Gemma 4 native audio input
- **TTS**: Piper TTS (Ukrainian voice)
- **Runtime**: LiteRT-LM (on-device)
- **Fine-tuning**: Unsloth (LoRA/QLoRA)

## Setup

```bash
conda create -n holos python=3.11 -y
conda activate holos
pip install -r requirements.txt
```

## Quick Test

```bash
# Text inference test
python scripts/test_inference.py

# Voice pipeline test (requires mic + speaker)
python scripts/test_voice_pipeline.py
```

## Project Structure

```
holos-medyk/
├── scripts/          # Testing and utility scripts
├── src/              # Core application code
│   ├── asr/          # Speech recognition
│   ├── llm/          # LLM inference
│   ├── tts/          # Text-to-speech
│   └── pipeline/     # End-to-end voice pipeline
├── data/             # Training data and knowledge base
│   ├── protocols/    # Medical protocols (from brother)
│   ├── glossary/     # Ukrainian medical terminology
│   └── training/     # Fine-tuning datasets
├── models/           # Model weights (gitignored)
├── voices/           # Piper TTS voices (gitignored)
├── android/          # Android app (Phase 3)
└── pi/               # Raspberry Pi deployment configs
```

## Team

- **Kevin Power** — Builder (MIT x3, IEEE x2)
- **Brother** — Medical Advisor (Trauma Surgeon)
- **Girlfriend** — Ukrainian Language & Cultural Advisor

---

*Голос Медик. A voice when no one else can answer.*
