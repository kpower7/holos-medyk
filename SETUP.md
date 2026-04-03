# Holos Medyk — Dev Environment Setup

## Hardware Requirements (Laptop Dev)
- NVIDIA GPU with 4GB+ VRAM (tested: RTX A1000 6GB)
- 32GB RAM recommended (16GB causes OOM crashes on Windows)
- Windows 11

## 1. Install llama.cpp (no compiler needed)

Download prebuilt CUDA binary from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases):

```
llama-<version>-bin-win-cuda-12.4-x64.zip
cudart-llama-bin-win-cuda-12.4-x64.zip
```

Extract both into `tools/llama-cpp/`. The CUDA runtime DLLs must be in the same folder as `llama-cli.exe`.

## 2. Download GGUF Models

```bash
conda activate holos
hf download bartowski/google_gemma-4-E4B-it-GGUF --include "google_gemma-4-E4B-it-Q3_K_M.gguf" --local-dir models/
hf download bartowski/google_gemma-4-E2B-it-GGUF --include "google_gemma-4-E2B-it-Q4_K_M.gguf" --local-dir models/
```

## 3. Run Inference

```bash
# E4B (primary model — 30 tok/s on RTX A1000)
python scripts/inference.py "Моя дочка кровоточить з руки. Що робити?"

# E2B (lighter model — 60 tok/s on RTX A1000)
python scripts/inference.py --model e2b "Моя дочка кровоточить з руки. Що робити?"

# From file (avoids terminal encoding issues with Cyrillic)
python scripts/inference.py --file scripts/prompt_uk.txt

# Interactive chat
python scripts/inference.py --interactive
```

## Key Flags
- `-ngl 99` — offload all layers to GPU (critical — CPU mode will OOM on Windows)
- `-c 2048` — context window size
- `-n 1024` — max output tokens
- `--single-turn` — generate once and exit (no interactive loop)

## What NOT to Do
- Do NOT use `transformers` + `bitsandbytes` for inference — Python ML overhead is massive, will crash on 6GB VRAM
- Do NOT use `-ngl 0` (CPU only) — loads entire model into RAM, will OOM even at 32GB on Windows
- Do NOT use partial GPU offload (`-ngl 10-20`) — triggers GGML split assertion bug with Gemma 4 architecture

## Performance (RTX A1000 6GB, 32GB RAM)

| Model | Quant | File Size | VRAM Used | Prompt | Generation |
|-------|-------|-----------|-----------|--------|------------|
| E4B   | Q3_K_M | 4.6 GB   | ~3.0 GB   | 383 t/s | 30 t/s    |
| E2B   | Q4_K_M | 3.3 GB   | ~2.0 GB   | 530 t/s | 60 t/s    |

## Production Runtime
- Android + Pi will use **LiteRT-LM** (not llama.cpp)
- llama.cpp is for Windows development/testing only
- LiteRT-LM does not support Windows — Linux/macOS/Android/Pi only
