# Holos Medyk — Context Handoff

Portable context for picking up this project on a new workstation. When opening
Claude Code on a fresh machine, read this file first and rebuild the memory
system from these entries.

---

## User profile (Kevin)

- 3x MIT Engineering (SCM '25, TPP '27, EECS '27)
- Winner of OpenAI red-teaming challenge, Hack-Nation Global AI Hackathon
- IEEE published x2, Founder @ Powerlab (agentic AI systems)
- Experienced with Unsloth fine-tuning, Ollama, agentic tool-calling, LLM safety
- Working on Gemma 4 Good Hackathon (deadline May 18, 2026)
- Has Ukrainian girlfriend Diana — native speaker, cultural/language advisor,
  AND active coding/build contributor
- Has brother who is a trauma surgeon — medical advisor
- Prefers fast, direct action. Gets frustrated when things are slow or when
  trial-and-error happens instead of planning
- Wants to see progress, doesn't want unnecessary explanations
- Works late, high intensity, "we never stop" mentality
- Does not want files committed to git unless explicitly asked

---

## Feedback rules (learned the hard way)

### 1. Plan before acting

STOP and research (web search, read docs) before attempting installs or writing
scripts. Do not trial-and-error pip installs or write code based on assumptions.

**Why:** Wasted significant time installing transformers+bitsandbytes (wrong
runtime entirely), then llama-cpp-python failed to build (no compiler), then
binary package was broken. Multiple failed attempts frustrated the user badly.

**How to apply:** Before any package install or new tool setup: (1) web search
for platform compatibility, (2) verify it works on target platform, (3) check
dependencies, (4) present a plan to the user BEFORE executing. Never just "try
and see."

### 2. Use production runtime for development

When building an edge/mobile deployment project, use the actual deployment
runtime (e.g. LiteRT-LM, llama.cpp) from the start — not research stacks like
transformers+bitsandbytes.

**Why:** User was furious when we used a heavy Python ML stack
(transformers+bitsandbytes) that couldn't even run on a $2000 laptop, when the
whole point is running on phones and Pi. The production runtime should have
been the starting point.

**How to apply:** For any on-device/edge ML project, identify the deployment
runtime first and develop against it from day one. If it doesn't run on the
dev machine, find the closest equivalent.

### 3. Always use parallel workers for LLM API calls

ALWAYS use parallel async workers (10-20 concurrent) when making LLM API calls
for batch data generation or processing. Never send requests one at a time.

**Why:** A sequential DeepSeek R1 run of 172 prompts took ~60 minutes. The
Vertex AI run of 3,511 prompts with 20 parallel workers took 26 minutes.
Sequential is 10-20x slower for no reason.

**How to apply:** Use `asyncio` + `aiohttp` or
`concurrent.futures.ThreadPoolExecutor(max_workers=20)` for any batch LLM API
script. Even rate-limited APIs can handle 10+ concurrent requests.

### 4. Don't commit every file without asking

Only commit when Kevin explicitly asks. Test things work before committing.
Don't treat every change as a commit-worthy event.

---

## Hardware context

**IMPORTANT:** The old laptop (RTX A1000 6GB, Windows 11, 32GB RAM) has
machine-specific state that does NOT carry over. The new workstation (Nvidia
hardware, unspecified at time of writing) needs its own inference setup.

On the old laptop, the working configuration was:
- `tools/llama-cpp/llama-cli.exe` prebuilt binary
- `models/google_gemma-4-E4B-it-Q3_K_M.gguf` (4.6GB, 30 tok/s)
- `models/google_gemma-4-E2B-it-Q4_K_M.gguf` (3.3GB, 60 tok/s)
- MUST use `-ngl 99` — CPU mode or partial offload crashes Windows
- Ukrainian text must be passed via `-f` file due to terminal encoding

On the new workstation, plan fresh:
- Identify the GPU and VRAM
- Pick inference runtime (llama.cpp, vLLM, or transformers depending on VRAM)
- Download model weights fresh
- Don't assume old paths work

---

## Hardware status (other devices)

**Raspberry Pi 5** (8GB, CanaKit Aluminum Pro) — set up as of 2026-04-04:
- LiteRT-LM installed via pip in `~/holos-env` venv
- Gemma 4 E4B at
  `~/.cache/huggingface/hub/models--litert-community--gemma-4-E4B-it-litert-lm/`
- Native audio input works via litert_lm Python API (no Whisper needed)
- Piper TTS Ukrainian voice installed at `~/voices/`
- Bluetooth headset (Sony WH-CH720N) paired
- SSH: username `voice-help-ua`, connect via Pi's IP
- **Problem:** inference is ~2 min text, ~10 min audio — CPU only, needs GPU
  backend experimentation

**Samsung A15** — Android deployment target, weak device

---

## Project state

**Core premise:** offline voice-interactive emergency medical assistant for
Ukrainian civilians under bombardment. Gemma 4 E4B model. Deployed as
Android app (Flutter + LiteRT-LM), Raspberry Pi device, and Windows dev
machine (llama.cpp).

### TTS status
- **Piper TTS Ukrainian is BROKEN** (native speaker verdict from Diana,
  2026-04-04). Words correct but voice unintelligible. Do NOT use.
- **Whisper ASR works fine** for Ukrainian transcription.
- TTS comparison script ready at `scripts/test_tts_comparison.py` testing
  Silero v5 cis_base, Silero v4_ua, robinhad/ukrainian-tts, eSpeak-ng.
  Waiting on Diana's evaluation to pick winner.

### Training status (as of Day 6, 2026-04-09)
- **5 SFT runs failed** (v1–v5) with various issues: generation looping,
  refusals, catastrophic overfitting, LoRA layer targeting bug, eval tokenizer
  bug.
- **Current approach: GRPO** with 4 reward functions:
  1. No-refusal (binary)
  2. Correctness (Gemini 2.5 Flash LLM-as-judge, 20 parallel workers)
  3. Similarity (cosine sim to teacher via sentence embeddings)
  4. Format (length, no repetition, no filler)
- **RunPod GRPO attempt BLOCKED** by Unsloth/TRL dependency conflicts
  (6 consecutive errors documented in `paper/holos_medyk.tex` Day 6 entry).
- **Current workaround:** `training/holos_medyk_grpo_colab.ipynb` — a Colab
  notebook adapted from Unsloth's tested Gemma 4 E4B Text SFT notebook,
  with GRPO trainer swapped in. **NOT YET TESTED** — upload to Colab L4
  and run.

### Key files
- [training/holos_medyk_grpo_colab.ipynb](training/holos_medyk_grpo_colab.ipynb)
  — GRPO notebook for Colab
- [training/train_grpo.py](training/train_grpo.py) — RunPod GRPO script
  (5 known bugs fixed but environment blocked)
- [training/train_holos_medyk.py](training/train_holos_medyk.py) — SFT
  script, v5 completed cleanly but eval tokenizer bug blocks benchmarking
- [data/training/train_holos_medyk_v3_curated.jsonl](data/training/train_holos_medyk_v3_curated.jsonl)
  — 266 audited examples, the current training set
- [data/training/responses/](data/training/responses/) — teacher responses
  from DeepSeek R1 and MedGemma (used for similarity reward)
- [evaluation/eval_scenarios.jsonl](evaluation/eval_scenarios.jsonl) — 17
  benchmark scenarios with clinical criteria
- [paper/holos_medyk.tex](paper/holos_medyk.tex) — living ICML 2026 paper,
  contains full daily changelog (Day 0 through Day 6)

---

## Immediate next steps (pick up here)

1. **Rotate leaked secrets:**
   - HuggingFace: https://huggingface.co/settings/tokens (old token was in git
     commit 4077a38 briefly)
   - Google AI Studio: https://aistudio.google.com/apikey
2. **Test the Colab GRPO notebook** on Colab L4:
   - File → Open notebook → GitHub tab → `kpower7/holos-medyk` →
     `training/holos_medyk_grpo_colab.ipynb`
   - Add `GOOGLE_API_KEY` and `HF_TOKEN` to Colab Secrets
   - Runtime type: L4 GPU
   - Run all cells
3. **If GRPO works:** evaluate against baseline, upload adapter to HF as
   `kevpower/holos-medyk-grpo-v1`
4. **If GRPO fails:** paste error, iterate on the notebook
5. **Deferred:** fix the eval tokenizer bug in
   `evaluation/run_eval_unsloth.py` so we can finally benchmark v5 SFT
6. **Deferred:** RAG integration for shelter locations / city protocols

---

## How to rebuild memory on new machine

On the new workstation, after opening Claude Code in this repo:

1. Read this file (`CONTEXT.md`)
2. Create the memory directory:
   `~/.claude/projects/<project-hash>/memory/` (path will differ by machine)
3. Ask me to write individual memory files for each section above into that
   directory, following Claude Code's auto-memory format (frontmatter + body)
4. Ask me to create the `MEMORY.md` index file

Or simpler: just leave this `CONTEXT.md` in the repo and I'll reference it
directly when relevant questions come up.
