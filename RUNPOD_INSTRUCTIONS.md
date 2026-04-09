# RunPod — Step by Step

Every time you need to train or benchmark on RunPod, follow these steps exactly.

---

## 1. Create the Pod

- Go to https://www.runpod.io/console/pods → Deploy
- **GPU**: A100 SXM 80GB ($1.49/hr)
- **Container disk: 50GB** (NOT 20GB — model downloads need temp space)
- **Volume disk**: 100GB
- **Template**: RunPod PyTorch 2.x
- **Check**: SSH terminal access
- Hit Deploy

---

## 2. Connect

- Click "Enable web terminal" on the pod page, then click the terminal link
- Or SSH: `ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519`

---

## 3. Setup (run these one at a time)

```
cd /workspace
```

```
git clone https://github.com/kpower7/holos-medyk.git
```

```
cd holos-medyk
```

```
pip install trl transformers peft accelerate sentence-transformers google-genai
```

```
export HF_HOME=/workspace/hf_cache
```

```
export HF_TOKEN=<your-hf-write-token>
```

```
export WANDB_MODE=disabled
```

```
export GOOGLE_API_KEY=<your-gemini-api-key>
```

---

## 4. Evaluate v5 SFT LoRA (pending from last session)

This loads the base model + your v5 LoRA adapter from HuggingFace and benchmarks
both against 15 eval scenarios. No merged model needed.

```
python evaluation/run_eval_unsloth.py
```

Run directly (not nohup) so you can watch. Takes ~20 min for 15 scenarios × 2 models.

If it errors on loading `kevpower/holos-medyk-lora-v5`, try loading the adapter
manually to debug:

```
python -c "
from unsloth import FastModel
m, t = FastModel.from_pretrained('kevpower/holos-medyk-lora-v5', max_seq_length=2048, load_in_4bit=True)
print('Loaded OK, type:', type(m))
print('Params:', sum(p.numel() for p in m.parameters()))
"
```

Results saved to `evaluation/results/benchmark_results.json`.

---

## 5. Train — GRPO

GRPO uses 4 reward functions instead of teacher forcing:

| Reward | What |
|---|---|
| No-refusal | Binary 0/1 — did you help or refuse? |
| Correctness | Gemini 2.5 Flash judges response against clinical criteria (20 parallel workers) |
| Similarity | Cosine similarity to teacher responses via sentence embeddings |
| Format | Length sweet spot, no repetition, no filler |

**Requires** `GOOGLE_API_KEY` for the LLM-as-judge correctness reward.

```
nohup python training/train_grpo.py > /workspace/grpo_log.txt 2>&1 &
```

```
tail -f /workspace/grpo_log.txt
```

- Single pass through 266 examples, 4 generations each
- ~1,064 Gemini Flash judge calls (20 concurrent, takes ~2 min of API time)
- Expect ~30-60 min total on A100 (model generation is the bottleneck)
- Watch the reward scores in the log — they should trend upward

---

## 6. Upload GRPO LoRA to HuggingFace

```
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('kevpower/holos-medyk-grpo-v1', exist_ok=True)
api.upload_folder(
    folder_path='training/outputs/holos_medyk_grpo_v1/lora_adapter',
    repo_id='kevpower/holos-medyk-grpo-v1',
)
print('Done')
"
```

---

## 7. Evaluate GRPO LoRA

Update the eval script to point at the GRPO adapter:

```
sed -i 's|kevpower/holos-medyk-lora-v5|kevpower/holos-medyk-grpo-v1|' evaluation/run_eval_unsloth.py
```

```
python evaluation/run_eval_unsloth.py
```

Results saved to `evaluation/results/benchmark_results.json`. Copy the old one first
if you want to keep v5 results:

```
cp evaluation/results/benchmark_results.json evaluation/results/benchmark_results_v5.json
```

---

## 8. Save Eval Results

Upload to HuggingFace for safekeeping:

```
python -c "
from huggingface_hub import HfApi
HfApi().upload_file(
    path_or_fileobj='evaluation/results/benchmark_results.json',
    path_in_repo='benchmark_results_grpo_v1.json',
    repo_id='kevpower/holos-medyk-grpo-v1',
)
print('Done')
"
```

---

## 9. STOP THE POD

Go to https://www.runpod.io/console/pods and hit **Stop**.

$1.49/hr adds up. Don't forget this.

---

## Typical Session Flows

### Eval only (~25 min, ~$0.60)

Steps: 1 → 2 → 3 → 4 → 9

### GRPO train + eval (~90 min, ~$2.25)

Steps: 1 → 2 → 3 → 5 → 6 → 7 → 8 → 9

---

## Common Gotchas

| Problem | Fix |
|---|---|
| `Disk quota exceeded` during model download | Container disk too small. Use 50GB not 20GB. Also set `export HF_HOME=/workspace/hf_cache` |
| `No module named 'unsloth'` | `pip install unsloth` |
| `wandb: No API key configured` | `export WANDB_MODE=disabled` |
| Training dies when laptop sleeps | Always use `nohup ... &`. Check with `tail -f` |
| GGUF export hangs on interactive prompt | Kill the process (`kill -9 $(pgrep -f train)`). LoRA is already saved. |
| `git clone` fails with auth error | Repo is public — just hit enter, don't enter credentials |
| Can't paste multi-line code into terminal | Write a script file locally, `git push`, then `git pull` on RunPod |
| `HF_TOKEN` not working | Make sure it has **write** permission. Get from https://huggingface.co/settings/tokens |
| `GOOGLE_API_KEY` errors | Get from https://aistudio.google.com/apikey or use Vertex AI service account |
| FastModel can't load LoRA from HF | Check the repo exists: `huggingface-cli repo info kevpower/holos-medyk-lora-v5` |

---

## File Locations on RunPod

| What | Where |
|---|---|
| Repo | `/workspace/holos-medyk/` |
| GRPO training script | `training/train_grpo.py` |
| SFT training script (legacy) | `training/train_holos_medyk.py` |
| Training data | `data/training/train_holos_medyk_v3_curated.jsonl` |
| Teacher responses | `data/training/responses/*.jsonl` |
| GRPO LoRA output | `training/outputs/holos_medyk_grpo_v1/lora_adapter/` |
| SFT LoRA output (legacy) | `training/outputs/holos_medyk_gemma4_e4b/lora_adapter/` |
| Training log | `/workspace/grpo_log.txt` |
| Eval script (Unsloth) | `evaluation/run_eval_unsloth.py` |
| Eval scenarios | `evaluation/eval_scenarios.jsonl` |
| Eval results | `evaluation/results/benchmark_results.json` |
| HF model cache | `/workspace/hf_cache/` |
