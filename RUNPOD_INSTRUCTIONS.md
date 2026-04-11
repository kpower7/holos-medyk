# RunPod — Step by Step

Fresh pod, clean start. No Unsloth — plain transformers + peft + trl.

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

## 3. Setup

Run these one at a time. Copy-paste each block individually.

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

## 4. Smoke test (run this BEFORE training)

Verify everything imports and the model loads:

```
python -c "
import torch
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())

from transformers import AutoModelForCausalLM, AutoTokenizer
print('transformers OK')

from peft import LoraConfig
print('peft OK')

from trl import GRPOConfig, GRPOTrainer
print('trl GRPOTrainer OK')

from sentence_transformers import SentenceTransformer
print('sentence-transformers OK')

from google import genai
client = genai.Client()
r = client.models.generate_content(model='gemini-2.5-flash', contents='Say OK')
print('Gemini judge OK:', r.text[:20])

print()
print('=== ALL IMPORTS PASSED ===')
"
```

If any import fails, fix it before proceeding. If Gemini fails, check your GOOGLE_API_KEY.

---

## 5. Train — GRPO

GRPO with 4 reward functions. No Unsloth — plain transformers + peft + trl.

| Reward | What |
|---|---|
| No-refusal | Binary 0/1 — did you help or refuse? |
| Correctness | Gemini 2.5 Flash judges response (20 parallel workers) |
| Similarity | Cosine sim to teacher responses (sentence embeddings) |
| Format | Length, no repetition, no filler |

```
nohup python training/train_grpo.py > /workspace/grpo_log.txt 2>&1 &
```

```
tail -f /workspace/grpo_log.txt
```

**What to watch for:**

- Model loads in ~1-2 min (16GB download first time)
- `LoRA config: text decoder only` — confirms vision/audio excluded
- `Initializing GRPO trainer...` — means config accepted
- `Starting GRPO training...` — generation begins
- Reward scores should appear per step and trend upward
- Single pass through 266 examples, 4 generations each (~1,064 judge calls)
- Expected time: ~30-60 min on A100

**If it errors:** Check the error, fix locally, `git push`, then on RunPod:
```
pkill -9 -f train_grpo ; git pull && nohup python training/train_grpo.py > /workspace/grpo_log.txt 2>&1 & tail -f /workspace/grpo_log.txt
```

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

## 7. STOP THE POD

Go to https://www.runpod.io/console/pods and hit **Stop**.

$1.49/hr adds up. Don't forget this.

---

## Common Gotchas

| Problem | Fix |
|---|---|
| `Disk quota exceeded` during model download | Container disk too small. Use 50GB not 20GB. Also set `export HF_HOME=/workspace/hf_cache` |
| `wandb: No API key configured` | `export WANDB_MODE=disabled` |
| Training dies when laptop sleeps | Always use `nohup ... &`. Check with `tail -f` |
| `git clone` fails with auth error | Repo is public — just hit enter, don't enter credentials |
| `HF_TOKEN` not working | Make sure it has **write** permission. Get from https://huggingface.co/settings/tokens |
| `GOOGLE_API_KEY` errors | Get from https://aistudio.google.com/apikey |
| `ClippableLinear` or peft errors | Should be handled by monkey-patch in train_grpo.py. If not, `pip install --upgrade peft` |
| Gemini 429 rate limit | Built-in retry (5 attempts, exponential backoff). If persistent, reduce JUDGE_WORKERS in script |

---

## File Locations on RunPod

| What | Where |
|---|---|
| Repo | `/workspace/holos-medyk/` |
| GRPO training script | `training/train_grpo.py` |
| Training data | `data/training/train_holos_medyk_v3_curated.jsonl` |
| Teacher responses | `data/training/responses/*.jsonl` |
| GRPO LoRA output | `training/outputs/holos_medyk_grpo_v1/lora_adapter/` |
| Training log | `/workspace/grpo_log.txt` |
| Eval scenarios | `evaluation/eval_scenarios.jsonl` |
| Eval results | `evaluation/results/benchmark_results.json` |
| HF model cache | `/workspace/hf_cache/` |

---

## Cost Tracking

| Task | Time | Cost |
|---|---|---|
| GRPO training (1 epoch, 266 examples) | ~30-60 min | ~$1-1.50 |
| Model upload to HF | ~5 min | ~$0.15 |
| **Typical session** | **~45-70 min** | **~$1.50-2** |
