# RunPod Training & Eval — Step by Step

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
pip install unsloth wandb
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

---

## 4. Train

```
nohup python training/train_holos_medyk.py > /workspace/train_log.txt 2>&1 &
```

Check progress (can disconnect and reconnect anytime):
```
tail -f /workspace/train_log.txt
```

**IMPORTANT**: Always use `nohup`. Your laptop can sleep — the training continues on the server.

Training takes ~35-45 minutes on A100. When you see the training metrics summary and "Saving LoRA adapter..." it's done.

**GGUF export will hang** asking for interactive input. If it hangs, open a second terminal tab and kill it:
```
kill -9 $(pgrep -f train_holos)
```
The LoRA adapter and merged model are already saved at this point.

---

## 5. Upload Models to HuggingFace

Replace `v2` with the version you're training (v3, v4, etc.):

**LoRA adapter (~357MB, fast):**
```
python -c "from huggingface_hub import HfApi; api = HfApi(); api.create_repo('kevpower/holos-medyk-lora-v2', exist_ok=True); api.upload_folder(folder_path='training/outputs/holos_medyk_gemma4_e4b/lora_adapter', repo_id='kevpower/holos-medyk-lora-v2'); print('LoRA done')"
```

**Merged model (~15GB, takes a few minutes):**
```
python -c "from huggingface_hub import HfApi; api = HfApi(); api.create_repo('kevpower/holos-medyk-merged-v2', exist_ok=True); api.upload_folder(folder_path='training/outputs/holos_medyk_gemma4_e4b/merged_16bit', repo_id='kevpower/holos-medyk-merged-v2'); print('Merged done')"
```

---

## 6. Run Evaluation Benchmark

**BEFORE running**: make sure `evaluation/run_eval_runpod.py` points to the correct fine-tuned model on HuggingFace. Check this line:
```
grep "holos-medyk-merged" evaluation/run_eval_runpod.py
```
If it says the wrong version, fix it:
```
sed -i 's|kevpower/holos-medyk-merged-v2|kevpower/holos-medyk-merged-v3|' evaluation/run_eval_runpod.py
```

**Install eval deps** (if not already installed from training):
```
pip install git+https://github.com/huggingface/transformers.git accelerate sentencepiece
```

**Run the benchmark:**
```
python evaluation/run_eval_runpod.py
```

Run it directly (not nohup) so you can watch it. Takes ~20 minutes for 15 scenarios x 2 models.

---

## 7. Save Eval Results to HuggingFace

```
python -c "from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='evaluation/results/benchmark_results.json', path_in_repo='benchmark_results_v2.json', repo_id='kevpower/holos-medyk-merged-v2'); print('Done')"
```

---

## 8. STOP THE POD

Go to https://www.runpod.io/console/pods and hit **Stop**.

$1.49/hr adds up. Don't forget this.

---

## Common Gotchas

| Problem | Fix |
|---|---|
| `Disk quota exceeded` during model download | Container disk too small. Use 50GB not 20GB. Also set `export HF_HOME=/workspace/hf_cache` |
| `No module named 'unsloth'` | `pip install unsloth` |
| `wandb: No API key configured` | `export WANDB_MODE=disabled` |
| Training dies when laptop sleeps | Always use `nohup ... &`. Check with `tail -f /workspace/train_log.txt` |
| GGUF export hangs on interactive prompt | Kill the process. LoRA + merged model are already saved. Convert GGUF locally later. |
| `git clone` fails with auth error | Repo is public — don't enter username/password, just hit enter or use the raw URL |
| Can't paste multi-line code into terminal | Write a script file locally, `git push`, then `git pull` on RunPod |
| `HF_TOKEN` not working | Make sure it has **write** permission. Get from https://huggingface.co/settings/tokens |

---

## Cost Tracking

| Task | Time | Cost |
|---|---|---|
| Training (3 epochs, 3,511 examples) | ~35 min | ~$1 |
| Model merge + upload | ~15 min | ~$0.40 |
| Evaluation benchmark (15 scenarios x 2 models) | ~20 min | ~$0.50 |
| **Typical full session** | **~70 min** | **~$2** |

---

## File Locations on RunPod

| What | Where |
|---|---|
| Repo | `/workspace/holos-medyk/` |
| Training script | `training/train_holos_medyk.py` |
| Training data | `data/training/train_holos_medyk_v2.jsonl` |
| LoRA adapter output | `training/outputs/holos_medyk_gemma4_e4b/lora_adapter/` |
| Merged model output | `training/outputs/holos_medyk_gemma4_e4b/merged_16bit/` |
| GGUF output (if successful) | `training/outputs/holos_medyk_gemma4_e4b/gguf_q4_k_m/` |
| Training log | `/workspace/train_log.txt` |
| Training metrics | `training/outputs/holos_medyk_gemma4_e4b/training_metrics.json` |
| Eval script | `evaluation/run_eval_runpod.py` |
| Eval results | `evaluation/results/benchmark_results.json` |
| HF model cache | `/workspace/hf_cache/` |
