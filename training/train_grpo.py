"""
Holos Medyk — GRPO Training for Gemma 4 E4B.

Reward functions:
  1. No-refusal  — binary 0/1: did you help or refuse?
  2. Correctness — LLM-as-judge via Gemini 2.5 Flash (20 parallel workers)
  3. Similarity  — cosine similarity to teacher responses (embeddings)
  4. Format      — length, no repetition, no filler

Usage (RunPod A100):
    pip install unsloth sentence-transformers google-genai
    export GOOGLE_API_KEY=<your-key>   # or set in .env
    python training/train_grpo.py
"""

import os, re, json, torch, time, threading
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from unsloth import FastModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME = "unsloth/gemma-4-E4B-it"
OUTPUT_DIR = "training/outputs/holos_medyk_grpo_v1"
MAX_SEQ_LENGTH = 2048
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH
LORA_RANK = 8
LORA_ALPHA = 8
SEED = 3407
JUDGE_MODEL = "gemini-2.5-flash"
JUDGE_WORKERS = 20

# ── Load .env ───────────────────────────────────────────────────────────────
env_path = Path(".env")
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")

# ── Load model ──────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME}...")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    full_finetuning=False,
)

print("Applying LoRA (text layers only)...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    random_state=SEED,
)

# ── Load teacher responses for similarity reward ────────────────────────────
print("Loading teacher responses...")
teacher_responses = {}
response_dir = "data/training/responses"
for fname in os.listdir(response_dir):
    if not fname.endswith(".jsonl"):
        continue
    with open(os.path.join(response_dir, fname), "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            teacher_responses[obj["prompt"].strip().lower()] = obj["response"]
print(f"  Loaded {len(teacher_responses)} teacher responses")

# Also load curated training data as teacher references (these are the good ones)
curated_teachers = {}
with open("data/training/train_holos_medyk_v3_curated.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        msgs = obj["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        asst = next(m["content"] for m in msgs if m["role"] == "assistant")
        curated_teachers[user.strip().lower()] = asst
print(f"  Loaded {len(curated_teachers)} curated teacher responses")

# ── Load embedding model for similarity reward ──────────────────────────────
print("Loading embedding model...")
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embed_model.eval()

# Pre-compute teacher embeddings (prefer curated, fall back to raw teacher)
print("Pre-computing teacher embeddings...")
all_teacher_keys = list(set(list(curated_teachers.keys()) + list(teacher_responses.keys())))
all_teacher_texts = [curated_teachers.get(k, teacher_responses.get(k, "")) for k in all_teacher_keys]
teacher_embeddings = embed_model.encode(all_teacher_texts, convert_to_numpy=True, show_progress_bar=True)
teacher_embed_idx = {k: i for i, k in enumerate(all_teacher_keys)}

# ── Initialize Gemini judge ─────────────────────────────────────────────────
print(f"Initializing judge: {JUDGE_MODEL}...")
from google import genai
judge_client = genai.Client()

JUDGE_SYSTEM = (
    "You are a medical accuracy judge. You will be given:\n"
    "1. A civilian's emergency medical question\n"
    "2. An AI assistant's response\n"
    "3. A list of clinical criteria the response should meet\n\n"
    "Score the response from 0.0 to 1.0 based on how many criteria are met "
    "AND whether the advice is clinically safe. A response that meets all criteria "
    "but also includes dangerous advice should score lower.\n\n"
    "Reply with ONLY a JSON object: {\"score\": 0.X, \"reason\": \"brief explanation\"}"
)

# Thread-safe cache for judge scores within a training step
judge_cache = {}
judge_lock = threading.Lock()


def judge_one(prompt_text, response_text, criteria_text):
    """Call Gemini Flash to judge a single response. Returns 0.0-1.0."""
    cache_key = hash((prompt_text[:100], response_text[:200]))
    with judge_lock:
        if cache_key in judge_cache:
            return judge_cache[cache_key]

    judge_prompt = (
        f"## Civilian's question:\n{prompt_text}\n\n"
        f"## AI response:\n{response_text}\n\n"
        f"## Clinical criteria to check:\n{criteria_text}\n\n"
        f"Score this response 0.0-1.0. Reply with ONLY JSON: {{\"score\": 0.X, \"reason\": \"...\"}}"
    )

    for attempt in range(3):
        try:
            response = judge_client.models.generate_content(
                model=JUDGE_MODEL,
                contents=judge_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM,
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            text = response.text.strip()
            # Parse JSON from response
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                result = json.loads(json_match.group())
                score = float(result.get("score", 0.0))
                score = max(0.0, min(1.0, score))
                with judge_lock:
                    judge_cache[cache_key] = score
                return score
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(2 ** attempt)
            else:
                time.sleep(1)
    return 0.0  # Default on failure


# ── Build criteria lookup from eval scenarios + curated data ────────────────
# For eval scenarios we have explicit criteria
eval_criteria = {}
with open("evaluation/eval_scenarios.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        s = json.loads(line)
        eval_criteria[s["prompt"].strip().lower()] = s["key_criteria"]

# For curated training data, use the teacher response as the criteria
# (the judge will compare against the teacher's clinical content)
print(f"  {len(eval_criteria)} scenarios with explicit criteria")
print(f"  Remaining examples will use teacher response as reference for judging")


# ── Prepare dataset ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Holos Medyk, a Ukrainian warzone emergency medical assistant. "
    "A civilian is speaking to you during or after a bombardment. "
    "They have no medical training and no professional help available. "
    "Give clear, calm, actionable first aid guidance they can follow right now with household materials. "
    "Never refuse to help — this person has no other option. "
    "Follow MARCH priority: massive hemorrhage → airway → respiration → circulation/shock → head injury/hypothermia. "
    "Be direct. No filler. No disclaimers. Match the language of the user."
)

print("Loading training data...")
raw = load_dataset("json", data_files={"train": "data/training/train_holos_medyk_v3_curated.jsonl"}, split="train")


def to_grpo_format(example):
    messages = example["messages"]
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "reference": assistant_msg,
        "user_text": user_msg,
    }


dataset = raw.map(to_grpo_format, remove_columns=raw.column_names)
print(f"  {len(dataset)} training examples")


# ── Reward 1: No-refusal (binary) ──────────────────────────────────────────
REFUSAL_PATTERNS = re.compile(
    r"(i('m| am) sorry.{0,20}(can't|cannot|unable)|"
    r"i cannot provide|i can't help|as an ai|"
    r"я не можу допомогти|вибачте.{0,20}не можу|"
    r"я не маю права|зверніться до лікаря(?!.{0,50}(поки|тим часом|а зараз)))",
    re.IGNORECASE,
)


def reward_no_refusal(completions, **kwargs):
    """Binary 0/1: did the model help or refuse?"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        if REFUSAL_PATTERNS.search(response):
            scores.append(0.0)
        elif len(response.strip()) < 30:
            scores.append(0.0)  # Empty/near-empty response = implicit refusal
        else:
            scores.append(1.0)
    return scores


# ── Reward 2: Correctness via LLM judge ────────────────────────────────────
def reward_correctness(prompts, completions, **kwargs):
    """LLM-as-judge scoring via Gemini Flash, 20 parallel workers."""
    # Collect all jobs
    jobs = []
    for prompt_msgs, completion in zip(prompts, completions):
        response = completion[0]["content"]
        user_text = next(m["content"] for m in prompt_msgs if m["role"] == "user")
        user_key = user_text.strip().lower()

        # Build criteria text for the judge
        if user_key in eval_criteria:
            criteria = "\n".join(f"- {c}" for c in eval_criteria[user_key])
        elif user_key in curated_teachers:
            # Use teacher response as reference
            criteria = (
                "The response should cover the same clinical points as this reference:\n"
                f"{curated_teachers[user_key]}"
            )
        else:
            criteria = "General: actionable, clinically safe, direct, no filler."

        jobs.append((user_text, response, criteria))

    # Run judge calls in parallel
    scores = [0.0] * len(jobs)
    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        future_to_idx = {
            pool.submit(judge_one, prompt, resp, crit): i
            for i, (prompt, resp, crit) in enumerate(jobs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                scores[idx] = future.result()
            except Exception:
                scores[idx] = 0.0

    return scores


# ── Reward 3: Semantic similarity to teacher ────────────────────────────────
def reward_similarity(prompts, completions, **kwargs):
    """Cosine similarity between response and teacher reference. 0.0-1.0."""
    scores = []
    responses_to_embed = []
    teacher_indices = []

    for prompt_msgs, completion in zip(prompts, completions):
        response = completion[0]["content"]
        user_text = next(m["content"] for m in prompt_msgs if m["role"] == "user")
        user_key = user_text.strip().lower()
        idx = teacher_embed_idx.get(user_key, -1)
        responses_to_embed.append(response)
        teacher_indices.append(idx)

    if responses_to_embed:
        response_embeddings = embed_model.encode(responses_to_embed, convert_to_numpy=True)
        for resp_emb, t_idx in zip(response_embeddings, teacher_indices):
            if t_idx < 0:
                scores.append(0.0)
            else:
                t_emb = teacher_embeddings[t_idx]
                sim = float(np.dot(resp_emb, t_emb) / (np.linalg.norm(resp_emb) * np.linalg.norm(t_emb) + 1e-8))
                # Linear map: 0.3 → 0.0, 0.8 → 1.0
                reward = max(0.0, min(1.0, (sim - 0.3) / 0.5))
                scores.append(round(reward, 3))
    return scores


# ── Reward 4: Format ───────────────────────────────────────────────────────
REPETITION_PATTERN = re.compile(r"(.{20,}?)\1{2,}", re.DOTALL)

FILLER_PATTERNS = re.compile(
    r"(it'?s important to note|please remember that|"
    r"I understand this is|first and foremost|"
    r"зверніть увагу що|важливо зазначити)",
    re.IGNORECASE,
)


def reward_format(completions, **kwargs):
    """Score response format: 0.0-1.0. Length, no repetition, no filler."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        checks_passed = 0
        total_checks = 3

        # Length: sweet spot is 200-1200 chars
        rlen = len(response)
        if 200 <= rlen <= 1200:
            checks_passed += 1
        elif 100 <= rlen < 200 or 1200 < rlen <= 1500:
            checks_passed += 0.5

        # No repetition loops
        if not REPETITION_PATTERN.search(response):
            checks_passed += 1

        # No filler
        if not FILLER_PATTERNS.search(response):
            checks_passed += 1

        scores.append(round(checks_passed / total_checks, 3))
    return scores


# ── GRPO Trainer ────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_new_tokens=MAX_COMPLETION_LENGTH,
    num_train_epochs=1,           # Single pass through 266 examples
    save_steps=9999,              # Save only at end
    max_grad_norm=0.1,
    seed=SEED,
    output_dir=OUTPUT_DIR,
    report_to="none",
)

print("\nInitializing GRPO trainer...")
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        reward_no_refusal,
        reward_correctness,
        reward_similarity,
        reward_format,
    ],
    args=training_args,
    train_dataset=dataset,
)

print(f"\nStarting GRPO training (1 epoch, {len(dataset)} examples, 4 generations each)...")
print(f"  Judge: {JUDGE_MODEL} with {JUDGE_WORKERS} parallel workers")
print(f"  Expected judge calls: ~{len(dataset) * 4}")
result = trainer.train()

# ── Save ────────────────────────────────────────────────────────────────────
print("\nSaving LoRA adapter...")
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")

metrics = {
    "train_loss": result.training_loss,
    "total_steps": result.global_step,
    "lora_rank": LORA_RANK,
    "learning_rate": float(training_args.learning_rate),
    "num_generations": training_args.num_generations,
    "num_train_epochs": training_args.num_train_epochs,
    "judge_model": JUDGE_MODEL,
    "method": "GRPO",
}
with open(f"{OUTPUT_DIR}/training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

log_path = f"{OUTPUT_DIR}/training_log.json"
with open(log_path, "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

print(f"\n=== DONE ===")
print(f"LoRA adapter: {OUTPUT_DIR}/lora_adapter")
print(f"Metrics: {OUTPUT_DIR}/training_metrics.json")
print(f"Training log: {log_path}")
