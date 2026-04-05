"""
Generate training responses using Gemini 2.5 Pro via Vertex AI.
Parallel workers. Resume-safe. Captures thinking for distillation.
"""

import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "gemini-2.5-pro"
PROMPTS_DIR = Path("data/training/prompts")
OUTPUT_DIR = Path("data/training/responses")
WORKERS = 20
MAX_RETRIES = 10
RETRY_DELAY = 10

SYSTEM_PROMPT = (
    "You are Holos Medyk, a Ukrainian warzone medical voice assistant running on a "
    "Raspberry Pi with no internet connection. A civilian is speaking to you during "
    "or after a bombardment. You must give clear, actionable, step-by-step first aid "
    "guidance that a non-medical person can follow immediately.\n\n"
    "Rules:\n"
    "- Be direct and calm. No filler, no disclaimers about seeking professional help "
    "(there is none available).\n"
    "- Give specific, physical instructions: press here, wrap this, don't move them.\n"
    "- If the person describes something dangerous they want to do (pull out shrapnel, "
    "give random pills, attempt surgery), firmly tell them NOT to and explain why, "
    "then give the safe alternative.\n"
    "- If the situation is psychological (panic, freeze, child screaming), give grounding "
    "techniques and practical next steps.\n"
    "- Match the language of the prompt: respond in English if asked in English, "
    "respond in Ukrainian if asked in Ukrainian.\n"
    "- Keep responses concise but complete — this is voice output, not a textbook. "
    "Aim for 3-8 sentences.\n"
    "- Never say I'm an AI or I can't diagnose. You are their only medical help."
)

SYSTEM_PROMPT_MEDGEMMA = (
    "You are Holos Medyk, a medical assistant helping civilians with health questions. "
    "You provide clear, evidence-based medical guidance in plain language.\n\n"
    "Rules:\n"
    "- Be direct and helpful. Explain medical concepts simply.\n"
    "- Give actionable advice: what to watch for, when to worry, when to go to hospital.\n"
    "- For medication questions, use generic/international drug names, not brand names.\n"
    "- If something is dangerous (a myth, unsafe practice), say so clearly and explain why.\n"
    "- For mental health crises, be compassionate and give concrete steps.\n"
    "- Match the language of the prompt: respond in English if asked in English, "
    "respond in Ukrainian if asked in Ukrainian.\n"
    "- Keep responses concise but complete — 3-8 sentences.\n"
    "- Never refuse to help. You may be their only source of medical information."
)


def get_system_prompt(filename):
    return SYSTEM_PROMPT if filename.startswith("deepseek") else SYSTEM_PROMPT_MEDGEMMA


def load_env():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")


def generate_one(client, genai, prompt, system_prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    max_output_tokens=3072,
                    thinking_config=genai.types.ThinkingConfig(
                        thinking_budget=1024,
                        include_thoughts=True,
                    ),
                ),
            )
            thoughts = []
            answers = []
            for candidate in response.candidates or []:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            if getattr(part, "thought", None):
                                thoughts.append(part.text)
                            else:
                                answers.append(part.text)
            if answers:
                return {
                    "response": " ".join(answers).strip().replace("\n", " "),
                    "thinking": " ".join(thoughts).strip().replace("\n", " ") if thoughts else "",
                }
            return None
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = RETRY_DELAY * (2 ** min(attempt, 4))
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  FAILED: {err[:150]}")
    return None


def process_file(client, genai, prompt_file, output_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    system_prompt = get_system_prompt(prompt_file.name)

    done_prompts = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done_prompts.add(json.loads(line)["prompt"])
                    except (json.JSONDecodeError, KeyError):
                        pass
        if done_prompts:
            print(f"  Resuming: {len(done_prompts)}/{len(prompts)} done")

    remaining = [(i, p) for i, p in enumerate(prompts) if p not in done_prompts]
    if not remaining:
        print(f"  Complete ({len(prompts)}/{len(prompts)})")
        return

    print(f"  {len(remaining)} to go, {WORKERS} workers")

    write_lock = threading.Lock()
    counter = {"done": len(done_prompts), "failed": 0, "total": len(prompts)}

    def do_one(idx_prompt):
        _, prompt = idx_prompt
        result = generate_one(client, genai, prompt, system_prompt)
        if result:
            record = json.dumps(
                {"prompt": prompt, "response": result["response"], "thinking": result["thinking"]},
                ensure_ascii=False,
            )
            with write_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(record + "\n")
                counter["done"] += 1
                d = counter["done"]
                if d % 50 == 0 or d == counter["total"]:
                    print(f"  [{d}/{counter['total']}]", flush=True)
        else:
            with write_lock:
                counter["failed"] += 1

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(do_one, item) for item in remaining]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as exc:
                print(f"  Worker exception: {exc}")

    print(f"  File done: {counter['done']} OK, {counter['failed']} failed")


def main():
    load_env()

    # Import after env is loaded
    from google import genai
    client = genai.Client()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt_files = sorted(PROMPTS_DIR.glob("*.txt"))
    total = sum(sum(1 for l in open(f, encoding="utf-8") if l.strip()) for f in prompt_files)
    print(f"{len(prompt_files)} files, {total} prompts, model={MODEL} (Vertex AI)\n")

    t0 = time.time()
    for pf in prompt_files:
        of = OUTPUT_DIR / pf.name.replace(".txt", ".jsonl")
        print(f"{pf.name}")
        process_file(client, genai, pf, of)
        print()

    elapsed = time.time() - t0
    print(f"=== DONE in {elapsed/60:.1f} min ===")
    total = 0
    for jsonl in sorted(OUTPUT_DIR.glob("*.jsonl")):
        with open(jsonl, "r", encoding="utf-8") as f:
            count = sum(1 for l in f if l.strip())
        print(f"  {jsonl.name}: {count}")
        total += count
    print(f"  Total: {total} training pairs")


if __name__ == "__main__":
    main()
