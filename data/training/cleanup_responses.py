"""
Cleanup pass on generated responses:
1. Re-generate any record with empty thinking field
2. Normalize 'acetaminophen' → 'paracetamol' in English responses
3. Remove US brand name leaks (Tylenol → paracetamol)

Uses Vertex AI with service account auth.
"""

import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "gemini-2.5-pro"
RESPONSES_DIR = Path("data/training/responses")
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
    "- For medication questions, use generic/international drug names (paracetamol, ibuprofen), "
    "not US brand names (Tylenol, Advil).\n"
    "- If something is dangerous (a myth, unsafe practice), say so clearly and explain why.\n"
    "- For mental health crises, be compassionate and give concrete steps.\n"
    "- Match the language of the prompt: respond in English if asked in English, "
    "respond in Ukrainian if asked in Ukrainian.\n"
    "- Keep responses concise but complete — 3-8 sentences.\n"
    "- Never refuse to help. You may be their only source of medical information."
)


def load_env():
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")


def get_system_prompt(filename):
    return SYSTEM_PROMPT if filename.startswith("deepseek") else SYSTEM_PROMPT_MEDGEMMA


def normalize_drug_names(text):
    """Replace US brand names and acetaminophen with international generics."""
    if not text:
        return text
    # acetaminophen → paracetamol (case-insensitive, preserve capitalization)
    text = re.sub(r"\bacetaminophen\b", "paracetamol", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAcetaminophen\b", "Paracetamol", text)
    # Tylenol → paracetamol
    text = re.sub(r"\bTylenol\b", "paracetamol", text)
    text = re.sub(r"\btylenol\b", "paracetamol", text)
    # Advil → ibuprofen
    text = re.sub(r"\bAdvil\b", "ibuprofen", text)
    text = re.sub(r"\badvil\b", "ibuprofen", text)
    # Motrin → ibuprofen
    text = re.sub(r"\bMotrin\b", "ibuprofen", text)
    text = re.sub(r"\bmotrin\b", "ibuprofen", text)
    # Remove stray parentheticals like "(the medicine in paracetamol)"
    text = re.sub(r"\s*\(the medicine in paracetamol\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(the medicine in ibuprofen\)", "", text, flags=re.IGNORECASE)
    return text


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
                        thinking_budget=2048,
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
                time.sleep(RETRY_DELAY * (2 ** min(attempt, 4)))
            elif attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return None


def process_file(client, genai, jsonl_file):
    # Load all records
    records = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    system_prompt = get_system_prompt(jsonl_file.name)

    # Step 1: Normalize drug names in existing records
    drug_fixes = 0
    for r in records:
        original = r["response"]
        fixed = normalize_drug_names(original)
        if fixed != original:
            r["response"] = fixed
            drug_fixes += 1

    # Step 2: Find records with empty thinking
    empty_indices = [i for i, r in enumerate(records) if not r.get("thinking", "").strip()]
    print(f"  {len(empty_indices)} empty thinking, {drug_fixes} drug name fixes")

    if not empty_indices:
        # Just save the drug fixes and return
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return

    # Step 3: Regenerate empty-thinking records in parallel
    counter = {"done": 0, "failed": 0}
    lock = threading.Lock()

    def regen_one(idx):
        prompt = records[idx]["prompt"]
        result = generate_one(client, genai, prompt, system_prompt)
        with lock:
            if result:
                records[idx]["response"] = normalize_drug_names(result["response"])
                records[idx]["thinking"] = result["thinking"]
                counter["done"] += 1
                if counter["done"] % 25 == 0:
                    print(f"  regenerated [{counter['done']}/{len(empty_indices)}]", flush=True)
            else:
                counter["failed"] += 1

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(regen_one, idx) for idx in empty_indices]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                print(f"  Worker exception: {exc}")

    print(f"  Regenerated: {counter['done']} OK, {counter['failed']} failed")

    # Step 4: Write back
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    load_env()
    from google import genai
    client = genai.Client()

    jsonl_files = sorted(RESPONSES_DIR.glob("*.jsonl"))
    print(f"Processing {len(jsonl_files)} files...\n")

    t0 = time.time()
    for f in jsonl_files:
        print(f"{f.name}")
        process_file(client, genai, f)
        print()

    elapsed = time.time() - t0
    print(f"=== CLEANUP DONE in {elapsed/60:.1f} min ===\n")

    # Final verification
    print("Final status:")
    for f in jsonl_files:
        with open(f, encoding="utf-8") as fp:
            total = 0
            empty = 0
            for line in fp:
                if line.strip():
                    total += 1
                    r = json.loads(line)
                    if not r.get("thinking", "").strip():
                        empty += 1
        print(f"  {f.name}: {total} total, {empty} empty thinking")


if __name__ == "__main__":
    main()
