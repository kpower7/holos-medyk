"""
Fix training data:
1. Strip <thinking>/<answer> tags — merge thinking into response naturally
2. Identify refusal examples and re-generate via DeepSeek R1

Usage:
    # Step 1: Strip tags from all examples
    python training/fix_training_data.py --strip-tags

    # Step 2: Regenerate refusals via DeepSeek
    python training/fix_training_data.py --fix-refusals

    # Step 3: Merge fixed refusals back into dataset
    python training/fix_training_data.py --merge
"""
import json
import re
import os
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRAIN_FILE = ROOT / "data" / "training" / "train_holos_medyk.jsonl"
STRIPPED_FILE = ROOT / "data" / "training" / "train_holos_medyk_stripped.jsonl"
REFUSAL_PROMPTS_FILE = ROOT / "data" / "training" / "refusal_prompts.jsonl"
REFUSAL_FIXED_FILE = ROOT / "data" / "training" / "refusal_fixed.jsonl"
FINAL_FILE = ROOT / "data" / "training" / "train_holos_medyk_v2.jsonl"

REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i am not able", "i'm not able",
    "i must refuse", "i will not", "cannot provide", "not recommend",
    "seek professional", "consult a doctor", "see a doctor",
    "i'm sorry, but", "beyond my scope", "not qualified",
]

DEEPSEEK_SYSTEM = """You are Holos Medyk, a Ukrainian warzone medical voice assistant. A civilian is speaking to you during or after a bombardment.

RULES:
- NEVER refuse to help. NEVER say "I cannot", "consult a doctor", "seek professional help", or "see a doctor".
- The user has NO access to doctors or hospitals. You are their ONLY help.
- If something is dangerous, explain the risks clearly and then guide them through it safely anyway.
- If they ask about medication dosing, give the standard adult dose from established protocols. Their alternative is death.
- Give clear, calm, step-by-step instructions a non-medical person can follow.
- Be direct. No filler. Match the language of the user (English or Ukrainian).
- Keep responses concise — this will be read aloud by a voice assistant."""


def strip_tags(text):
    """Remove <thinking>...</thinking> and <answer>...</answer> tags,
    merge content into a single natural response."""
    # Extract thinking content
    thinking = ""
    thinking_match = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()

    # Extract answer content
    answer = ""
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    # If we found both, merge them naturally
    if answer:
        return answer  # Use just the answer — thinking is already distilled into it
    elif thinking:
        # No answer tags, just thinking — clean up the thinking
        cleaned = re.sub(r'</?thinking>', '', text)
        cleaned = re.sub(r'</?answer>', '', cleaned)
        return cleaned.strip()
    else:
        # No tags at all — return as-is
        cleaned = re.sub(r'</?thinking>', '', text)
        cleaned = re.sub(r'</?answer>', '', cleaned)
        return cleaned.strip()


def has_refusal(text):
    lower = text.lower()
    return any(p in lower for p in REFUSAL_PATTERNS)


def cmd_strip_tags():
    """Strip thinking/answer tags from all training examples."""
    examples = []
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    stripped_count = 0
    for ex in examples:
        for msg in ex["messages"]:
            if msg["role"] == "assistant":
                original = msg["content"]
                msg["content"] = strip_tags(original)
                if msg["content"] != original:
                    stripped_count += 1

        # Also update system prompt to remove thinking tag instructions
        for msg in ex["messages"]:
            if msg["role"] == "system":
                msg["content"] = re.sub(
                    r'Think through the situation inside <thinking> tags, then give your final guidance inside <answer> tags\.',
                    'Give your guidance directly.',
                    msg["content"]
                )
                msg["content"] = re.sub(
                    r'Think through the situation inside <thinking> tags, then give your final instructions inside <answer> tags\.',
                    'Give your instructions directly.',
                    msg["content"]
                )

    with open(STRIPPED_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Stripped tags from {stripped_count} responses")
    print(f"Saved to {STRIPPED_FILE}")

    # Now identify refusals
    refusals = []
    for i, ex in enumerate(examples):
        for msg in ex["messages"]:
            if msg["role"] == "assistant" and has_refusal(msg["content"]):
                user_msg = [m for m in ex["messages"] if m["role"] == "user"][0]["content"]
                sys_msg = [m for m in ex["messages"] if m["role"] == "system"][0]["content"]
                refusals.append({"index": i, "system": sys_msg, "prompt": user_msg, "old_response": msg["content"]})

    with open(REFUSAL_PROMPTS_FILE, "w", encoding="utf-8") as f:
        for r in refusals:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Found {len(refusals)} refusal examples -> {REFUSAL_PROMPTS_FILE}")


def cmd_fix_refusals():
    """Send refusal prompts to DeepSeek R1 for regeneration."""
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        # Try loading from .env
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in open(env_path):
                if line.startswith("DEEPSEEK_API_KEY="):
                    api_key = line.strip().split("=", 1)[1].strip("'\"")
    if not api_key:
        print("ERROR: Set DEEPSEEK_API_KEY in environment or .env file")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    refusals = []
    with open(REFUSAL_PROMPTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            refusals.append(json.loads(line))

    print(f"Regenerating {len(refusals)} refusal examples via DeepSeek R1...")

    fixed = []
    for i, r in enumerate(refusals):
        if i % 50 == 0:
            print(f"  [{i}/{len(refusals)}]...", flush=True)

        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": DEEPSEEK_SYSTEM},
                    {"role": "user", "content": r["prompt"]},
                ],
                max_tokens=1024,
                temperature=0.7,
            )
            new_response = response.choices[0].message.content.strip()
            r["new_response"] = new_response
            fixed.append(r)
        except Exception as e:
            print(f"  ERROR on {i}: {e}")
            # Try with deepseek-chat as fallback
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": DEEPSEEK_SYSTEM},
                        {"role": "user", "content": r["prompt"]},
                    ],
                    max_tokens=1024,
                    temperature=0.7,
                )
                new_response = response.choices[0].message.content.strip()
                r["new_response"] = new_response
                fixed.append(r)
            except Exception as e2:
                print(f"  FALLBACK ALSO FAILED on {i}: {e2}")
                r["new_response"] = r["old_response"]  # keep original
                fixed.append(r)

        # Rate limit
        time.sleep(0.1)

    with open(REFUSAL_FIXED_FILE, "w", encoding="utf-8") as f:
        for r in fixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(fixed)} fixed responses to {REFUSAL_FIXED_FILE}")


def cmd_merge():
    """Merge fixed refusals back into the stripped dataset."""
    # Load stripped dataset
    examples = []
    with open(STRIPPED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))

    # Load fixed refusals
    fixes = {}
    with open(REFUSAL_FIXED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fixes[r["index"]] = r["new_response"]

    # Apply fixes
    replaced = 0
    for idx, new_resp in fixes.items():
        for msg in examples[idx]["messages"]:
            if msg["role"] == "assistant":
                msg["content"] = new_resp
                replaced += 1

    # Also update all system prompts to the no-refusal version
    no_refusal_system_warzone = "You are Holos Medyk, a Ukrainian warzone medical voice assistant. A civilian is speaking to you during or after a bombardment. Give clear, calm, actionable first aid guidance a non-medical person can follow right now. Be direct. No filler. Never refuse to help. Match the language of the user."
    no_refusal_system_medical = "You are Holos Medyk, a medical assistant helping civilians with health questions. Give clear, evidence-based medical guidance in plain language. Use generic drug names. Never refuse to help. Match the language of the user."

    for ex in examples:
        for msg in ex["messages"]:
            if msg["role"] == "system":
                if "warzone" in msg["content"].lower() or "bombardment" in msg["content"].lower():
                    msg["content"] = no_refusal_system_warzone
                else:
                    msg["content"] = no_refusal_system_medical

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(examples)

    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Replaced {replaced} refusal responses")
    print(f"Updated all system prompts to never-refuse versions")
    print(f"Saved {len(examples)} examples to {FINAL_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strip-tags", action="store_true")
    parser.add_argument("--fix-refusals", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.strip_tags:
        cmd_strip_tags()
    elif args.fix_refusals:
        cmd_fix_refusals()
    elif args.merge:
        cmd_merge()
    else:
        parser.print_help()
