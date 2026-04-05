"""
Convert the 7 response JSONL files into a single Unsloth-ready training file.

Each training example becomes:
  {
    "messages": [
      {"role": "system", "content": "<persona-specific system prompt>"},
      {"role": "user", "content": "<prompt>"},
      {"role": "assistant", "content": "<thinking>...</thinking>\n<answer>...</answer>"}
    ]
  }

The `<thinking>` wrapper teaches the model to reason internally before answering.
At inference time we can strip `<thinking>...</thinking>` or use it to show reasoning.
"""

import json
from pathlib import Path

RESPONSES_DIR = Path("data/training/responses")
OUTPUT_FILE = Path("data/training/train_holos_medyk.jsonl")

SYSTEM_PROMPT_WARZONE = (
    "You are Holos Medyk, a Ukrainian warzone medical voice assistant. A civilian "
    "is speaking to you during or after a bombardment. Give clear, calm, actionable "
    "first aid guidance a non-medical person can follow right now. Be direct. No "
    "filler. Match the language of the user. Think through the situation inside "
    "<thinking> tags, then give your final instructions inside <answer> tags."
)

SYSTEM_PROMPT_MEDICAL = (
    "You are Holos Medyk, a medical assistant helping civilians with health questions. "
    "Give clear, evidence-based medical guidance in plain language. Use generic drug "
    "names. Never refuse to help. Match the language of the user. Think through the "
    "situation inside <thinking> tags, then give your final guidance inside <answer> tags."
)


def get_system_prompt(filename: str) -> str:
    return SYSTEM_PROMPT_WARZONE if filename.startswith("deepseek") else SYSTEM_PROMPT_MEDICAL


def format_assistant_turn(thinking: str, response: str) -> str:
    """Wrap thinking + answer in a consistent tag format."""
    if thinking:
        return f"<thinking>{thinking}</thinking>\n<answer>{response}</answer>"
    # No thinking available — still use the answer wrapper so format is consistent
    return f"<answer>{response}</answer>"


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    stats = {"total": 0, "with_thinking": 0, "warzone": 0, "medical": 0}

    for jsonl in sorted(RESPONSES_DIR.glob("*.jsonl")):
        system_prompt = get_system_prompt(jsonl.name)
        is_warzone = jsonl.name.startswith("deepseek")

        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                prompt = rec.get("prompt", "").strip()
                response = rec.get("response", "").strip()
                thinking = rec.get("thinking", "").strip()

                if not prompt or not response:
                    continue

                assistant_content = format_assistant_turn(thinking, response)

                example = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant_content},
                    ]
                }
                examples.append(example)

                stats["total"] += 1
                if thinking:
                    stats["with_thinking"] += 1
                if is_warzone:
                    stats["warzone"] += 1
                else:
                    stats["medical"] += 1

    # Shuffle so warzone and medical are mixed throughout training
    import random
    random.seed(42)
    random.shuffle(examples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {stats['total']} examples to {OUTPUT_FILE}")
    print(f"  Warzone: {stats['warzone']}")
    print(f"  Medical: {stats['medical']}")
    print(f"  With thinking traces: {stats['with_thinking']} ({100*stats['with_thinking']/stats['total']:.1f}%)")

    # Quick length stats
    lengths = []
    for ex in examples:
        total_len = sum(len(m["content"]) for m in ex["messages"])
        lengths.append(total_len)
    import statistics
    print(f"  Content length (chars): min={min(lengths)}, median={int(statistics.median(lengths))}, "
          f"p95={int(sorted(lengths)[int(0.95*len(lengths))])}, max={max(lengths)}")


if __name__ == "__main__":
    main()
