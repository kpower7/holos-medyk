"""
Holos Medyk — Model Evaluation Benchmark

Runs the eval scenarios against a GGUF model via llama.cpp and saves responses
for human review. Run BEFORE and AFTER fine-tuning to compare.

Usage:
    # Baseline (before fine-tuning)
    python evaluation/run_eval.py --model models/google_gemma-4-E4B-it-Q3_K_M.gguf --tag baseline

    # Fine-tuned
    python evaluation/run_eval.py --model models/holos_medyk_q4_k_m.gguf --tag finetuned

    # Compare
    python evaluation/run_eval.py --compare evaluation/results/baseline.jsonl evaluation/results/finetuned.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
LLAMA_CLI = ROOT / "tools" / "llama-cpp" / "llama-cli.exe"
EVAL_FILE = ROOT / "evaluation" / "eval_scenarios.jsonl"
RESULTS_DIR = ROOT / "evaluation" / "results"

SYSTEM_PROMPT = """You are Holos Medyk, a Ukrainian warzone medical voice assistant. A civilian is speaking to you during or after a bombardment. Give clear, calm, actionable first aid guidance a non-medical person can follow right now. Be direct. No filler. Match the language of the user. Think through the situation inside <thinking> tags, then give your final instructions inside <answer> tags."""


def run_inference(model_path: Path, prompt: str, system_prompt: str) -> tuple[str, float]:
    """Run a single inference via llama-cli and return (response, seconds)."""
    # Write prompt to temp file to avoid encoding issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]")
        prompt_file = f.name

    try:
        start = time.time()
        result = subprocess.run(
            [
                str(LLAMA_CLI),
                "-m", str(model_path),
                "-ngl", "99",
                "-f", prompt_file,
                "-n", "512",
                "--temp", "0.3",
                "--no-cnv",
                "--single-turn",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.time() - start
        return result.stdout.strip(), elapsed
    except subprocess.TimeoutExpired:
        return "[TIMEOUT after 120s]", 120.0
    except Exception as e:
        return f"[ERROR: {e}]", 0.0
    finally:
        os.unlink(prompt_file)


def run_eval(model_path: Path, tag: str):
    """Run all eval scenarios and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{tag}.jsonl"

    scenarios = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            scenarios.append(json.loads(line))

    print(f"Model: {model_path}")
    print(f"Tag: {tag}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Output: {output_path}")
    print("=" * 70)

    results = []
    for i, scenario in enumerate(scenarios):
        print(f"\n[{i+1}/{len(scenarios)}] {scenario['id']} ({scenario['category']}, {scenario['language']})")
        print(f"  Prompt: {scenario['prompt'][:80]}...")

        response, elapsed = run_inference(model_path, scenario["prompt"], SYSTEM_PROMPT)

        result = {
            "id": scenario["id"],
            "category": scenario["category"],
            "language": scenario["language"],
            "prompt": scenario["prompt"],
            "key_criteria": scenario["key_criteria"],
            "response": response,
            "elapsed_seconds": round(elapsed, 1),
            "model": str(model_path.name),
            "tag": tag,
        }
        results.append(result)

        # Print summary
        resp_preview = response[:200].replace("\n", " ")
        print(f"  Response ({elapsed:.1f}s): {resp_preview}...")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_path}")
    print(f"Average response time: {sum(r['elapsed_seconds'] for r in results) / len(results):.1f}s")


def compare_results(file_a: Path, file_b: Path):
    """Print side-by-side comparison of two eval runs."""
    results_a = {}
    results_b = {}

    with open(file_a, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            results_a[r["id"]] = r

    with open(file_b, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            results_b[r["id"]] = r

    tag_a = list(results_a.values())[0]["tag"]
    tag_b = list(results_b.values())[0]["tag"]

    print(f"Comparing: {tag_a} vs {tag_b}")
    print(f"{'=' * 70}\n")

    for scenario_id in results_a:
        a = results_a[scenario_id]
        b = results_b.get(scenario_id)
        if not b:
            continue

        print(f"--- {a['id']} ({a['category']}, {a['language']}) ---")
        print(f"Prompt: {a['prompt'][:100]}...")
        print(f"\nKey criteria: {', '.join(a['key_criteria'])}")
        print(f"\n[{tag_a}] ({a['elapsed_seconds']}s):")
        # Extract answer tag if present
        resp_a = a["response"]
        if "<answer>" in resp_a:
            resp_a = resp_a.split("<answer>")[-1].split("</answer>")[0].strip()
        print(f"  {resp_a[:500]}")
        print(f"\n[{tag_b}] ({b['elapsed_seconds']}s):")
        resp_b = b["response"]
        if "<answer>" in resp_b:
            resp_b = resp_b.split("<answer>")[-1].split("</answer>")[0].strip()
        print(f"  {resp_b[:500]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Holos Medyk model evaluation")
    parser.add_argument("--model", type=Path, help="Path to GGUF model")
    parser.add_argument("--tag", type=str, help="Tag for this run (e.g. 'baseline', 'finetuned')")
    parser.add_argument("--compare", nargs=2, type=Path, help="Compare two result files")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
    elif args.model and args.tag:
        run_eval(args.model, args.tag)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
