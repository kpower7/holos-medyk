"""
Holos Medyk — Gemma 4 Inference via llama.cpp
Wraps llama-cli for easy testing of medical prompts in Ukrainian.

Usage:
    python scripts/inference.py "Your prompt here"
    python scripts/inference.py --file prompts/test.txt
    python scripts/inference.py --model e2b "Your prompt here"
    python scripts/inference.py --interactive
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LLAMA_CLI = ROOT / "tools" / "llama-cpp" / "llama-cli.exe"
MODELS = {
    "e4b": ROOT / "models" / "google_gemma-4-E4B-it-Q3_K_M.gguf",
    "e2b": ROOT / "models" / "google_gemma-4-E2B-it-Q4_K_M.gguf",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are Holos Medyk, an emergency medical assistant for Ukrainian civilians. "
    "Give clear step-by-step first aid instructions in Ukrainian. "
    "Be concise and direct."
)


def run_inference(
    prompt: str = None,
    prompt_file: str = None,
    model: str = "e4b",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    n_gpu_layers: int = 99,
    ctx_size: int = 2048,
    max_tokens: int = 1024,
    interactive: bool = False,
):
    model_path = MODELS.get(model)
    if model_path is None or not model_path.exists():
        print(f"Model not found: {model_path}")
        print(f"Available models: {', '.join(k for k, v in MODELS.items() if v.exists())}")
        sys.exit(1)

    if not LLAMA_CLI.exists():
        print(f"llama-cli not found at: {LLAMA_CLI}")
        print("Download from: https://github.com/ggml-org/llama.cpp/releases")
        sys.exit(1)

    cmd = [
        str(LLAMA_CLI),
        "-m", str(model_path),
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx_size),
        "-n", str(max_tokens),
        "-sys", system_prompt,
    ]

    if not interactive:
        cmd.append("--single-turn")

    if prompt_file:
        cmd.extend(["-f", prompt_file])
    elif prompt:
        cmd.extend(["-p", prompt])

    print(f"Model: {model_path.name}")
    print(f"GPU layers: {n_gpu_layers} | Context: {ctx_size} | Max tokens: {max_tokens}")
    print("-" * 60)

    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Holos Medyk — Gemma 4 inference")
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt text")
    parser.add_argument("--file", "-f", default=None, help="Read prompt from file")
    parser.add_argument("--model", "-m", default="e4b", choices=MODELS.keys(),
                        help="Model to use (default: e4b)")
    parser.add_argument("--system", "-s", default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt")
    parser.add_argument("--gpu-layers", "-ngl", type=int, default=99,
                        help="GPU layers to offload (default: 99 = all)")
    parser.add_argument("--ctx", "-c", type=int, default=2048, help="Context size")
    parser.add_argument("--max-tokens", "-n", type=int, default=1024, help="Max tokens")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive multi-turn chat mode")
    args = parser.parse_args()

    if not args.prompt and not args.file and not args.interactive:
        parser.print_help()
        sys.exit(1)

    run_inference(
        prompt=args.prompt,
        prompt_file=args.file,
        model=args.model,
        system_prompt=args.system,
        n_gpu_layers=args.gpu_layers,
        ctx_size=args.ctx,
        max_tokens=args.max_tokens,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
