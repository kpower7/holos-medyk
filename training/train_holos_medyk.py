"""
Fine-tune Gemma 4 E4B on Holos Medyk training data using Unsloth.

Uses FastModel (not FastLanguageModel) with finetune_vision_layers=False
to ensure LoRA only targets the text decoder, not vision/audio encoders.

Usage (on a GPU box, e.g. A100 on RunPod):
    pip install unsloth
    python train_holos_medyk.py

Expected runtime: ~5-10 minutes on A100 with 266 examples.
"""

import os
import torch
from unsloth import FastModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# --- Config ---
MODEL_NAME = "unsloth/gemma-4-E4B-it"
TRAIN_FILE = "data/training/train_holos_medyk_v3_curated.jsonl"
OUTPUT_DIR = "training/outputs/holos_medyk_gemma4_e4b"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 8
LORA_ALPHA = 8
LEARNING_RATE = 2e-4
MAX_STEPS = 60
BATCH_SIZE = 2
GRAD_ACCUM = 4  # effective batch = 8
SEED = 3407


def main():
    # Setup Weights & Biases logging
    try:
        import wandb
        wandb.init(
            project="holos-medyk",
            name="gemma4-e4b-sft-v5-textonly",
            config={
                "model": MODEL_NAME,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "learning_rate": LEARNING_RATE,
                "max_steps": MAX_STEPS,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "max_seq_length": MAX_SEQ_LENGTH,
                "finetune_vision_layers": False,
                "finetune_language_layers": True,
            },
        )
        report_to = "wandb"
        print("Logging to Weights & Biases")
    except (ImportError, Exception):
        report_to = "none"
        print("wandb not available, logging to stdout only")

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        full_finetuning=False,
    )

    print("Applying LoRA (text layers only, vision/audio frozen)...")
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

    print(f"Loading dataset from {TRAIN_FILE}...")
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE}, split="train")
    print(f"  {len(dataset)} examples")

    # Apply Gemma 4 chat template to the messages field
    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # Quick sanity check
    print("\n=== First formatted training example ===")
    print(dataset[0]["text"][:1500])
    print("...\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            max_steps=MAX_STEPS,
            warmup_steps=5,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="linear",
            logging_steps=5,
            save_steps=30,
            save_total_limit=2,
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",
            weight_decay=0.001,
            seed=SEED,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to=report_to,
        ),
    )

    print(f"\nStarting training ({MAX_STEPS} steps)...")
    result = trainer.train()

    # Save metrics
    import json
    metrics_path = f"{OUTPUT_DIR}/training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "train_loss": result.training_loss,
            "train_runtime": result.metrics.get("train_runtime"),
            "train_samples_per_second": result.metrics.get("train_samples_per_second"),
            "total_steps": result.global_step,
            "max_steps": MAX_STEPS,
            "lora_rank": LORA_RANK,
            "learning_rate": LEARNING_RATE,
            "finetune_vision_layers": False,
        }, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

    log_path = f"{OUTPUT_DIR}/training_log.json"
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Full training log saved to {log_path}")

    print("\nSaving LoRA adapter...")
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")

    print("\nMerging LoRA into base model and saving...")
    model.save_pretrained_merged(
        f"{OUTPUT_DIR}/merged_16bit",
        tokenizer,
        save_method="merged_16bit",
    )

    print("\nExporting to GGUF (q4_k_m)...")
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}/gguf_q4_k_m",
        tokenizer,
        quantization_method="q4_k_m",
    )

    print(f"\n=== DONE ===")
    print(f"LoRA adapter: {OUTPUT_DIR}/lora_adapter")
    print(f"Merged 16bit: {OUTPUT_DIR}/merged_16bit")
    print(f"GGUF q4_k_m: {OUTPUT_DIR}/gguf_q4_k_m")


if __name__ == "__main__":
    main()
