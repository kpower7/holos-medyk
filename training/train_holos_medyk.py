"""
Fine-tune Gemma 4 E4B on Holos Medyk training data using Unsloth.

Training format: <thinking>...</thinking><answer>...</answer>
The thinking traces come from Gemini 3 Pro — this is reasoning distillation.
At inference time we prompt the model to skip <thinking> and go straight to <answer>
for fast voice response on the Pi, while the reasoning capability is baked into weights.

Usage (on a GPU box, e.g. A100 on RunPod):
    pip install unsloth
    python train_holos_medyk.py

Expected runtime: ~1-2 hours on a single A100 40GB.
"""

import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# --- Config ---
MODEL_NAME = "google/gemma-4-E4b-it"
TRAIN_FILE = "data/training/train_holos_medyk.jsonl"
OUTPUT_DIR = "training/outputs/holos_medyk_gemma4_e4b"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4  # effective batch = 8
WARMUP_RATIO = 0.05
SEED = 42


def main():
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
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

    # Quick sanity check on the first formatted example
    print("\n=== First formatted training example ===")
    print(dataset[0]["text"][:1500])
    print("...")
    print()

    # Steps per epoch
    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = len(dataset) // effective_batch
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    print(f"Effective batch size: {effective_batch}")
    print(f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, Warmup: {warmup_steps}")

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
            num_train_epochs=NUM_EPOCHS,
            warmup_steps=warmup_steps,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",
            weight_decay=0.01,
            seed=SEED,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            report_to="none",
        ),
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving LoRA adapter...")
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")

    print("\nMerging LoRA into base model and saving...")
    model.save_pretrained_merged(
        f"{OUTPUT_DIR}/merged_16bit",
        tokenizer,
        save_method="merged_16bit",
    )

    print("\nExporting to GGUF (q4_k_m for Pi, f16 for quality reference)...")
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}/gguf_q4_k_m",
        tokenizer,
        quantization_method="q4_k_m",
    )
    model.save_pretrained_gguf(
        f"{OUTPUT_DIR}/gguf_f16",
        tokenizer,
        quantization_method="f16",
    )

    print(f"\n=== DONE ===")
    print(f"LoRA adapter: {OUTPUT_DIR}/lora_adapter")
    print(f"Merged 16bit: {OUTPUT_DIR}/merged_16bit")
    print(f"GGUF q4_k_m (Pi deploy): {OUTPUT_DIR}/gguf_q4_k_m")
    print(f"GGUF f16 (quality ref):  {OUTPUT_DIR}/gguf_f16")


if __name__ == "__main__":
    main()
