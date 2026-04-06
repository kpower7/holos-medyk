from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch, os, json

MODEL_NAME = "google/gemma-4-E4b-it"
CHECKPOINT = "training/outputs/holos_medyk_gemma4_e4b/checkpoint-400"
OUTPUT_DIR = "training/outputs/holos_medyk_gemma4_e4b"
TRAIN_FILE = "data/training/train_holos_medyk.jsonl"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=2048,
    load_in_4bit=False, load_in_16bit=True, full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model, r=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=64, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", max_seq_length=2048,
)

dataset = load_dataset("json", data_files={"train": TRAIN_FILE}, split="train")
dataset = dataset.map(lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)}, remove_columns=dataset.column_names)

trainer = SFTTrainer(
    model=model, train_dataset=dataset, tokenizer=tokenizer,
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=2048,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        num_train_epochs=3, warmup_steps=65, learning_rate=2e-4,
        lr_scheduler_type="cosine", logging_steps=10, save_steps=200,
        save_total_limit=3, output_dir=OUTPUT_DIR, optim="adamw_8bit",
        weight_decay=0.01, seed=42,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    ),
)

print("Resuming from checkpoint-400...")
trainer.train(resume_from_checkpoint=CHECKPOINT)

print("Saving LoRA adapter...")
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")

print("Merging and saving...")
model.save_pretrained_merged(f"{OUTPUT_DIR}/merged_16bit", tokenizer, save_method="merged_16bit")

print("Exporting GGUF...")
model.save_pretrained_gguf(f"{OUTPUT_DIR}/gguf_q4_k_m", tokenizer, quantization_method="q4_k_m")

with open(f"{OUTPUT_DIR}/training_log.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

print("DONE")
