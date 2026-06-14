#!/usr/bin/env python3
"""stage2_train.py - Visual Instruction Tuning (v3 - Local Data Support)

Fine-tunes the Projector + LLM (with LoRA) for visual instruction following.

Key parameters:
- LoRA r=128, alpha=128 (effective scaling ratio alpha/r=1.0)
- Higher learning rate (5e-5) justified by higher LoRA alpha
- 2 training epochs
- Supports JSON format training data
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from transformers import TrainingArguments, Trainer

from siq_vl.model.modeling import get_stage2_model_and_processor
from scripts.stage1_train import VLMJsonDataset, VLMCollator

# ============================================================
# Configuration
# ============================================================
STAGE1_CHECKPOINT = "./outputs/stage1/final"
OUTPUT_DIR = "./outputs/stage2"

TRAINING_CONFIG = {
    "per_device_train_batch_size": 8,       # Stage 2 uses LoRA, more memory needed
    "gradient_accumulation_steps": 4,       # effective batch_size = 8 × 4 = 32
    "learning_rate": 2e-5,                  # Lower LR for more stable fine-tuning
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "num_train_epochs": 5,                  # More epochs for smaller, higher-quality dataset
    "bf16": True,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "optim": "adamw_torch_fused",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_steps": 200,
    "logging_steps": 10,
    "save_total_limit": 3,
    "remove_unused_columns": False,
    "report_to": "none",
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
}


# ============================================================
# Training
# ============================================================
def train():
    print("=" * 80)
    print("Stage 2: Visual Instruction Tuning (v3 - LoRA r=128)")
    print("=" * 80)

    # 1. Load Stage 1 model
    print("\n[1/4] Loading Stage 1 checkpoint...")
    model, processor = get_stage2_model_and_processor(
        stage_1_checkpoint_path=STAGE1_CHECKPOINT,
        use_lora=True,
        lora_r=128,
        lora_alpha=128,
        lora_dropout=0.05,
    )

    # Fix eos_token_id
    model.config.eos_token_id = [151643, 151645]

    # 2. Load dataset
    print("\n[2/4] Loading instruction dataset...")
    data_path = "./data/stage2/train_v4.json"
    if not os.path.exists(data_path):
        print(f"  Warning: {data_path} not found, falling back to v3 data")
        data_path = "./data/stage2/train_v3.json"
    if not os.path.exists(data_path):
        print(f"  Warning: {data_path} not found, falling back to v2 data")
        data_path = "./data/stage2/train_v2.json"

    dataset = VLMJsonDataset(data_path, processor)
    print(f"  Dataset size: {len(dataset)}")

    collator = VLMCollator(processor)

    # 3. Configure training
    print("\n[3/4] Configuring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **TRAINING_CONFIG,
    )

    # 4. Train
    print("\n[4/4] Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    # Save - merge LoRA first for clean inference
    save_path = os.path.join(OUTPUT_DIR, "final")
    print("\n[5/5] Merging LoRA and saving model...")

    # Merge LoRA weights into base model for clean inference
    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(save_path)
        print("  LoRA merged and saved successfully!")
    except Exception as e:
        print(f"  Warning: Could not merge LoRA ({e}), saving as-is")
        model.save_pretrained(save_path)

    processor.save_pretrained(save_path)
    print(f"  Model saved to {save_path}")

    print("\n✅ Stage 2 training complete!")


if __name__ == "__main__":
    train()
