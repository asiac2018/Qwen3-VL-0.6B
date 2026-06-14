#!/usr/bin/env python3
"""stage1_train.py - Projector Pre-training (v3 - Local Data Support)

Trains the 2-Layer MLP Projector to align SigLIP2 visual features
with Qwen3-0.6B's embedding space.

Supports both:
- JSON format: list of {"image": "path", "conversations": [...]}
- HuggingFace dataset format

Uses proper data collation and gradient checkpointing for H20 GPU.
"""

import os
import sys
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from PIL import Image

from siq_vl.model.modeling import get_stage1_model_and_processor
from siq_vl.model.processing import SiQ_VLProcessor

# ============================================================
# Configuration
# ============================================================
VISION_MODEL_PATH = "./models/siglip2-so400m-patch14-224"
TEXT_MODEL_PATH = "/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B"
OUTPUT_DIR = "./outputs/stage1"

# Training hyperparameters (optimized for H20 95GB VRAM)
TRAINING_CONFIG = {
    "per_device_train_batch_size": 32,       # H20 can handle large batch for projector-only training
    "gradient_accumulation_steps": 2,        # effective batch_size = 32 × 2 = 64
    "learning_rate": 1e-3,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "max_steps": 3000,                        # Enough for ~1000 samples repeated
    "bf16": True,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "optim": "adamw_torch_fused",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_steps": 500,
    "logging_steps": 10,
    "save_total_limit": 3,
    "remove_unused_columns": False,
    "report_to": "none",
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
}


# ============================================================
# Dataset
# ============================================================
class VLMJsonDataset(Dataset):
    """Vision-Language dataset from JSON format.

    Expected format: list of dicts with 'image' (path) and 'conversations' keys.
    conversations: [{"from": "human", "value": "<image>\\nQuestion"}, {"from": "gpt", "value": "Answer"}]
    """

    def __init__(self, json_path: str, processor: SiQ_VLProcessor):
        self.processor = processor
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        print(f"  Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_path = item.get("image", None)
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), color="black")
        else:
            image = Image.new("RGB", (224, 224), color="black")

        # Parse conversations
        conversations = item.get("conversations", [])
        question = "Describe this image."
        answer = None

        if conversations:
            # Human message
            human_msg = conversations[0].get("value", "Describe this image.")
            question = human_msg.replace("<image>", "").strip()
            if not question:
                question = "Describe this image."

            # GPT response
            if len(conversations) > 1:
                answer = conversations[1].get("value", None)

        return image, question, answer


class VLMCollator:
    """Custom collation function that processes batches through SiQ_VLProcessor."""

    def __init__(self, processor: SiQ_VLProcessor):
        self.processor = processor

    def __call__(self, batch):
        """Process a batch of (image, question, answer) tuples."""
        return self.processor(batch=batch, return_tensors="pt")


# ============================================================
# Training
# ============================================================
def train():
    print("=" * 80)
    print("Stage 1: Projector Pre-training (v3 - 2-Layer MLP)")
    print("=" * 80)

    # 1. Initialize model and processor
    print("\n[1/5] Initializing model and processor...")
    model, processor = get_stage1_model_and_processor(
        pretrained_vision_model_path=VISION_MODEL_PATH,
        pretrained_text_model_path=TEXT_MODEL_PATH,
        vision_pixel_shuffle_factor=2,
        enable_dynamic_tiling=False,
    )

    # Fix eos_token_id for correct generation stopping
    model.config.eos_token_id = [151643, 151645]

    # 2. Load dataset
    print("\n[2/5] Loading dataset...")
    data_path = "./data/stage1/train.json"
    if not os.path.exists(data_path):
        print(f"  Warning: {data_path} not found, using dummy data")
        data_path = None

    if data_path:
        dataset = VLMJsonDataset(data_path, processor)
    else:
        # Fallback to dummy dataset
        from datasets import Dataset as HFDataset
        dummy_data = {
            "image": [Image.new("RGB", (224, 224), color=c) for c in ["red", "blue", "green", "yellow"] * 250],
            "conversations": [
                [{"from": "human", "value": "<image>\nDescribe this image."},
                 {"from": "gpt", "value": f"This is a {c} image."}]
                for c in ["red", "blue", "green", "yellow"] * 250
            ],
        }
        dataset = VLMJsonDataset.__new__(VLMJsonDataset)
        dataset.data = dummy_data
        dataset.processor = processor

    print(f"  Dataset size: {len(dataset)}")

    # 3. Create collator
    collator = VLMCollator(processor)

    # 4. Configure training
    print("\n[3/5] Configuring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        **TRAINING_CONFIG,
    )

    # 5. Train
    print("\n[4/5] Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    # 6. Save
    print("\n[5/5] Saving model...")
    save_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"  Model saved to {save_path}")

    # Quick test
    print("\n[Quick Test]")
    model.eval()
    test_img = Image.new("RGB", (224, 224), color="red")
    inputs = processor(
        batch=[(test_img, "What color is this image?", None)],
        return_tensors="pt",
    ).to(model.device if hasattr(model, 'device') else 'cuda')

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            attention_mask=inputs.attention_mask,
            max_new_tokens=64,
            eos_token_id=[151643, 151645],
            do_sample=False,
            temperature=0.0,
        )
    response = processor.decode(output_ids[0], assistant_only=True, skip_special_tokens=True)
    print(f"  Test response: {response}")

    print("\n✅ Stage 1 training complete!")


if __name__ == "__main__":
    train()
