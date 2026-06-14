#!/usr/bin/env python3
"""local_test.py - Test with locally available models only

Uses Qwen3-0.6B text model + randomly initialized SigLIP2 to verify
the complete pipeline works. No network download required.

This test verifies:
1. Model initialization with Qwen3-0.6B weights
2. Forward pass with vision features
3. Training step (loss computation + backward)
4. Generation
5. LoRA application for Stage 2
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from PIL import Image

from siq_vl.model.configuration import get_siq_vl_config
from siq_vl.model.modeling import SiQ_VLForCausalLM, SiQ_VLTextModel, get_stage1_model_and_processor
from siq_vl.model.processing import SiQ_VLProcessor
from transformers import AutoTokenizer


def test_full_pipeline():
    """Complete pipeline test with local resources."""
    print("=" * 70)
    print("  SiQ-VL v2 Local Pipeline Test (No Download Required)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"\n  Device: {device}, dtype: {dtype}")

    # 1. Create config
    print("\n[1/7] Creating model config...")
    config = get_siq_vl_config(
        text_model_name_or_path='/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B',
        vision_model_name_or_path='google/siglip2-so400m-patch14-224',
        vision_pixel_shuffle_factor=2,
    )

    # 2. Initialize model
    print("[2/7] Initializing model (with Qwen3-0.6B weights)...")
    model = SiQ_VLForCausalLM(config)
    model.text_model = SiQ_VLTextModel.from_pretrained(
        '/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B'
    )
    # Vision model will be randomly initialized (no SigLIP2 weights yet)
    # This is fine for testing the pipeline

    model = model.to(device=device, dtype=dtype)
    model.freez_vision_model()
    model.freez_text_model()

    # 3. Create processor
    print("[3/7] Creating processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        '/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B'
    )
    processor = SiQ_VLProcessor(
        tokenizer=tokenizer,
        vit_image_size=224,
        vit_patch_size=14,
        pixel_shuffle_factor=2,
        enable_dynamic_tiling=False,
    )

    # 4. Forward pass
    print("[4/7] Testing forward pass...")
    img = Image.new("RGB", (224, 224), color="red")
    inputs = processor(
        batch=[(img, "What color is this image?", None)],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values.to(dtype),
            attention_mask=inputs.attention_mask,
        )
    print(f"  ✅ Forward pass: logits shape = {outputs.logits.shape}")

    # 5. Training step
    print("[5/7] Testing training step...")
    inputs_train = processor(
        batch=[(img, "What color is this image?", "The image is red.")],
        return_tensors="pt",
    ).to(device)

    # Forward + backward (only projector is trainable)
    model.train()
    outputs = model(
        input_ids=inputs_train.input_ids,
        pixel_values=inputs_train.pixel_values.to(dtype),
        attention_mask=inputs_train.attention_mask,
        labels=inputs_train.labels,
    )
    loss = outputs.loss
    loss.backward()
    print(f"  ✅ Training step: loss = {loss.item():.4f}")

    # Check gradients flow to projector only
    proj_grads = sum(p.grad.numel() for p in model.projector.parameters() if p.grad is not None)
    text_grads = sum(p.grad.numel() for p in model.text_model.parameters() if p.grad is not None)
    vision_grads = sum(p.grad.numel() for p in model.vision_model.parameters() if p.grad is not None)
    print(f"  Projector gradients: {proj_grads:,}")
    print(f"  Text model gradients: {text_grads:,} (should be 0)")
    print(f"  Vision model gradients: {vision_grads:,} (should be 0)")
    assert text_grads == 0, "Text model should be frozen in Stage 1"
    assert vision_grads == 0, "Vision model should be frozen in Stage 1"
    assert proj_grads > 0, "Projector should have gradients in Stage 1"

    # Zero gradients for next test
    model.zero_grad()

    # 6. Generation
    print("[6/7] Testing generation...")
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values.to(dtype),
            attention_mask=inputs.attention_mask,
            max_new_tokens=30,
            eos_token_id=[151643, 151645],
            do_sample=False,
            temperature=0.0,
        )
    raw_response = processor.decode(output_ids[0], assistant_only=True, skip_special_tokens=True)
    print(f"  ✅ Generation: \"{raw_response[:80]}...\"")

    # 7. LoRA test (Stage 2 preparation)
    print("[7/7] Testing LoRA application...")
    from peft import LoraConfig, get_peft_model, TaskType

    model.unfreez_text_model()

    lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  ✅ LoRA applied: trainable {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Test LoRA forward + backward
    peft_model.train()
    outputs = peft_model(
        input_ids=inputs_train.input_ids,
        pixel_values=inputs_train.pixel_values.to(dtype),
        attention_mask=inputs_train.attention_mask,
        labels=inputs_train.labels,
    )
    lora_loss = outputs.loss
    lora_loss.backward()

    lora_grads = sum(p.grad.numel() for p in peft_model.parameters() if p.grad is not None and p.requires_grad)
    print(f"  ✅ LoRA training step: loss = {lora_loss.item():.4f}, trainable grads = {lora_grads:,}")

    # Summary
    print("\n" + "=" * 70)
    print("  🎉 ALL LOCAL TESTS PASSED!")
    print("=" * 70)
    print("""
  Pipeline verification complete. The code is ready for training.

  Next steps when network is available:
  1. Download SigLIP2 weights:
     python -c "from transformers import SiglipVisionModel; m = SiglipVisionModel.from_pretrained('google/siglip2-so400m-patch14-224'); m.save_pretrained('./models/siglip2-so400m-patch14-224')"

  2. Download training data:
     python -c "from datasets import load_dataset; ds = load_dataset('liuhaotian/LLaVA-Pretrain', split='train'); ds.save_to_disk('./data/stage1/llava_pretrain')"

  3. Run full training:
     CUDA_VISIBLE_DEVICES=3 python scripts/setup_and_train.py --skip-download
    """)


if __name__ == "__main__":
    test_full_pipeline()
