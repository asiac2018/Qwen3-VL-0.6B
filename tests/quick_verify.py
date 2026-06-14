#!/usr/bin/env python3
"""quick_verify.py - Quick verification of the modified SiQ-VL code

Tests the code modifications without requiring full model downloads.
Uses CPU and small tensors for fast verification.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
from PIL import Image


def test_configuration():
    """Test configuration.py modifications."""
    print("=" * 60)
    print("Test 1: Configuration (Qwen3Config compatibility)")
    print("=" * 60)

    from siq_vl.model.configuration import SiQ_VLTextConfig, SiQ_VLConfig, get_siq_vl_config
    from transformers import Qwen3Config

    # Test 1a: SiQ_VLTextConfig inherits from Qwen3Config
    assert issubclass(SiQ_VLTextConfig, Qwen3Config), "SiQ_VLTextConfig must inherit from Qwen3Config"
    print("  ✅ SiQ_VLTextConfig inherits from Qwen3Config")

    # Test 1b: Default projector config matches Qwen3-0.6B
    from siq_vl.model.configuration import SiQ_VLProjectorConfig
    proj_config = SiQ_VLProjectorConfig()
    assert proj_config.text_hidden_size == 1024, f"Expected 1024, got {proj_config.text_hidden_size}"
    assert proj_config.vision_hidden_size == 1152, f"Expected 1152, got {proj_config.vision_hidden_size}"
    print(f"  ✅ Projector defaults: vision_hidden={proj_config.vision_hidden_size}, text_hidden={proj_config.text_hidden_size}")

    # Test 1c: Config generation with local Qwen3-0.6B
    config = get_siq_vl_config(
        text_model_name_or_path='/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B',
        vision_model_name_or_path='google/siglip2-so400m-patch14-224',
        vision_pixel_shuffle_factor=2,
    )
    assert config.text_config.hidden_size == 1024
    assert config.vision_config.hidden_size == 1152
    assert config.projector_config.vision_pixel_shuffle_factor == 2
    print(f"  ✅ Full config created: text={config.text_config.hidden_size}, vision={config.vision_config.hidden_size}")

    # Test 1d: Verify Qwen3-specific attributes
    assert hasattr(config.text_config, 'layer_types'), "Qwen3Config must have layer_types"
    assert len(config.text_config.layer_types) == 28, f"Expected 28 layers, got {len(config.text_config.layer_types)}"
    print(f"  ✅ Qwen3 layer_types: {len(config.text_config.layer_types)} layers")

    return True


def test_modeling():
    """Test modeling.py modifications."""
    print("\n" + "=" * 60)
    print("Test 2: Modeling (Qwen3ForCausalLM + 2-Layer Projector)")
    print("=" * 60)

    from siq_vl.model.modeling import SiQ_VLTextModel, SiQ_VLProjector
    from transformers import Qwen3ForCausalLM

    # Test 2a: SiQ_VLTextModel inherits from Qwen3ForCausalLM
    assert issubclass(SiQ_VLTextModel, Qwen3ForCausalLM), "SiQ_VLTextModel must inherit from Qwen3ForCausalLM"
    print("  ✅ SiQ_VLTextModel inherits from Qwen3ForCausalLM")

    # Test 2b: Projector has 2-layer MLP structure
    from siq_vl.model.configuration import SiQ_VLProjectorConfig
    proj_config = SiQ_VLProjectorConfig(
        vision_hidden_size=1152,
        text_hidden_size=1024,
        vision_pixel_shuffle_factor=2,
    )
    projector = SiQ_VLProjector(proj_config)

    assert hasattr(projector, 'linear1'), "Projector must have linear1"
    assert hasattr(projector, 'linear2'), "Projector must have linear2"
    assert hasattr(projector, 'act'), "Projector must have activation (act)"
    assert hasattr(projector, 'norm'), "Projector must have LayerNorm (norm)"
    print(f"  ✅ Projector structure: linear1={projector.linear1}, linear2={projector.linear2}")

    # Test 2c: Projector forward pass
    # SigLIP2 output: (B, 256, 1152) for 224x224 image with patch_size=14
    dummy_vision_features = torch.randn(1, 256, 1152)
    output = projector(dummy_vision_features)
    assert output.shape == (1, 64, 1024), f"Expected (1, 64, 1024), got {output.shape}"
    print(f"  ✅ Projector forward: input (1, 256, 1152) → output {output.shape}")

    # Test 2d: Projector parameter count
    total_params = sum(p.numel() for p in projector.parameters())
    print(f"  ✅ Projector params: {total_params:,} (~{total_params/1e6:.1f}M)")

    # Test 2e: unfreez_text_model uses train() not eval()
    import inspect
    from siq_vl.model.modeling import SiQ_VLForCausalLM
    source = inspect.getsource(SiQ_VLForCausalLM.unfreez_text_model)
    assert '.train()' in source, "unfreez_text_model must call .train() not .eval()"
    assert '.eval()' not in source, "unfreez_text_model must NOT call .eval()"
    print("  ✅ unfreez_text_model uses .train() (not .eval())")

    return True


def test_processing():
    """Test processing.py modifications."""
    print("\n" + "=" * 60)
    print("Test 3: Processing (System prompt + Tokenizer)")
    print("=" * 60)

    from siq_vl.model.processing import DEFAULT_SYSTEM_PROMPT

    # Test 3a: System prompt doesn't mention Qwen2
    assert "Qwen2" not in DEFAULT_SYSTEM_PROMPT, "System prompt should not mention Qwen2"
    assert "SiQ-VL" not in DEFAULT_SYSTEM_PROMPT, "System prompt should not mention SiQ-VL"
    print(f"  ✅ System prompt: '{DEFAULT_SYSTEM_PROMPT[:60]}...'")

    # Test 3b: Processor can be created with Qwen3-0.6B tokenizer
    from transformers import AutoTokenizer
    from siq_vl.model.processing import SiQ_VLProcessor

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
    print(f"  ✅ Processor created with Qwen3-0.6B tokenizer")

    # Test 3c: Processor correctly handles vision tokens
    dummy_img = Image.new("RGB", (224, 224), color="red")
    inputs = processor(
        batch=[(dummy_img, "What color is this?", None)],
        return_tensors="pt",
    )
    assert inputs.input_ids.shape[0] == 1, "Batch size should be 1"
    assert inputs.pixel_values.shape[0] == 1, "Should have 1 image"
    assert inputs.pixel_values.shape[2:] == (224, 224), f"Image size should be 224x224, got {inputs.pixel_values.shape[2:]}"
    print(f"  ✅ Processor output: input_ids={inputs.input_ids.shape}, pixel_values={inputs.pixel_values.shape}")

    # Test 3d: Verify image_pad token in input_ids
    image_pad_count = (inputs.input_ids == 151655).sum().item()
    assert image_pad_count > 0, "input_ids should contain <|image_pad|> tokens"
    print(f"  ✅ Image pad tokens in input_ids: {image_pad_count} (expected: 64)")

    return True


def test_eos_token():
    """Test EOS token handling."""
    print("\n" + "=" * 60)
    print("Test 4: EOS Token Handling")
    print("=" * 60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        '/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B'
    )

    # Test 4a: Correct eos_token_id
    assert tokenizer.eos_token_id == 151645, f"eos_token_id should be 151645, got {tokenizer.eos_token_id}"
    print(f"  ✅ eos_token_id = {tokenizer.eos_token_id} (<|im_end|>)")

    # Test 4b: Decode verification
    assert tokenizer.decode(151645) == '<|im_end|>', "Token 151645 should decode to <|im_end|>"
    print(f"  ✅ Token 151645 decodes to '<|im_end|>'")

    # Test 4c: Vision tokens present
    vision_tokens = {
        '<|vision_start|>': 151652,
        '<|vision_end|>': 151653,
        '<|vision_pad|>': 151654,
        '<|image_pad|>': 151655,
    }
    for token, expected_id in vision_tokens.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        assert actual_id == expected_id, f"{token} should be {expected_id}, got {actual_id}"
    print(f"  ✅ All vision tokens present in tokenizer")

    return True


def test_qwen3_weight_compatibility():
    """Test that Qwen3-0.6B weights are compatible with Qwen3ForCausalLM."""
    print("\n" + "=" * 60)
    print("Test 5: Qwen3-0.6B Weight Compatibility")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, Qwen3ForCausalLM

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        '/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B',
        torch_dtype=torch.bfloat16,
        device_map='cpu',
    )

    # Test 5a: Model is Qwen3ForCausalLM
    assert isinstance(model, Qwen3ForCausalLM), f"Model should be Qwen3ForCausalLM, got {type(model)}"
    print(f"  ✅ Model type: {type(model).__name__}")

    # Test 5b: Has q_norm/k_norm (Qwen3-specific)
    q_norm_keys = [k for k in model.state_dict().keys() if 'q_norm' in k or 'k_norm' in k]
    assert len(q_norm_keys) > 0, "Qwen3 model must have q_norm/k_norm weights"
    print(f"  ✅ q_norm/k_norm keys: {len(q_norm_keys)} (Qwen3-specific)")

    # Test 5c: No attention bias
    bias_keys = [k for k in model.state_dict().keys() if 'self_attn' in k and 'bias' in k]
    assert len(bias_keys) == 0, f"Qwen3 should have no attention bias, found {bias_keys}"
    print(f"  ✅ No attention bias (Qwen3 has q_norm/k_norm instead)")

    # Test 5d: Forward pass works
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    assert outputs.logits.shape[-1] == 151936, f"Vocab size should be 151936"
    print(f"  ✅ Forward pass OK, logits shape: {outputs.logits.shape}")

    return True


def main():
    print("\n" + "🔬 " * 20)
    print("  SiQ-VL v2 Quick Verification Suite")
    print("🔬 " * 20 + "\n")

    results = {}

    tests = [
        ("Configuration", test_configuration),
        ("Modeling", test_modeling),
        ("Processing", test_processing),
        ("EOS Token", test_eos_token),
        ("Qwen3 Weight Compatibility", test_qwen3_weight_compatibility),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = "✅ PASS"
        except Exception as e:
            results[name] = f"❌ FAIL: {e}"
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("  Verification Summary")
    print("=" * 60)
    for name, result in results.items():
        print(f"  {name}: {result}")

    all_passed = all(r == "✅ PASS" for r in results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The code modifications are correct.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

    return all_passed


if __name__ == "__main__":
    main()
