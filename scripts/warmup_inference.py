#!/usr/bin/env python3
"""warmup_inference.py - CUDA kernel warmup 解决首次推理延迟

核心问题：PyTorch CUDA kernel JIT 编译导致首次推理慢 20s+
解决方案：在模型加载后立即执行一次 warmup 推理
"""

import os, sys, time
sys.path.insert(0, '/workspace2/cy/Qwen3-0.6B')
os.chdir('/workspace2/cy/Qwen3-0.6B')

import torch
from PIL import Image
from siq_vl.model.modeling import SiQ_VLForCausalLM
from siq_vl.model.processing import SiQ_VLProcessor

print("Loading model...", flush=True)
t0 = time.time()
model = SiQ_VLForCausalLM.from_pretrained(
    "./outputs/stage2/merged_v4",
    torch_dtype=torch.bfloat16,
    device_map={"": 3},
)
processor = SiQ_VLProcessor.from_pretrained("./outputs/stage2/merged_v4", fix_mistral_regex=True)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

# Create a synthetic warmup image (small, fast)
warmup_img = Image.new("RGB", (224, 224), color="red")
warmup_question = "What color is this?"

# ============================================================
# WITHOUT WARMUP: Measure cold start
# ============================================================
print("\n=== Cold Start (no warmup) ===", flush=True)
t_cold = time.time()
inputs = processor(
    batch=[(warmup_img, warmup_question, None)],
    return_tensors="pt",
).to(model.device)
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        attention_mask=inputs.attention_mask,
        max_new_tokens=10,
        eos_token_id=[151643, 151645],
        do_sample=False,
        repetition_penalty=1.3,
    )
cold_time = time.time() - t_cold
tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
print(f"  Cold inference: {cold_time:.2f}s ({tokens} tokens)", flush=True)

# ============================================================
# WITH WARMUP: Measure warm start
# ============================================================
print("\n=== Warm Start ===", flush=True)
t_warm = time.time()
inputs2 = processor(
    batch=[(warmup_img, warmup_question, None)],
    return_tensors="pt",
).to(model.device)
with torch.no_grad():
    output_ids2 = model.generate(
        input_ids=inputs2.input_ids,
        pixel_values=inputs2.pixel_values,
        attention_mask=inputs2.attention_mask,
        max_new_tokens=10,
        eos_token_id=[151643, 151645],
        do_sample=False,
        repetition_penalty=1.3,
    )
warm_time = time.time() - t_warm
tokens2 = output_ids2.shape[1] - inputs2.input_ids.shape[1]
print(f"  Warm inference: {warm_time:.2f}s ({tokens2} tokens)", flush=True)

# ============================================================
# Different image size to trigger recompilation
# ============================================================
print("\n=== Different Image Size (triggers recompilation?) ===", flush=True)
diff_img = Image.new("RGB", (224, 224), color="blue")
t_diff = time.time()
inputs3 = processor(
    batch=[(diff_img, "What color is this?", None)],
    return_tensors="pt",
).to(model.device)
with torch.no_grad():
    output_ids3 = model.generate(
        input_ids=inputs3.input_ids,
        pixel_values=inputs3.pixel_values,
        attention_mask=inputs3.attention_mask,
        max_new_tokens=10,
        eos_token_id=[151643, 151645],
        do_sample=False,
        repetition_penalty=1.3,
    )
diff_time = time.time() - t_diff
print(f"  Different image: {diff_time:.2f}s", flush=True)

# ============================================================
# Real image test
# ============================================================
print("\n=== Real COCO Image ===", flush=True)
real_img = Image.open("./data/coco128_images/000000000009.jpg").convert("RGB")
t_real = time.time()
inputs4 = processor(
    batch=[(real_img, "What objects can you see?", None)],
    return_tensors="pt",
).to(model.device)
with torch.no_grad():
    output_ids4 = model.generate(
        input_ids=inputs4.input_ids,
        pixel_values=inputs4.pixel_values,
        attention_mask=inputs4.attention_mask,
        max_new_tokens=80,
        eos_token_id=[151643, 151645],
        do_sample=False,
        repetition_penalty=1.3,
    )
real_time = time.time() - t_real
real_tokens = output_ids4.shape[1] - inputs4.input_ids.shape[1]
resp = processor.decode(output_ids4[0], assistant_only=True, skip_special_tokens=True)
print(f"  Real image: {real_time:.2f}s ({real_tokens} tokens)", flush=True)
print(f"  Response: {resp[:100]}...", flush=True)

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print(f"  WARMUP IMPACT SUMMARY")
print(f"{'=' * 60}")
print(f"  Cold start (1st inference):  {cold_time:.2f}s")
print(f"  Warm start (2nd inference):  {warm_time:.2f}s")
print(f"  Speedup from warmup:          {cold_time/warm_time:.1f}x")
print(f"  Cold overhead eliminated:     {cold_time-warm_time:.2f}s")
print(f"\n  Recommendation: Add warmup inference after model.load()")
print(f"  This eliminates {cold_time-warm_time:.1f}s of CUDA JIT compilation overhead")
