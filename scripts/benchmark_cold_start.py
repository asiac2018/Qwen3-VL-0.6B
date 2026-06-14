#!/usr/bin/env python3
"""benchmark_cold_start.py - 精确测量 Gradio 冷启动 vs 热启动的各阶段耗时

关键发现：第一次推理 25s，后续 2s。瓶颈在哪？
"""

import os, sys, time, json
sys.path.insert(0, '/workspace2/cy/Qwen3-0.6B')
os.chdir('/workspace2/cy/Qwen3-0.6B')

import torch
from PIL import Image
from siq_vl.model.modeling import SiQ_VLForCausalLM
from siq_vl.model.processing import SiQ_VLProcessor

print("Loading model...", flush=True)
t_load_start = time.time()
model = SiQ_VLForCausalLM.from_pretrained(
    "./outputs/stage2/merged_v4",
    torch_dtype=torch.bfloat16,
    device_map={"": 3},
)
processor = SiQ_VLProcessor.from_pretrained("./outputs/stage2/merged_v4", fix_mistral_regex=True)
model.eval()
print(f"Model loaded in {time.time()-t_load_start:.1f}s\n", flush=True)

img = Image.open("./data/coco128_images/000000000009.jpg").convert("RGB")
question = "What objects can you see in this image?"

# ============================================================
# Phase-by-phase measurement for COLD START (first inference)
# ============================================================
print("=" * 60)
print("  COLD START (1st inference)")
print("=" * 60)

t_total_start = time.time()

# Step 1: Processor call
t1 = time.time()
inputs = processor(
    batch=[(img, question, None)],
    return_tensors="pt",
).to(model.device)
t2 = time.time()
print(f"  [1] Processor:         {t2-t1:.3f}s", flush=True)

# Step 2: Vision forward
t3 = time.time()
with torch.no_grad():
    vision_outputs = model.vision_model(inputs.pixel_values)
    vision_features = vision_outputs.last_hidden_state
t4 = time.time()
print(f"  [2] Vision forward:    {t4-t3:.3f}s  (COLD)", flush=True)

# Step 3: Projector
t5 = time.time()
with torch.no_grad():
    vision_embeddings = model.projector(vision_features)
    vision_flat = vision_embeddings.view(-1, vision_embeddings.size(-1))
t6 = time.time()
print(f"  [3] Projector:         {t6-t5:.3f}s", flush=True)

# Step 4: Token embedding + replacement
t7 = time.time()
with torch.no_grad():
    token_embeddings = model.text_model.get_input_embeddings()(inputs.input_ids)
    vision_flat_on_device = vision_flat.to(token_embeddings.device)
    image_token_positions = (inputs.input_ids == model.config.image_token_index).nonzero(as_tuple=True)
    token_embeddings[image_token_positions] = vision_flat_on_device.to(dtype=token_embeddings.dtype)
t8 = time.time()
print(f"  [4] Embed+Replace:      {t8-t7:.3f}s", flush=True)

# Step 5: First text forward pass (this is the KV-cache initialization)
t9 = time.time()
with torch.no_grad():
    text_output = model.text_model(
        inputs_embeds=token_embeddings,
        attention_mask=inputs.attention_mask,
        use_cache=True,
    )
t10 = time.time()
print(f"  [5] Text 1st forward:  {t10-t9:.3f}s  (COLD - includes CUDA kernel compilation)", flush=True)

# Step 6: Generate with detailed timing
t_gen_start = time.time()
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        attention_mask=inputs.attention_mask,
        max_new_tokens=80,
        eos_token_id=[151643, 151645],
        do_sample=False,
        repetition_penalty=1.3,
    )
t_gen_end = time.time()
print(f"  [6] Full generate:      {t_gen_end-t_gen_start:.3f}s  (includes step 1-5 internally)", flush=True)

t_total_end = time.time()
print(f"\n  TOTAL cold start:       {t_total_end-t_total_start:.3f}s", flush=True)
print(f"  GENERATED tokens:       {output_ids.shape[1] - inputs.input_ids.shape[1]}", flush=True)
print(f"  Per-token (overall):     {(t_total_end-t_total_start)/(output_ids.shape[1]-inputs.input_ids.shape[1])*1000:.1f}ms", flush=True)

# ============================================================
# WARM START (2nd inference - same image)
# ============================================================
print(f"\n{'=' * 60}")
print("  WARM START (2nd inference - same image)")
print("=" * 60)

# Clear CUDA cache to see if it's a GPU cache effect
torch.cuda.empty_cache()

t_total_start2 = time.time()
t1 = time.time()
inputs2 = processor(
    batch=[(img, question, None)],
    return_tensors="pt",
).to(model.device)
t2 = time.time()
print(f"  [1] Processor:         {t2-t1:.3f}s", flush=True)

t3 = time.time()
with torch.no_grad():
    vision_outputs2 = model.vision_model(inputs2.pixel_values)
t4 = time.time()
print(f"  [2] Vision forward:    {t4-t3:.3f}s  (WARM)", flush=True)

t_gen_start2 = time.time()
with torch.no_grad():
    output_ids2 = model.generate(
        input_ids=inputs2.input_ids,
        pixel_values=inputs2.pixel_values,
        attention_mask=inputs2.attention_mask,
        max_new_tokens=80,
        eos_token_id=[151643, 151645],
        do_sample=False,
        repetition_penalty=1.3,
    )
t_gen_end2 = time.time()
print(f"  [6] Full generate:      {t_gen_end2-t_gen_start2:.3f}s  (WARM)", flush=True)

t_total_end2 = time.time()
tokens2 = output_ids2.shape[1] - inputs2.input_ids.shape[1]
print(f"\n  TOTAL warm start:       {t_total_end2-t_total_start2:.3f}s", flush=True)
print(f"  Per-token (warm):       {(t_gen_end2-t_gen_start2)/tokens2*1000:.1f}ms", flush=True)

# ============================================================
# CUDA kernel compilation check
# ============================================================
print(f"\n{'=' * 60}")
print("  CUDA COMPILATION CHECK")
print("=" * 60)
print(f"  Cold→Warm speedup:     {t_total_end-t_total_start:.1f}s → {t_total_end2-t_total_start2:.1f}s", flush=True)
print(f"  Speedup ratio:         {(t_total_end-t_total_start)/(t_total_end2-t_total_start2):.1f}x", flush=True)

# Check if CUDA graph / JIT compilation is happening
torch.cuda.synchronize()
t_sync = time.time()
with torch.no_grad():
    _ = model.vision_model(inputs2.pixel_values)
torch.cuda.synchronize()
t_sync2 = time.time()
print(f"  Vision w/ sync:         {t_sync2-t_sync:.3f}s", flush=True)

# ============================================================
# Third run (should be same as warm)
# ============================================================
t_gen_start3 = time.time()
with torch.no_grad():
    output_ids3 = model.generate(
        input_ids=inputs2.input_ids,
        pixel_values=inputs2.pixel_values,
        attention_mask=inputs2.attention_mask,
        max_new_tokens=80,
        eos_token_id=[151643, 151645],
        do_sample=False,
        repetition_penalty=1.3,
    )
t_gen_end3 = time.time()
print(f"\n  3rd generate:           {t_gen_end3-t_gen_start3:.3f}s", flush=True)

# Save results
results = {
    "cold_start_s": t_total_end - t_total_start,
    "warm_start_s": t_total_end2 - t_total_start2,
    "cold_per_token_ms": (t_total_end-t_total_start)/(output_ids.shape[1]-inputs.input_ids.shape[1])*1000,
    "warm_per_token_ms": (t_gen_end2-t_gen_start2)/tokens2*1000,
    "speedup": (t_total_end-t_total_start)/(t_total_end2-t_total_start2),
    "third_run_s": t_gen_end3 - t_gen_start3,
}
with open("/workspace2/cy/Qwen3-0.6B/outputs/cold_start_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to outputs/cold_start_benchmark.json")
