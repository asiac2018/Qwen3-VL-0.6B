#!/usr/bin/env python3
"""benchmark_inference.py - 精确测量推理各阶段耗时

目标：找出20s推理延迟的瓶颈所在
"""

import os, sys, time
sys.path.insert(0, '/workspace2/cy/Qwen3-0.6B')
os.chdir('/workspace2/cy/Qwen3-0.6B')

import torch
from PIL import Image
from siq_vl.model.modeling import SiQ_VLForCausalLM
from siq_vl.model.processing import SiQ_VLProcessor

# Load model
print("Loading model...", flush=True)
t0 = time.time()
model = SiQ_VLForCausalLM.from_pretrained(
    "./outputs/stage2/merged_v4",
    torch_dtype=torch.bfloat16,
    device_map={"": 3},
)
processor = SiQ_VLProcessor.from_pretrained("./outputs/stage2/merged_v4", fix_mistral_regex=True)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s, device: {next(model.parameters()).device}", flush=True)

# Test image
img = Image.open("./data/coco128_images/000000000009.jpg").convert("RGB")
question = "What objects can you see in this image?"

# ============================================================
# Phase 1: Image preprocessing
# ============================================================
t1 = time.time()
# This is what processor does internally - just the image part
from siq_vl.model.processing import SiQ_VLImageProcessor
image_processor_obj = processor.image_processor
pixel_values = image_processor_obj([img], return_tensors="pt")["pixel_values"]
t2 = time.time()
print(f"\n[1] Image preprocessing: {t2-t1:.3f}s", flush=True)
print(f"    pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}", flush=True)

# ============================================================
# Phase 2: Text tokenization
# ============================================================
t3 = time.time()
tokenizer = processor.tokenizer
# Build the chat template
system_prompt = processor.system_prompt
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"<|vision_start|><|image_pad|><|vision_end|>{question}"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
t4 = time.time()
print(f"\n[2] Text tokenization: {t4-t3:.3f}s", flush=True)
print(f"    input_ids shape: {input_ids.shape}, tokens: {input_ids.shape[1]}", flush=True)

# ============================================================
# Phase 3: Full processor call
# ============================================================
t5 = time.time()
inputs = processor(
    batch=[(img, question, None)],
    return_tensors="pt",
).to(model.device)
t6 = time.time()
print(f"\n[3] Full processor call: {t6-t5:.3f}s", flush=True)
print(f"    input_ids: {inputs.input_ids.shape}, device: {inputs.input_ids.device}", flush=True)
print(f"    pixel_values: {inputs.pixel_values.shape}, device: {inputs.pixel_values.device}", flush=True)
print(f"    attention_mask: {inputs.attention_mask.shape}", flush=True)

# ============================================================
# Phase 4: Vision model forward
# ============================================================
t7 = time.time()
with torch.no_grad():
    vision_outputs = model.vision_model(inputs.pixel_values)
    vision_features = vision_outputs.last_hidden_state
t8 = time.time()
print(f"\n[4] Vision forward: {t8-t7:.3f}s", flush=True)
print(f"    vision_features shape: {vision_features.shape}", flush=True)
print(f"    vision_features device: {vision_features.device}", flush=True)

# ============================================================
# Phase 5: Projector forward
# ============================================================
t9 = time.time()
with torch.no_grad():
    vision_embeddings = model.projector(vision_features)
    vision_flat = vision_embeddings.view(-1, vision_embeddings.size(-1))
t10 = time.time()
print(f"\n[5] Projector forward: {t10-t9:.3f}s", flush=True)
print(f"    vision_flat shape: {vision_flat.shape}", flush=True)

# ============================================================
# Phase 6: Text embedding + replacement + full forward
# ============================================================
# First: just embedding lookup
t11 = time.time()
with torch.no_grad():
    token_embeddings = model.text_model.get_input_embeddings()(inputs.input_ids)
t12 = time.time()
print(f"\n[6a] Text embedding lookup: {t12-t11:.3f}s", flush=True)

# Replacement
vision_flat_on_device = vision_flat.to(token_embeddings.device)
image_token_positions = (inputs.input_ids == model.config.image_token_index).nonzero(as_tuple=True)
token_embeddings[image_token_positions] = vision_flat_on_device.to(dtype=token_embeddings.dtype)
print(f"    Replaced {image_token_positions[0].numel()} image tokens", flush=True)

# Full text model forward (single step, no generation)
t13 = time.time()
with torch.no_grad():
    text_output = model.text_model(
        inputs_embeds=token_embeddings,
        attention_mask=inputs.attention_mask,
        use_cache=True,
    )
t14 = time.time()
print(f"\n[6b] Text model single forward: {t14-t13:.3f}s", flush=True)
print(f"    Output logits shape: {text_output.logits.shape}", flush=True)

# ============================================================
# Phase 7: Generation (token by token)
# ============================================================
# Measure generation with different max_new_tokens
for max_tokens in [10, 20, 40, 80, 160]:
    t15 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            eos_token_id=[151643, 151645],
            do_sample=False,
            repetition_penalty=1.3,
        )
    t16 = time.time()
    gen_time = t16 - t15
    generated = output_ids.shape[1] - inputs.input_ids.shape[1]
    per_token = gen_time / max(generated, 1)
    print(f"\n[7] Generation max_new={max_tokens}: {gen_time:.2f}s, actual={generated} tokens, per_token={per_token*1000:.1f}ms", flush=True)

# ============================================================
# Phase 8: Full pipeline end-to-end
# ============================================================
t_total_start = time.time()
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
t_total_end = time.time()

response = processor.decode(output_ids[0], assistant_only=True, skip_special_tokens=True)
decode_time = time.time() - t_total_end

print(f"\n" + "=" * 60, flush=True)
print(f"  TOTAL BENCHMARK SUMMARY", flush=True)
print(f"=" * 60, flush=True)
print(f"  Image preprocessing:    {t2-t1:.3f}s", flush=True)
print(f"  Text tokenization:      {t4-t3:.3f}s", flush=True)
print(f"  Full processor call:    {t6-t5:.3f}s", flush=True)
print(f"  Vision model forward:   {t8-t7:.3f}s", flush=True)
print(f"  Projector forward:      {t10-t9:.3f}s", flush=True)
print(f"  Text embedding:         {t12-t11:.3f}s", flush=True)
print(f"  Text model 1-step:      {t14-t13:.3f}s", flush=True)
print(f"  Generation (80 tokens): {t_total_end-t_total_start:.2f}s", flush=True)
print(f"  Decode response:        {decode_time:.3f}s", flush=True)
print(f"  Total end-to-end:       {t_total_end-t_total_start+decode_time:.2f}s", flush=True)
print(f"\n  Per-token generation speed: {(t_total_end-t_total_start)/80*1000:.1f}ms/token", flush=True)
print(f"\n  Bottleneck analysis:", flush=True)
pre_gen = t6 - t5  # preprocessing
gen_only = t_total_end - t_total_start
print(f"    Preprocessing: {pre_gen:.2f}s ({pre_gen/(pre_gen+gen_only)*100:.1f}% of total)", flush=True)
print(f"    Generation:    {gen_only:.2f}s ({gen_gen/(pre_gen+gen_only)*100:.1f}% of total)" if False else f"    Generation:    {gen_only:.2f}s ({gen_only/(pre_gen+gen_only)*100:.1f}% of total)", flush=True)

# Response preview
print(f"\n  Response ({len(response)} chars): {response[:150]}...", flush=True)
