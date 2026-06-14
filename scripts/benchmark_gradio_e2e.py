#!/usr/bin/env python3
"""benchmark_gradio_e2e.py - 精确测量用户在浏览器中的真实体验延迟

对比：
- Gradio 按钮点击 → 界面响应（用户真实体验）
- Gradio API 调用 → 结果返回（服务端延迟）
- 直接 Python 推理（模型真实速度）
"""

import os, sys, time, json, threading
import requests
import base64
from io import BytesIO
from PIL import Image
from playwright.sync_api import sync_playwright

GRADIO_URL = "http://localhost:7861"
TEST_IMAGE = "/workspace2/cy/Qwen3-0.6B/data/coco128_images/000000000009.jpg"

results = {}

# ============================================================
# Test 1: Direct Python inference
# ============================================================
print("=" * 60)
print("  Test 1: Direct Python inference (no Gradio)")
print("=" * 60)

sys.path.insert(0, '/workspace2/cy/Qwen3-0.6B')
os.chdir('/workspace2/cy/Qwen3-0.6B')

import torch
from siq_vl.model.modeling import SiQ_VLForCausalLM
from siq_vl.model.processing import SiQ_VLProcessor

model = SiQ_VLForCausalLM.from_pretrained(
    "./outputs/stage2/merged_v4",
    torch_dtype=torch.bfloat16,
    device_map={"": 3},
)
processor = SiQ_VLProcessor.from_pretrained("./outputs/stage2/merged_v4", fix_mistral_regex=True)
model.eval()

img = Image.open(TEST_IMAGE).convert("RGB")
question = "What objects can you see in this image?"

# Warmup
inputs = processor(batch=[(img, question, None)], return_tensors="pt").to(model.device)
with torch.no_grad():
    model.generate(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values,
                   attention_mask=inputs.attention_mask, max_new_tokens=10,
                   eos_token_id=[151643, 151645], do_sample=False, repetition_penalty=1.3)

# Measure direct
trials = []
for i in range(3):
    t0 = time.time()
    inputs = processor(batch=[(img, question, None)], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids, pixel_values=inputs.pixel_values,
            attention_mask=inputs.attention_mask, max_new_tokens=80,
            eos_token_id=[151643, 151645], do_sample=False, repetition_penalty=1.3,
        )
    elapsed = time.time() - t0
    tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
    trials.append({"time": elapsed, "tokens": tokens})
    print(f"  Trial {i+1}: {elapsed:.2f}s ({tokens} tokens, {elapsed/tokens*1000:.1f}ms/token)")

results["direct_python"] = trials

# ============================================================
# Test 2: Gradio API call
# ============================================================
print(f"\n{'=' * 60}")
print("  Test 2: Gradio API round-trip")
print("=" * 60)

# Upload image
buf = BytesIO()
img.save(buf, format="JPEG", quality=85)
img_bytes = buf.getvalue()

upload_resp = requests.post(f"{GRADIO_URL}/gradio_api/upload",
    files={"files": ("test.jpg", img_bytes, "image/jpeg")}, timeout=10)
upload_path = upload_resp.json()[0]

trials = []
for i in range(3):
    t0 = time.time()
    payload = {
        "data": [
            {"path": upload_path, "meta": {"_type": "gradio.FileData"}},
            "What objects can you see in this image?",
            80, 0.1, 0.95, False, False,
        ]
    }
    submit_resp = requests.post(f"{GRADIO_URL}/gradio_api/call/predict", json=payload, timeout=10)
    event_id = submit_resp.json()["event_id"]

    stream_resp = requests.get(f"{GRADIO_URL}/gradio_api/call/predict/{event_id}", stream=True, timeout=120)
    for line in stream_resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            data = line[5:]
            if "complete" in line or '"msg":"complete"' in data:
                elapsed = time.time() - t0
                trials.append({"time": elapsed})
                print(f"  Trial {i+1}: {elapsed:.2f}s")
                break

results["gradio_api"] = trials

# ============================================================
# Test 3: Browser Playwright (user experience)
# ============================================================
print(f"\n{'=' * 60}")
print("  Test 3: Browser Playwright (real user experience)")
print("=" * 60)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})

    # Load page
    page.goto(GRADIO_URL, timeout=30000, wait_until="networkidle")
    time.sleep(3)

    # Upload image
    page.locator('input[type="file"]').first.set_input_files(TEST_IMAGE)
    time.sleep(2)

    # Set question
    page.locator('textarea').first.fill("What objects can you see?")

    # Click generate and measure
    trials = []
    for i in range(3):
        # Clear output first
        textareas = page.locator('textarea')
        if textareas.count() >= 2:
            # Can't easily clear readonly textarea, but we can measure when it changes

        t0 = time.time()
        page.locator('button:has-text("Generate")').click()

        # Wait for response to appear
        for j in range(60):  # up to 120s
            time.sleep(2)
            textareas = page.locator('textarea')
            if textareas.count() >= 2:
                try:
                    output_val = textareas.nth(1).input_value()
                    if output_val and len(output_val) > 10:
                        elapsed = time.time() - t0
                        trials.append({"time": elapsed, "chars": len(output_val)})
                        print(f"  Trial {i+1}: {elapsed:.2f}s ({len(output_val)} chars)")
                        break
                except:
                    pass
        else:
            elapsed = time.time() - t0
            trials.append({"time": elapsed, "chars": 0})
            print(f"  Trial {i+1}: TIMEOUT after {elapsed:.1f}s")

        # Wait between trials
        time.sleep(2)

    results["browser_playwright"] = trials
    browser.close()

# ============================================================
# Summary
# ============================================================
print(f"\n{'=' * 60}")
print("  SUMMARY: Where is the 20s delay?")
print("=" * 60)

avg_direct = sum(t["time"] for t in results["direct_python"]) / len(results["direct_python"])
avg_api = sum(t["time"] for t in results["gradio_api"]) / len(results["gradio_api"])
avg_browser = sum(t["time"] for t in results["browser_playwright"]) / len(results["browser_playwright"])

print(f"  Direct Python:     {avg_direct:.2f}s  (model capability)")
print(f"  Gradio API:       {avg_api:.2f}s  (Gradio server overhead)")
print(f"  Browser (actual): {avg_browser:.2f}s  (user experience)")
print(f"\n  Overhead analysis:")
print(f"    Gradio server:  +{avg_api - avg_direct:.2f}s")
print(f"    Browser→server:  +{avg_browser - avg_api:.2f}s")
print(f"\n  If browser is ~20s, the bottleneck is likely:")
print(f"    - Image base64 encoding/decoding in browser")
print(f"    - Gradio's queue/event system")
print(f"    - SSE connection establishment")

with open("/workspace2/cy/Qwen3-0.6B/outputs/inference_benchmark.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to outputs/inference_benchmark.json")
