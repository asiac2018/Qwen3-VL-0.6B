#!/usr/bin/env python3
"""gradio_demo.py - Gradio Web Interface for Qwen3-VL-0.6B

Features:
- Image upload with preview
- Multiple preset questions via buttons
- Thinking mode toggle (show/hide Qwen3 reasoning)
- Adjustable generation parameters
- Real-time model info display
- Example images gallery with select-to-load
- Single-GPU deployment for stability

Bug fixes (v2):
- PIL Image type handling: isinstance(image, Image.Image) before Image.fromarray
- Device mismatch: use device_map={"": 3} to keep model on single GPU
- Dropdown replaced with clickable question buttons (more reliable in Gradio 6)
- Gallery select event connected to load image into Image component
"""

import os
import sys
import re
import glob
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
import gradio as gr
from PIL import Image

from siq_vl.model.modeling import SiQ_VLForCausalLM
from siq_vl.model.processing import SiQ_VLProcessor


def strip_thinking(text):
    """Remove Qwen3 thinking blocks from response."""
    think_pattern = r'⁠⁠.*?⁠⁠'
    cleaned = re.sub(think_pattern, '', text, flags=re.DOTALL)
    return cleaned.strip() if cleaned.strip() else text.strip()


# ============================================================
# Model Loading — Use single GPU to avoid device mismatch
# ============================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "./outputs/stage2/merged_v4")
MODEL_GPU = int(os.environ.get("MODEL_GPU", "3"))  # Use GPU 3 for model

# Try to find the best available checkpoint
if not os.path.exists(MODEL_PATH):
    candidates = sorted(glob.glob("./outputs/stage2/merged_v4"), key=os.path.getmtime, reverse=True)
    if not candidates:
        candidates = sorted(glob.glob("./outputs/stage2/merged_v3"), key=os.path.getmtime, reverse=True)
    if not candidates:
        candidates = sorted(glob.glob("./outputs/stage2/merged_v2"), key=os.path.getmtime, reverse=True)
    if not candidates:
        candidates = sorted(glob.glob("./outputs/stage2/merged"), key=os.path.getmtime, reverse=True)
    if not candidates:
        candidates = sorted(glob.glob("./outputs/stage1/checkpoint-*"), key=os.path.getmtime, reverse=True)
    if candidates:
        MODEL_PATH = candidates[0]
        print(f"Auto-detected checkpoint: {MODEL_PATH}")

print(f"Loading model from {MODEL_PATH} on cuda:{MODEL_GPU}...")
model = SiQ_VLForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map={"": MODEL_GPU},  # Force single GPU to avoid cross-GPU device mismatch
)
processor = SiQ_VLProcessor.from_pretrained(MODEL_PATH, fix_mistral_regex=True)
model.eval()

# ============================================================
# CUDA Warmup — Pre-compile kernels to avoid 20s+ cold start delay
# PyTorch compiles CUDA kernels on first use with new tensor shapes.
# Running a small dummy inference here ensures all kernels are
# compiled BEFORE the user sends their first request.
# ============================================================
print("Warming up CUDA kernels...")
_warmup_start = time.time()
_warmup_img = Image.new("RGB", (224, 224), color="red")
_warmup_inputs = processor(
    batch=[(_warmup_img, "test", None)],
    return_tensors="pt",
).to(model.device)
with torch.no_grad():
    _ = model.generate(
        input_ids=_warmup_inputs.input_ids,
        pixel_values=_warmup_inputs.pixel_values,
        attention_mask=_warmup_inputs.attention_mask,
        max_new_tokens=5,
        eos_token_id=[151643, 151645],
        do_sample=False,
        repetition_penalty=1.3,
    )
torch.cuda.synchronize()
print(f"CUDA warmup done in {time.time()-_warmup_start:.1f}s — first user request will be fast")

# Model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model loaded on cuda:{MODEL_GPU}! Total params: {total_params/1e9:.2f}B, Trainable: {trainable_params/1e6:.1f}M")


# ============================================================
# Inference Function
# ============================================================
def predict(image, question, max_tokens, temperature, top_p, do_sample, show_thinking):
    """Run inference and return the response."""
    t_start = time.time()
    if image is None:
        return "Please upload an image first."

    if not question.strip():
        question = "Describe this image in detail."

    try:
        if isinstance(image, Image.Image):
            # Gradio type="pil" passes PIL.Image directly
            image = image.convert("RGB")
        elif isinstance(image, str):
            # File path
            image = Image.open(image).convert("RGB")
        else:
            # numpy array
            image = Image.fromarray(image).convert("RGB")
    except Exception as e:
        return f"Error loading image: {e}"

    t_proc = time.time()
    inputs = processor(
        batch=[(image, question, None)],
        return_tensors="pt",
    ).to(model.device)

    t_gen = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            attention_mask=inputs.attention_mask,
            max_new_tokens=int(max_tokens),
            eos_token_id=[151643, 151645],
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else 0.0,
            top_p=float(top_p),
            top_k=20,
            repetition_penalty=1.3,
        )

    response = processor.decode(output_ids[0], assistant_only=True, skip_special_tokens=True)

    if not show_thinking:
        response = strip_thinking(response)

    t_end = time.time()
    n_tokens = output_ids.shape[1] - inputs.input_ids.shape[1]
    print(f"  [TIMING] proc={t_gen-t_proc:.3f}s gen={t_end-t_gen:.2f}s total={t_end-t_start:.2f}s tokens={n_tokens}", flush=True)

    return response


# ============================================================
# Example Questions — Use buttons instead of Dropdown for reliability
# ============================================================
EXAMPLE_QUESTIONS = [
    "Describe this image in detail.",
    "What objects can you see in this image?",
    "What is the main subject of this image?",
    "What colors are prominent in this image?",
    "Is there any text visible in this image?",
    "What is happening in this image?",
    "Describe the setting or background.",
    "How many objects are in this image?",
]

# Find example images
EXAMPLE_IMAGES = []
coco_dir = "./data/coco128_images"
if os.path.exists(coco_dir):
    for img_name in sorted(os.listdir(coco_dir))[:6]:
        if img_name.endswith('.jpg'):
            EXAMPLE_IMAGES.append(os.path.join(coco_dir, img_name))


# ============================================================
# Gradio Interface
# ============================================================
with gr.Blocks(title="Qwen3-VL-0.6B Vision-Language Model") as demo:
    gr.Markdown(
        """
        # 🌟 Qwen3-VL-0.6B: Vision-Language Model Demo

        Upload an image and ask questions about it. The model will analyze the image and generate a response.

        **Architecture**: SigLIP2 + 2-Layer MLP Projector (Pixel Shuffle) + Qwen3-0.6B
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📷 Upload Image",
                type="pil",
                height=350,
            )

            # Example images as gallery — clicking loads into Image component
            if EXAMPLE_IMAGES:
                gr.Markdown("**Example Images** (click to load):")
                example_gallery = gr.Gallery(
                    value=[(Image.open(p), os.path.basename(p)) for p in EXAMPLE_IMAGES[:6]],
                    label="Example Images",
                    show_label=False,
                    columns=3,
                    height=150,
                    allow_preview=False,
                    interactive=False,
                )

            question_input = gr.Textbox(
                label="❓ Question",
                placeholder="Ask a question about the image...",
                value="Describe this image in detail.",
                lines=2,
            )

            # Question buttons — more reliable than Dropdown in Gradio 6
            gr.Markdown("**Quick Questions** (click to set):")
            with gr.Row():
                q_btn1 = gr.Button("Describe", size="sm")
                q_btn2 = gr.Button("Objects", size="sm")
                q_btn3 = gr.Button("Subject", size="sm")
            with gr.Row():
                q_btn4 = gr.Button("Colors", size="sm")
                q_btn5 = gr.Button("Text?", size="sm")
                q_btn6 = gr.Button("Count", size="sm")

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                max_tokens = gr.Slider(
                    minimum=32, maximum=1024, value=512,
                    step=32, label="Max New Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.6,
                    step=0.1, label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.95,
                    step=0.05, label="Top P"
                )
                do_sample = gr.Checkbox(
                    label="Enable Sampling",
                    value=True,
                )
                show_thinking = gr.Checkbox(
                    label="Show Thinking (Qwen3 reasoning)",
                    value=False,
                )

            submit_btn = gr.Button("🚀 Generate Response", variant="primary", size="lg")

            # Model info
            gr.Markdown(
                f"""
                <div style="background:#f0f4f8; padding:10px; border-radius:8px; margin:10px 0;">
                <strong>Model Info</strong><br>
                Path: <code>{MODEL_PATH}</code><br>
                GPU: <code>cuda:{MODEL_GPU}</code><br>
                Total Parameters: {total_params/1e9:.2f}B<br>
                Trainable Parameters: {trainable_params/1e6:.1f}M
                </div>
                """
            )

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="💬 Model Response",
                lines=20,
                interactive=False,
            )

    # ============================================================
    # Event Connections
    # ============================================================

    # Question buttons — set question text on click
    q_btn1.click(fn=lambda: "Describe this image in detail.", outputs=question_input)
    q_btn2.click(fn=lambda: "What objects can you see in this image?", outputs=question_input)
    q_btn3.click(fn=lambda: "What is the main subject of this image?", outputs=question_input)
    q_btn4.click(fn=lambda: "What colors are prominent in this image?", outputs=question_input)
    q_btn5.click(fn=lambda: "Is there any text visible in this image? If so, what does it say?", outputs=question_input)
    q_btn6.click(fn=lambda: "How many objects are in this image?", outputs=question_input)

    # Gallery select — load clicked image into the Image component
    if EXAMPLE_IMAGES:
        def gallery_select_handler(evt: gr.SelectData):
            """When a gallery image is clicked, load it into the image input."""
            # evt.value contains the path of the selected image
            # evt.index is the index in the gallery
            if evt.index is not None and evt.index < len(EXAMPLE_IMAGES):
                img_path = EXAMPLE_IMAGES[evt.index]
                return Image.open(img_path).convert("RGB")
            return None

        example_gallery.select(
            fn=gallery_select_handler,
            outputs=image_input,
        )

    # Generate button
    submit_btn.click(
        fn=predict,
        inputs=[image_input, question_input, max_tokens, temperature, top_p, do_sample, show_thinking],
        outputs=output_text,
    )

    # Also submit on Enter in question input
    question_input.submit(
        fn=predict,
        inputs=[image_input, question_input, max_tokens, temperature, top_p, do_sample, show_thinking],
        outputs=output_text,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--gpu", type=int, default=3, help="GPU index to load model on")
    args = parser.parse_args()

    MODEL_GPU = args.gpu

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
