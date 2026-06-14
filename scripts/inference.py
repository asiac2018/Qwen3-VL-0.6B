#!/usr/bin/env python3
"""inference.py - VLM Inference Script (v2)

Loads a trained SiQ-VL model and runs inference on images.
Handles Qwen3 thinking mode properly.
"""

import os
import sys
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from PIL import Image

from siq_vl.model.modeling import SiQ_VLForCausalLM
from siq_vl.model.processing import SiQ_VLProcessor


def strip_thinking(text: str) -> str:
    """Remove Qwen3 thinking blocks from response.
    
    Qwen3 models may generate thinking blocks like:
    <think>internal reasoning</think>actual response
    
    This function strips the thinking part and returns only the actual response.
    """
    # Pattern: <think>...</think> followed by the actual response
    pattern = r'<think>.*?</think>\s*'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned.strip() if cleaned.strip() else text.strip()


def load_model(model_path: str = "./outputs/stage2/final", device: str = "auto"):
    """Load the trained SiQ-VL model and processor."""
    model = SiQ_VLForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    )
    processor = SiQ_VLProcessor.from_pretrained(model_path)
    return model, processor


def inference(model, processor, image: Image.Image, question: str,
              max_new_tokens: int = 512, temperature: float = 0.6,
              do_sample: bool = True, strip_think: bool = True):
    """Run inference on a single image-question pair.
    
    Args:
        strip_think: If True, remove Qwen3 thinking blocks from response.
    """
    model.eval()

    inputs = processor(
        batch=[(image, question, None)],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=[151643, 151645],
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.2,
        )

    response = processor.decode(output_ids[0], assistant_only=True, skip_special_tokens=True)
    
    if strip_think:
        response = strip_thinking(response)
    
    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SiQ-VL Inference")
    parser.add_argument("--model_path", type=str, default="./outputs/stage2/final",
                        help="Path to the trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, default="Describe this image in detail.",
                        help="Question to ask about the image")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--no_strip_think", action="store_true",
                        help="Keep thinking blocks in output")
    args = parser.parse_args()

    model, processor = load_model(args.model_path)
    image = Image.open(args.image).convert("RGB")
    response = inference(model, processor, image, args.question,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        strip_think=not args.no_strip_think)
    print(f"\nResponse:\n{response}")
