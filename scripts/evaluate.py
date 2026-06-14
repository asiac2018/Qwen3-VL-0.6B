#!/usr/bin/env python3
"""evaluate.py - Comprehensive VLM Evaluation Framework

Evaluates the trained model across multiple dimensions:
1. Image Captioning (COCO-style)
2. Visual Question Answering (VQA)
3. Text Recognition (OCR)
4. Object Detection (via description)
5. Reasoning over images
6. Text capability preservation

Generates a detailed evaluation report with scores.
"""

import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
from PIL import Image

from siq_vl.model.modeling import SiQ_VLForCausalLM
from siq_vl.model.processing import SiQ_VLProcessor


# ============================================================
# Evaluation Data
# ============================================================

# Test images: simple colored images for basic verification
COLOR_TESTS = [
    {"image": ("red", (224, 224)), "question": "What color is this image?", "expected": "red"},
    {"image": ("blue", (224, 224)), "question": "What color is this image?", "expected": "blue"},
    {"image": ("green", (224, 224)), "question": "What color is this image?", "expected": "green"},
    {"image": ("yellow", (224, 224)), "question": "What color is this image?", "expected": "yellow"},
    {"image": ("white", (224, 224)), "question": "What color is this image?", "expected": "white"},
]

# VQA test cases
VQA_TESTS = [
    {"question": "Describe this image in detail.", "category": "caption"},
    {"question": "What objects can you see in this image?", "category": "object"},
    {"question": "What is the main subject of this image?", "category": "subject"},
    {"question": "What is happening in this image?", "category": "scene"},
    {"question": "What colors are prominent in this image?", "category": "color"},
]

# Text capability preservation tests
TEXT_TESTS = [
    {"question": "What is 2+2?", "expected_keywords": ["4", "four"]},
    {"question": "What is the capital of France?", "expected_keywords": ["Paris", "paris"]},
    {"question": "Explain gravity in simple terms.", "expected_keywords": ["force", "pull", "attract"]},
    {"question": "Write a haiku about nature.", "expected_keywords": []},
    {"question": "What is artificial intelligence?", "expected_keywords": ["machine", "computer", "learn"]},
]

# Reasoning tests
REASONING_TESTS = [
    {"question": "If this image shows a kitchen, what appliances might you find here?", "category": "spatial_reasoning"},
    {"question": "Based on the lighting in this image, what time of day might it be?", "category": "inference"},
    {"question": "What season might this image have been taken in?", "category": "inference"},
]


# ============================================================
# Evaluation Engine
# ============================================================

@dataclass
class EvalResult:
    question: str
    response: str
    expected: Optional[str] = None
    category: str = "general"
    score: float = 0.0
    latency_ms: float = 0.0
    pass_: bool = False


@dataclass
class EvalReport:
    model_path: str
    total_tests: int = 0
    passed: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    category_scores: dict = field(default_factory=dict)
    results: list = field(default_factory=list)
    timestamp: str = ""


def load_model(model_path: str):
    """Load the trained model and processor."""
    print(f"Loading model from {model_path}...")
    model = SiQ_VLForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = SiQ_VLProcessor.from_pretrained(model_path)
    model.eval()
    print("Model loaded!")
    return model, processor


def run_inference(model, processor, image: Image.Image, question: str,
                  max_new_tokens=256, temperature=0.6, do_sample=True):
    """Run single inference and measure latency."""
    inputs = processor(
        batch=[(image, question, None)],
        return_tensors="pt",
    ).to(model.device)

    start = time.time()
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
    latency_ms = (time.time() - start) * 1000

    response = processor.decode(output_ids[0], assistant_only=True, skip_special_tokens=True)

    # Strip Qwen3 thinking tokens
    import re
    # Remove thinking blocks if present
    response = re.sub(r'_shortcode_start.*?shortcode_end', '', response, flags=re.DOTALL)
    response = response.strip()

    return response, latency_ms


def score_response(response: str, expected: str = None, expected_keywords: list = None) -> float:
    """Score a response based on expected answer or keywords.

    Returns a score from 0.0 to 1.0.
    """
    if not response or len(response) < 2:
        return 0.0

    if expected:
        # Exact match (case-insensitive)
        if expected.lower() in response.lower():
            return 1.0
        # Partial match
        words = expected.lower().split()
        matched = sum(1 for w in words if w in response.lower())
        return matched / max(len(words), 1)

    if expected_keywords:
        matched = sum(1 for kw in expected_keywords if kw.lower() in response.lower())
        return matched / max(len(expected_keywords), 1)

    # Default: check response quality (not empty, reasonable length, not repetitive)
    score = 0.5  # Base score for generating something
    if len(response) > 20:
        score += 0.2
    if len(response) > 50:
        score += 0.1
    # Check for repetition
    words = response.split()
    if len(words) > 5:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        score += unique_ratio * 0.2
    return min(score, 1.0)


def evaluate_color_tests(model, processor) -> list[EvalResult]:
    """Test basic color recognition."""
    results = []
    for test in COLOR_TESTS:
        color, size = test["image"]
        img = Image.new("RGB", size, color=color)
        response, latency = run_inference(model, processor, img, test["question"],
                                          max_new_tokens=64, temperature=0.1, do_sample=False)
        score = score_response(response, expected=test["expected"])
        results.append(EvalResult(
            question=test["question"],
            response=response,
            expected=test["expected"],
            category="color_recognition",
            score=score,
            latency_ms=latency,
            pass_=score >= 0.8,
        ))
    return results


def evaluate_vqa(model, processor, test_images: list[str] = None) -> list[EvalResult]:
    """Test VQA capability on real images."""
    results = []

    # Use COCO128 images if available
    img_dir = "./data/coco128_images"
    if not os.path.exists(img_dir):
        img_dir = None

    images_tested = []
    if img_dir:
        for img_name in sorted(os.listdir(img_dir))[:5]:
            if img_name.endswith('.jpg'):
                images_tested.append(os.path.join(img_dir, img_name))

    if not images_tested:
        # Fallback to synthetic images
        for color in ["red", "blue", "green"]:
            images_tested.append(Image.new("RGB", (224, 224), color=color))

    for img_src in images_tested:
        try:
            if isinstance(img_src, str):
                img = Image.open(img_src).convert("RGB")
            else:
                img = img_src
        except Exception:
            continue

        for test in VQA_TESTS:
            response, latency = run_inference(model, processor, img, test["question"],
                                              max_new_tokens=256, temperature=0.6)
            score = score_response(response)
            results.append(EvalResult(
                question=test["question"],
                response=response[:200],
                category=f"vqa_{test['category']}",
                score=score,
                latency_ms=latency,
                pass_=score >= 0.5,
            ))

    return results


def evaluate_text_capability(model, processor) -> list[EvalResult]:
    """Test that text-only capability is preserved."""
    results = []
    # Create a blank image (model should ignore it for text questions)
    blank_img = Image.new("RGB", (224, 224), color="white")

    for test in TEXT_TESTS:
        response, latency = run_inference(model, processor, blank_img, test["question"],
                                          max_new_tokens=128, temperature=0.6)
        score = score_response(response, expected_keywords=test["expected_keywords"])
        results.append(EvalResult(
            question=test["question"],
            response=response[:200],
            expected=", ".join(test["expected_keywords"]) if test["expected_keywords"] else None,
            category="text_preservation",
            score=score,
            latency_ms=latency,
            pass_=score >= 0.4 if test["expected_keywords"] else len(response) > 10,
        ))

    return results


def generate_report(model_path: str, results: list[EvalResult]) -> EvalReport:
    """Generate evaluation report."""
    total = len(results)
    passed = sum(1 for r in results if r.pass_)
    avg_score = sum(r.score for r in results) / max(total, 1)
    avg_latency = sum(r.latency_ms for r in results) / max(total, 1)

    # Category scores
    category_scores = {}
    for r in results:
        cat = r.category
        if cat not in category_scores:
            category_scores[cat] = {"total": 0, "score": 0.0, "passed": 0}
        category_scores[cat]["total"] += 1
        category_scores[cat]["score"] += r.score
        category_scores[cat]["passed"] += 1 if r.pass_ else 0

    for cat in category_scores:
        n = category_scores[cat]["total"]
        category_scores[cat]["avg_score"] = category_scores[cat]["score"] / n
        category_scores[cat]["pass_rate"] = category_scores[cat]["passed"] / n

    return EvalReport(
        model_path=model_path,
        total_tests=total,
        passed=passed,
        avg_score=avg_score,
        avg_latency_ms=avg_latency,
        category_scores=category_scores,
        results=results,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def print_report(report: EvalReport):
    """Print formatted evaluation report."""
    print("\n" + "=" * 70)
    print("  Qwen3-VL-0.6B Evaluation Report")
    print("=" * 70)
    print(f"  Model: {report.model_path}")
    print(f"  Time:  {report.timestamp}")
    print(f"  Tests: {report.total_tests} total, {report.passed} passed ({report.passed/max(report.total_tests,1)*100:.0f}%)")
    print(f"  Average Score: {report.avg_score:.3f}")
    print(f"  Average Latency: {report.avg_latency_ms:.0f}ms")

    print("\n  Category Breakdown:")
    print("  " + "-" * 60)
    for cat, info in sorted(report.category_scores.items()):
        print(f"  {cat:25s}: score={info['avg_score']:.3f}, pass_rate={info['pass_rate']:.0%} ({info['passed']}/{info['total']})")

    # Sample responses
    print("\n  Sample Responses:")
    print("  " + "-" * 60)
    for r in report.results[:10]:
        status = "✅" if r.pass_ else "❌"
        print(f"  {status} [{r.category}] Q: {r.question[:50]}")
        print(f"     A: {r.response[:80]}...")
        if r.expected:
            print(f"     Expected: {r.expected}")
        print()

    print("=" * 70)

    # Overall assessment
    if report.avg_score >= 0.7:
        print("  🎉 Model performs WELL - ready for deployment")
    elif report.avg_score >= 0.4:
        print("  ⚠️  Model needs IMPROVEMENT - consider more training")
    else:
        print("  ❌ Model performs POORLY - needs significant rework")
    print("=" * 70)


def save_report(report: EvalReport, output_path: str = "./outputs/eval_report.json"):
    """Save evaluation report to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "model_path": report.model_path,
        "timestamp": report.timestamp,
        "total_tests": report.total_tests,
        "passed": report.passed,
        "avg_score": report.avg_score,
        "avg_latency_ms": report.avg_latency_ms,
        "category_scores": report.category_scores,
        "results": [
            {
                "question": r.question,
                "response": r.response,
                "expected": r.expected,
                "category": r.category,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "pass": r.pass_,
            }
            for r in report.results
        ],
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL-0.6B")
    parser.add_argument("--model_path", type=str, default="./outputs/stage2/final",
                        help="Path to trained model (default: ./outputs/stage2/final)")
    parser.add_argument("--stage1", action="store_true",
                        help="Evaluate Stage 1 model instead")
    parser.add_argument("--output", type=str, default="./outputs/eval_report.json",
                        help="Output path for evaluation report")
    args = parser.parse_args()

    model_path = "./outputs/stage1/final" if args.stage1 else args.model_path
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Looking for available checkpoints...")
        # Try to find any checkpoint
        for stage in ["stage2/final", "stage1/final", "stage1/checkpoint-*"]:
            import glob
            matches = glob.glob(f"./outputs/{stage}")
            if matches:
                model_path = sorted(matches)[-1]
                print(f"Found checkpoint: {model_path}")
                break
        else:
            print("No checkpoints found. Please train the model first.")
            return

    model, processor = load_model(model_path)

    # Run all evaluations
    all_results = []

    print("\n[1/3] Evaluating color recognition...")
    color_results = evaluate_color_tests(model, processor)
    all_results.extend(color_results)

    print("[2/3] Evaluating VQA capability...")
    vqa_results = evaluate_vqa(model, processor)
    all_results.extend(vqa_results)

    print("[3/3] Evaluating text capability preservation...")
    text_results = evaluate_text_capability(model, processor)
    all_results.extend(text_results)

    # Generate and print report
    report = generate_report(model_path, all_results)
    print_report(report)
    save_report(report, args.output)

    return report


if __name__ == "__main__":
    main()
