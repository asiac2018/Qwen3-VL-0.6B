#!/usr/bin/env python3
"""setup_and_train.py - One-click setup and training script

This script handles the complete pipeline:
1. Download SigLIP2 vision encoder (requires network)
2. Verify model compatibility
3. Run Stage 1 training (Projector alignment)
4. Run Stage 2 training (Visual instruction tuning)
5. Launch Gradio demo

Usage:
    # Full pipeline
    CUDA_VISIBLE_DEVICES=3 python scripts/setup_and_train.py

    # Skip downloads (if already done)
    CUDA_VISIBLE_DEVICES=3 python scripts/setup_and_train.py --skip-download

    # Quick test only (no training)
    python scripts/setup_and_train.py --test-only
"""

import os
import sys
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def check_dependencies():
    """Check and install required dependencies."""
    print("\n" + "=" * 60)
    print("Step 0: Checking Dependencies")
    print("=" * 60)

    required = {
        'torch': 'torch',
        'transformers': 'transformers',
        'peft': 'peft',
        'accelerate': 'accelerate',
        'datasets': 'datasets',
        'PIL': 'pillow',
        'einops': 'einops',
        'torchmetrics': 'torchmetrics',
        'gradio': 'gradio',
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ❌ {package} (missing)")

    if missing:
        print(f"\n  Installing missing packages: {', '.join(missing)}")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing, check=True)

    return True


def download_models():
    """Download required model weights."""
    print("\n" + "=" * 60)
    print("Step 1: Downloading Model Weights")
    print("=" * 60)

    # 1. SigLIP2 vision encoder
    siglip_path = "./models/siglip2-so400m-patch14-224"
    if os.path.exists(os.path.join(siglip_path, "model.safetensors")):
        print(f"  ✅ SigLIP2 already exists at {siglip_path}")
    else:
        print(f"  Downloading SigLIP2 to {siglip_path}...")
        from transformers import SiglipVisionModel
        model = SiglipVisionModel.from_pretrained('google/siglip2-so400m-patch14-224')
        model.save_pretrained(siglip_path)
        print(f"  ✅ SigLIP2 downloaded")

    # 2. Qwen3-0.6B (should already be local)
    qwen3_path = "/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B"
    if os.path.exists(os.path.join(qwen3_path, "model.safetensors")):
        print(f"  ✅ Qwen3-0.6B exists at {qwen3_path}")
    else:
        print(f"  ❌ Qwen3-0.6B not found at {qwen3_path}")
        return False

    return True


def download_data():
    """Download training data."""
    print("\n" + "=" * 60)
    print("Step 2: Downloading Training Data")
    print("=" * 60)

    # Stage 1 data
    stage1_path = "./data/stage1/llava_pretrain"
    if os.path.exists(stage1_path):
        print(f"  ✅ Stage 1 data exists at {stage1_path}")
    else:
        print(f"  Downloading LLaVA-Pretrain...")
        from datasets import load_dataset
        ds = load_dataset("liuhaotian/LLaVA-Pretrain", split="train")
        ds.save_to_disk(stage1_path)
        print(f"  ✅ Stage 1 data: {len(ds)} samples")

    # Stage 2 data
    stage2_path = "./data/stage2/llava_instruct"
    if os.path.exists(stage2_path):
        print(f"  ✅ Stage 2 data exists at {stage2_path}")
    else:
        print(f"  Downloading LLaVA-Instruct...")
        from datasets import load_dataset
        ds = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")
        ds.save_to_disk(stage2_path)
        print(f"  ✅ Stage 2 data: {len(ds)} samples")

    return True


def run_verification():
    """Run quick verification tests."""
    print("\n" + "=" * 60)
    print("Step 3: Running Verification Tests")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "tests/quick_verify.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return True


def run_stage1_training():
    """Run Stage 1 training."""
    print("\n" + "=" * 60)
    print("Step 4: Stage 1 Training (Projector Alignment)")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "scripts/stage1_train.py"],
        capture_output=False
    )
    return result.returncode == 0


def run_stage2_training():
    """Run Stage 2 training."""
    print("\n" + "=" * 60)
    print("Step 5: Stage 2 Training (Visual Instruction Tuning)")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "scripts/stage2_train.py"],
        capture_output=False
    )
    return result.returncode == 0


def launch_demo():
    """Launch Gradio demo."""
    print("\n" + "=" * 60)
    print("Step 6: Launching Gradio Demo")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "demo/gradio_demo.py"],
        capture_output=False
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="SiQ-VL Setup and Training Pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip model/data downloads")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run verification tests, no training")
    parser.add_argument("--stage", type=int, choices=[1, 2],
                        help="Run only a specific training stage")
    parser.add_argument("--demo-only", action="store_true",
                        help="Only launch the Gradio demo")
    args = parser.parse_args()

    print("🔬 SiQ-VL v2: Qwen3-0.6B Vision-Language Model Pipeline")
    print(f"   Project root: {PROJECT_ROOT}")

    # Step 0: Dependencies
    check_dependencies()

    if args.demo_only:
        launch_demo()
        return

    # Step 1-2: Downloads
    if not args.skip_download:
        if not download_models():
            print("\n❌ Model download failed. Check network connectivity.")
            print("   You can retry later with: python scripts/setup_and_train.py")
            return
        if not download_data():
            print("\n⚠️  Data download failed. Training will use dummy data.")

    # Step 3: Verification
    if not run_verification():
        print("\n❌ Verification failed. Please fix the errors before training.")
        return

    if args.test_only:
        print("\n✅ All tests passed! Ready for training.")
        print("   Run with --stage 1 to start Stage 1 training.")
        return

    # Step 4-5: Training
    if args.stage is None or args.stage == 1:
        if not run_stage1_training():
            print("\n❌ Stage 1 training failed.")
            return

    if args.stage is None or args.stage == 2:
        if not run_stage2_training():
            print("\n❌ Stage 2 training failed.")
            return

    # Step 6: Demo
    print("\n✅ Training complete! Launching demo...")
    launch_demo()


if __name__ == "__main__":
    main()
