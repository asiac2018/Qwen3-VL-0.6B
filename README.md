# Qwen3-VL-0.6B

> Adding visual capabilities to Qwen3-0.6B using SiQ-VL architecture

## Architecture

```
Image (224×224) → SigLIP2-so400m (430M) → 2-Layer MLP + Pixel Shuffle → Qwen3-0.6B (600M) → Text Response
```

- **SigLIP2-so400m-patch14-224**: Vision encoder (frozen)
- **2-Layer MLP Projector**: Pixel Shuffle factor=2, 256→64 vision tokens (trainable)
- **Qwen3-0.6B**: Text decoder (LoRA r=128 in Stage 2)
- **Total**: ~1.03B parameters

## Quick Start

### 1. Download Models

```bash
# Vision encoder
python3 scripts/download_models.py --vision

# Text model
python3 scripts/download_models.py --text
```

### 2. Stage 1 Training (Projector Alignment)

```bash
python3 scripts/stage1_train.py
```

### 3. Stage 2 Training (LoRA Instruction Tuning)

```bash
python3 scripts/stage2_train.py
```

### 4. Launch Demo

```bash
python3 demo/gradio_demo.py --port 7860 --gpu 3
```

### 5. Evaluate

```bash
python3 scripts/evaluate.py --model_path ./outputs/stage2/merged_v4
```

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| SiQ-VL architecture (not official Qwen3-VL) | Official Qwen3-VL not released for 0.6B |
| Qwen3ForCausalLM (not Qwen2) | Qwen3 has q_norm/k_norm, no attention bias |
| eos_token_id=[151643, 151645] | 151645 = `<|im_end|>`, 151643 is empty token |
| 2-Layer MLP + Pixel Shuffle | Better alignment than single-layer Linear |
| LoRA r=128, alpha=128 | Effective scaling ratio alpha/r=1.0 |
| device_map={"": 3} | Single GPU avoids cross-device mismatch |

## Known Issues & Fixes

1. **PIL Image type**: `gr.Image(type="pil")` passes PIL.Image, not numpy array → use `isinstance(image, Image.Image)` check
2. **Device mismatch**: `device_map="auto"` splits model across GPUs → use single GPU `device_map={"": GPU_ID}`
3. **LoRA merge**: Custom wrapper can't `merge_and_unload()` directly → manual key mapping required
4. **Tokenizer regex**: Mistral regex warning → pass `fix_mistral_regex=True`

## Iterative Optimization (v1→v4)

| Version | Color | Text | Anti-hallucination | Key Change |
|---------|-------|------|--------------------|------------|
| v1 | 0% | 87.5% | 0/3 | Initial |
| v2 | 80% | 100% | 1/3 | +Color +Negative |
| v3 | 70% | 62.5% ⚠️ | 1/3 | +Concise (regression!) |
| **v4** ⭐ | 70% | 87.5% | **3/3** | Balanced detail + anti-hallucination |

## Project Structure

```
Qwen3-VL-0.6B/
├── siq_vl/model/          # Core model code
│   ├── modeling.py         # SiQ_VLForCausalLM (560 lines)
│   ├── configuration.py    # Model config (186 lines)
│   └── processing.py       # Data processor (672 lines)
├── scripts/                # Training & evaluation
│   ├── stage1_train.py     # Projector alignment training
│   ├── stage2_train.py     # LoRA instruction tuning
│   ├── evaluate.py         # Multi-dimension evaluation
│   ├── merge_lora.py       # LoRA weight merging
│   └── download_models.py  # Download base models
├── demo/
│   └── gradio_demo.py      # Gradio 6 web interface
├── data/                   # Training data (JSON format)
├── outputs/                # Model checkpoints & reports
│   ├── EVALUATION_REPORT.md
│   └── PROJECT_SUMMARY.md
├── PLAN.md                 # Original plan
├── PLAN_v2.md              # Revised plan (with fixes)
└── README.md               # This file
```

## Environment

- GPU: 8×H20 (95GB each)
- Python 3.12 + PyTorch 2.x + Transformers 4.x
- Gradio 6.8.0
- PEFT (LoRA)

## License

This project uses open-source models:
- SigLIP2: Apache 2.0
- Qwen3-0.6B: Apache 2.0 (Qwen license)
- Training code: MIT
