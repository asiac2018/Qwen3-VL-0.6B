# Qwen3-0.6B 拥有视觉 — 完整可落地方案

> 基于 H20 GPU 集群（8×95GB）、transformers 4.57.6、PyTorch 2.9.0 环境制定
> 日期：2026-06-13

---

## 一、核心思路

将纯文本 LLM（Qwen3-0.6B）升级为视觉语言模型（VLM），本质上是**在 LLM 前端嫁接一个视觉编码器（Vision Encoder），并通过投影器（Projector）将视觉特征对齐到 LLM 的嵌入空间**。

存在两条技术路线：

| 路线 | 方案 | 优势 | 劣势 |
|------|------|------|------|
| **A. 官方 Qwen3-VL 架构** | 复用 `Qwen3VLForConditionalGeneration`，从 Qwen3-VL-2B 提取视觉编码器，拼接 Qwen3-0.6B 语言模型 | 官方架构，原生支持 mRoPE、DeepStack、视频 | Qwen3-0.6B 的 `rope_scaling` 为 null，需要适配 mRoPE；Qwen3VL 使用自定义 Vision Transformer，非标准 SigLIP |
| **B. SiQ-VL 架构**（推荐） | 使用开源项目 [siq-vl](https://huggingface.co/duoan/siq-vl_siglip2-so400m-patch14-224_qwen3-0.6b-base_stage1) 的架构：SigLIP2 + Pixel Shuffle Projector + Qwen3 | 社区已有 Qwen3-0.6B 的 stage1 权重可直接加载；架构简洁、代码自包含；Pixel Shuffle 压缩 token 数量 | 非官方架构，需自定义 modeling/processing 代码 |

**推荐方案 B（SiQ-VL 架构）**，理由：
1. HuggingFace 上已有 `duoan/siq-vl_siglip2-so400m-patch14-224_qwen3-0.6b-base_stage1` 的预训练权重
2. 代码完全自包含，可直接下载使用
3. 架构简洁（SigLIP2 + Pixel Shuffle MLP + Qwen3），训练收敛更快
4. 社区已有完整的 stage1 和 stage2 训练流程

---

## 二、架构详解

### 2.1 SiQ-VL 模型架构

```
┌─────────────────────────────────────────────────────┐
│                    SiQ_VLForCausalLM                │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  Vision Model │  │  Projector   │  │ Text Model │ │
│  │  (SigLIP2)   │→│  (Pixel      │→│  (Qwen3    │ │
│  │  1152-dim     │  │  Shuffle +   │  │  0.6B)     │ │
│  │  27 layers    │  │  Linear)     │  │  1024-dim  │ │
│  │  ~430M params │  │  ~4.7M params│  │  ~600M     │ │
│  └──────────────┘  └──────────────┘  └───────────┘ │
│                                                     │
│  输入: pixel_values (B, C, H, W) + input_ids       │
│  输出: logits (B, seq_len, vocab_size)              │
└─────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
Image (PIL)
  → SiQ_VLImageProcessor: resize/normalize → pixel_values (B, 3, 224, 224)
  → SigLIP2 Vision Encoder: 27层 ViT → vision_features (B, 256, 1152)
  → Pixel Shuffle: (B, 256, 1152) → (B, 64, 4608)    # factor=2, 256/4=64 tokens
  → Linear Projection: (B, 64, 4608) → (B, 64, 1024)  # 对齐到 LLM hidden_size
  → 替换 input_ids 中的 <|image_pad|> token embeddings
  → Qwen3-0.6B: 自回归生成文本
```

### 2.3 关键参数

| 组件 | 参数量 | 维度 | 说明 |
|------|--------|------|------|
| SigLIP2-so400m-patch14-224 | ~430M | hidden=1152, 27层 | 视觉编码器，冻结 |
| Pixel Shuffle Projector | ~4.7M | 4608→1024 | 唯一训练部分（Stage 1） |
| Qwen3-0.6B | ~600M | hidden=1024, 28层 | 语言模型，Stage 1 冻结 |
| LM Head | ~155M | 1024→151936 | 与 LLM 共享（tie_word_embeddings=true） |
| **总计** | **~1.19B** | | |

### 2.4 特殊 Token

Qwen3-0.6B 的 tokenizer 已包含视觉 token：
- `<|vision_start|>` (151652) — 视觉区域开始标记
- `<|vision_end|>` (151653) — 视觉区域结束标记
- `<|vision_pad|>` (151654) — 视觉填充
- `<|image_pad|>` (151655) — 图像占位符（核心 token）
- `<|video_pad|>` (151656) — 视频占位符

---

## 三、训练流程

### 3.1 总览：两阶段训练

```
Stage 1: Projector Pre-training（对齐阶段）
  ─── 冻结 Vision Encoder + LLM，只训练 Projector ───
  目的: 让 Projector 学会将视觉特征映射到 LLM 的嵌入空间
  数据: 大规模图文对（如 LLaVA-Pretrain, 558K）
  时长: ~5000 steps, 约 2-4 小时（单卡 H20）

Stage 2: Visual Instruction Tuning（指令微调阶段）
  ─── 冻结 Vision Encoder，训练 Projector + LLM（LoRA） ───
  目的: 让模型学会理解指令并生成合理的视觉回复
  数据: 高质量指令微调数据（如 LLaVA-Instruct, 665K）
  时长: ~10000 steps, 约 4-8 小时（单卡 H20）
```

### 3.2 Stage 1: Projector Pre-training

**训练策略**:
- Vision Encoder: **冻结** ❌
- Projector: **训练** ✅
- Text Model (Qwen3-0.6B): **冻结** ❌

**超参数**（来自已有训练记录）:
```python
TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,        # 有效 batch_size = 4 × 1 GPU × 4 = 16
    learning_rate=1e-3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_steps=5000,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    max_grad_norm=1.0,
)
```

**训练数据**: LLaVA-Pretrain (558K 图文对)
```python
# 数据格式: (image, question, answer)
# 例:
# (image.jpg, "Describe this image.", "A cat sitting on a windowsill...")
```

### 3.3 Stage 2: Visual Instruction Tuning

**训练策略**:
- Vision Encoder: **冻结** ❌
- Projector: **训练** ✅
- Text Model (Qwen3-0.6B): **LoRA 微调** ✅

**LoRA 配置**:
```python
LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)
```

**超参数**:
```python
TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,                   # 比 Stage 1 小 50 倍
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,
)
```

**训练数据**: LLaVA-Instruct-150K / ShareGPT4V / LLaVA-OneVision 等

---

## 四、环境与资源

### 4.1 硬件资源

| 资源 | 状态 | 说明 |
|------|------|------|
| GPU | 8× NVIDIA H20 (95GB each) | GPU 3 空闲最多，可用于训练 |
| 内存 | 1.5TB | 充足 |
| 磁盘 | 3.4TB 可用 | 充足 |
| CUDA | 12.4 | 兼容 PyTorch 2.9 |

### 4.2 软件环境

| 组件 | 版本 | 状态 |
|------|------|------|
| Python | 3.12.7 | ✅ |
| PyTorch | 2.9.0 | ✅ |
| Transformers | 4.57.6 | ✅ 支持 Qwen3VL |
| PEFT | 0.17.0 | ✅ 支持 LoRA |
| Accelerate | 1.10.0 | ✅ |
| Datasets | 4.0.0 | ✅ |
| Pillow | 10.4.0 | ✅ |
| qwen-vl-utils | 0.0.14 | ✅ |
| einops | 0.8.1 | ✅ |
| torchmetrics | ❌ 缺失 | 需安装 |

### 4.3 本地已有模型

| 模型 | 路径 | 说明 |
|------|------|------|
| Qwen3-0.6B | `/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B/` | 文本基础模型 |
| Qwen3-VL-8B-int4 | `/root/models/Qwen3-VL-8B-int4/` | 可参考的 VL 模型 |
| siq-vl stage1 | HuggingFace: `duoan/siq-vl_siglip2-so400m-patch14-224_qwen3-0.6b-base_stage1` | 已缓存到本地 |

---

## 五、实施步骤

### Step 0: 环境准备

```bash
cd /workspace2/cy/Qwen3-0.6B

# 安装缺失依赖
pip install torchmetrics

# 安装 siq-vl 包（如果可用），或直接使用自定义代码
# pip install siq-vl  # 如果存在
```

### Step 1: 下载和组织模型

```bash
# 1.1 Qwen3-0.6B 已在本地
QWEN3_PATH=/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B

# 1.2 下载 SigLIP2 视觉编码器
python3 -c "
from transformers import SiglipVisionModel
model = SiglipVisionModel.from_pretrained('google/siglip2-so400m-patch14-224')
model.save_pretrained('./models/siglip2-so400m-patch14-224')
"

# 1.3 下载 siq-vl stage1 权重（可选，用于跳过 stage1 训练）
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('duoan/siq-vl_siglip2-so400m-patch14-224_qwen3-0.6b-base_stage1',
                  local_dir='./models/siq-vl-stage1')
"
```

### Step 2: 准备模型代码

从 siq-vl 仓库下载三个核心文件，放置到项目目录：

```bash
mkdir -p /workspace2/cy/Qwen3-0.6B/siq_vl/model

# 下载 modeling.py, processing.py, configuration.py
python3 -c "
from huggingface_hub import hf_hub_download
for f in ['modeling.py', 'processing.py', 'configuration.py']:
    path = hf_hub_download('duoan/siq-vl_siglip2-so400m-patch14-224_qwen3-0.6b-base_stage1', f)
    import shutil
    shutil.copy(path, f'./siq_vl/model/{f}')
"
```

> ⚠️ **关键修改**: siq-vl 的 `SiQ_VLTextModel` 继承自 `Qwen2ForCausalLM`，但 Qwen3-0.6B 是 `Qwen3ForCausalLM`。
> 需要将 `SiQ_VLTextModel` 改为继承 `Qwen3ForCausalLM`，并将配置基类从 `Qwen2Config` 改为 `Qwen3Config`。

### Step 3: 修改代码适配 Qwen3

修改 `configuration.py`:
```python
# 将 Qwen2Config 改为 Qwen3Config
from transformers import Qwen3Config  # 改这里

class SiQ_VLTextConfig(Qwen3Config):  # 改这里
    model_type = "siq_vl"
    base_config_key = "text_config"
```

修改 `modeling.py`:
```python
# 将 Qwen2ForCausalLM 改为 Qwen3ForCausalLM
from transformers import Qwen3ForCausalLM  # 改这里

class SiQ_VLTextModel(Qwen3ForCausalLM):  # 改这里
    config: SiQ_VLTextConfig = None

    def __init__(self, config: SiQ_VLTextConfig = None):
        super().__init__(config)
        self.config = config

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()
```

修改 `get_stage1_model_and_processor` 函数中的默认路径：
```python
def get_stage1_model_and_processor(
    pretrained_vision_model_path: str = "google/siglip2-so400m-patch14-224",
    pretrained_text_model_path: str = "/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B",  # 改这里
    ...
```

### Step 4: 下载训练数据

#### Stage 1 数据: LLaVA-Pretrain
```bash
mkdir -p ./data/stage1
# 下载 LLaVA-Pretrain 数据集
python3 -c "
from datasets import load_dataset
ds = load_dataset('liuhaotian/LLaVA-Pretrain', split='train')
ds.save_to_disk('./data/stage1/llava_pretrain')
"
```

#### Stage 2 数据: LLaVA-Instruct
```bash
mkdir -p ./data/stage2
python3 -c "
from datasets import load_dataset
ds = load_dataset('liuhaotian/LLaVA-Instruct-150K', split='train')
ds.save_to_disk('./data/stage2/llava_instruct')
"
```

### Step 5: Stage 1 训练

```python
#!/usr/bin/env python3
"""stage1_train.py - Projector Pre-training"""

import sys
sys.path.insert(0, '/workspace2/cy/Qwen3-0.6B')

import torch
from transformers import TrainingArguments, Trainer
from siq_vl.model.modeling import get_stage1_model_and_processor
from siq_vl.model.processing import SiQ_VLProcessor

# 1. 初始化模型和处理器
model, processor = get_stage1_model_and_processor(
    pretrained_vision_model_path="./models/siglip2-so400m-patch14-224",
    pretrained_text_model_path="/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B",
    vision_pixel_shuffle_factor=2,
    enable_dynamic_tiling=False,
)

# 2. 加载数据集
from datasets import load_from_disk
dataset = load_from_disk("./data/stage1/llava_pretrain")

# 3. 数据预处理函数
def preprocess_function(examples):
    from PIL import Image
    batch = []
    for img_path, question, answer in zip(examples["image"], examples["question"], examples["answer"]):
        if isinstance(img_path, str):
            try:
                img = Image.open(img_path).convert("RGB")
            except:
                img = Image.new("RGB", (224, 224), color="black")
        else:
            img = img_path.convert("RGB") if hasattr(img_path, 'convert') else Image.new("RGB", (224, 224))
        batch.append((img, question, answer))

    return processor(batch=batch, return_tensors="pt")

# 4. 训练配置
training_args = TrainingArguments(
    output_dir="./outputs/stage1",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_steps=5000,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    max_grad_norm=1.0,
    save_steps=500,
    eval_steps=500,
    logging_steps=10,
    remove_unused_columns=False,
    report_to="none",  # 或 "wandb"
)

# 5. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # preprocessing 需要通过自定义 collate_fn 实现
)

trainer.train()
model.save_pretrained("./outputs/stage1/final")
processor.save_pretrained("./outputs/stage1/final")
```

### Step 6: Stage 2 训练

```python
#!/usr/bin/env python3
"""stage2_train.py - Visual Instruction Tuning"""

import sys
sys.path.insert(0, '/workspace2/cy/Qwen3-0.6B')

from siq_vl.model.modeling import get_stage2_model_and_processor

# 1. 加载 Stage 1 检查点
model, processor = get_stage2_model_and_processor(
    stage_1_checkpoint_path="./outputs/stage1/final",
    use_lora=True,
    lora_r=64,
    lora_alpha=16,
    lora_dropout=0.05,
)

# 2. 加载 Stage 2 数据集
from datasets import load_from_disk
dataset = load_from_disk("./data/stage2/llava_instruct")

# 3. 训练配置（学习率大幅降低）
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./outputs/stage2",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    max_grad_norm=1.0,
    save_steps=500,
    logging_steps=10,
    remove_unused_columns=False,
)

# 4. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
model.save_pretrained("./outputs/stage2/final")
processor.save_pretrained("./outputs/stage2/final")
```

### Step 7: 推理验证

```python
#!/usr/bin/env python3
"""inference.py - 验证模型视觉能力"""

import sys
sys.path.insert(0, '/workspace2/cy/Qwen3-0.6B')

import torch
from PIL import Image
from siq_vl.model.modeling import SiQ_VLForCausalLM
from siq_vl.model.processing import SiQ_VLProcessor

# 加载训练好的模型
model = SiQ_VLForCausalLM.from_pretrained(
    "./outputs/stage2/final",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = SiQ_VLProcessor.from_pretrained("./outputs/stage2/final")

# 测试推理
image = Image.open("test_image.jpg").convert("RGB")
inputs = processor(
    batch=[(image, "Describe this image in detail.", None)],
    return_tensors="pt",
).to(model.device)

model.eval()
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        attention_mask=inputs.attention_mask,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
    )

response = processor.decode(output_ids[0], assistant_only=True, skip_special_tokens=True)
print(f"Model response: {response}")
```

---

## 六、GPU 资源分配建议

### 方案 1: 单卡训练（最简单，推荐先用）

使用 GPU 3（当前仅占用 ~2.2GB）：

```bash
CUDA_VISIBLE_DEVICES=3 python stage1_train.py
CUDA_VISIBLE_DEVICES=3 python stage2_train.py
```

模型参数量估算：
- Stage 1 仅训练 Projector (~4.7M 参数) → 显存占用约 5-8GB
- Stage 2 训练 Projector + LoRA (~4.7M + ~30M 参数) → 显存占用约 15-20GB
- H20 95GB 完全足够，甚至可以用更大 batch size

### 方案 2: 多卡训练（加速）

```bash
# 使用 accelerate 启动多卡
accelerate launch --num_processes 4 --gpu_ids 0,1,2,3 stage1_train.py
```

---

## 七、风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| siq-vl 使用 Qwen2ForCausalLM，Qwen3 接口不同 | 需修改代码 | Qwen3 和 Qwen2 的 CausalLM 接口高度兼容，仅需替换基类 |
| Qwen3-0.6B 没有 rope_scaling (mRoPE) | mRoPE 不支持 | siq-vl 方案不需要 mRoPE，使用标准 RoPE 即可 |
| 训练数据下载可能受限 | 数据缺失 | 可使用 HuggingFace datasets 或本地数据集 |
| 模型精度可能不如官方 Qwen3-VL | 效果差 | 0.6B 模型能力有限，可通过更多数据/更长训练弥补 |
| SigLIP2 模型下载需要网络 | 下载失败 | 可提前缓存到本地 |

---

## 八、快速启动（5 分钟验证）

如果只想快速验证流程，可以**直接加载已有的 stage1 权重**跳过训练：

```python
import sys
sys.path.insert(0, '/workspace2/cy/Qwen3-0.6B')

import torch
from PIL import Image

# 1. 下载 siq-vl 代码
from huggingface_hub import hf_hub_download
import shutil, os
os.makedirs('./siq_vl/model', exist_ok=True)
for f in ['modeling.py', 'processing.py', 'configuration.py']:
    path = hf_hub_download('duoan/siq-vl_siglip2-so400m-patch14-224_qwen3-0.6b-base_stage1', f)
    shutil.copy(path, f'./siq_vl/model/{f}')

# 2. 创建 __init__.py
with open('./siq_vl/__init__.py', 'w') as f:
    f.write('')
with open('./siq_vl/model/__init__.py', 'w') as f:
    f.write('')

# 3. 修改代码适配 Qwen3（见 Step 3）
# ...（修改 configuration.py 和 modeling.py 中的 Qwen2 → Qwen3）

# 4. 加载模型
from siq_vl.model.modeling import SiQ_VLForCausalLM
from siq_vl.model.processing import SiQ_VLProcessor

model = SiQ_VLForCausalLM.from_pretrained(
    "duoan/siq-vl_siglip2-so400m-patch14-224_qwen3-0.6b-base_stage1",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = SiQ_VLProcessor.from_pretrained(
    "duoan/siq-vl_siglip2-so400m-patch14-224_qwen3-0.6b-base_stage1"
)

# 5. 测试推理
image = Image.new("RGB", (224, 224), color="red")
inputs = processor(
    batch=[(image, "What color is this image?", None)],
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        attention_mask=inputs.attention_mask,
        max_new_tokens=64,
    )
print(processor.decode(output_ids[0], assistant_only=True, skip_special_tokens=True))
```

---

## 九、时间估算

| 阶段 | 操作 | 预计时间 |
|------|------|----------|
| 环境准备 | 安装依赖、下载代码 | 10 分钟 |
| 代码适配 | 修改 Qwen2→Qwen3 | 30 分钟 |
| 模型下载 | SigLIP2 + siq-vl 权重 | 20 分钟 |
| 数据下载 | LLaVA-Pretrain + Instruct | 30 分钟 |
| Stage 1 训练 | 5000 steps (单卡 H20) | 2-4 小时 |
| Stage 2 训练 | 1 epoch (单卡 H20) | 4-8 小时 |
| **总计** | | **~8-14 小时** |

---

## 十、备选方案：官方 Qwen3-VL 架构

如果需要使用官方 Qwen3-VL 架构（支持 mRoPE、DeepStack、视频理解），可以：

1. 从 Qwen3-VL-2B 中提取视觉编码器权重
2. 使用 Qwen3-0.6B 的语言模型权重
3. 新建 Qwen3VLConfig，vision_config 的 `out_hidden_size` 设为 1024
4. 初始化 Projector（Merger）权重随机
5. 进行 Stage 1/Stage 2 训练

但这种方式需要更复杂的适配工作，包括 mRoPE 位置编码的处理。建议先走 SiQ-VL 路线验证概念，再考虑迁移到官方架构。
