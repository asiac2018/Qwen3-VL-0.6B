# Qwen3-0.6B 拥有视觉 — 优化落地方案 v2.0

> 基于 H20 GPU 集群（8×95GB）、transformers 4.57.6、PyTorch 2.9.0 环境制定
> 日期：2026-06-14
> 基于 PLAN.md v1.0 的多视角优化迭代版本

---

## 〇、v1.0 方案的关键问题与修复

经过深入验证，v1.0 方案存在以下 **必须修复** 的问题：

| # | 问题 | 严重性 | 说明 |
|---|------|--------|------|
| 1 | **Qwen2/Qwen3 权重不兼容** | 🔴 HIGH | siq-vl 代码使用 `Qwen2ForCausalLM`，但 Qwen3 有 `q_norm/k_norm`（7,168 params）而 Qwen2 有 `attention bias`（86,016 params）。加载 Qwen3 权重到 Qwen2 架构会丢弃 q_norm/k_norm 并随机初始化 86K 个 bias 参数，导致文本模型退化 |
| 2 | **eos_token_id 错误** | 🔴 HIGH | siq-vl config.json 设置 `eos_token_id=151643`（空 token），正确值应为 `151645`（`<|im_end|>`），否则 generate() 永远不会正确停止 |
| 3 | **Projector 设计过于简单** | 🟡 MEDIUM | 单层 Linear(4608→1024) 无非线性，只能学习线性映射，无法充分对齐视觉-语言嵌入空间 |
| 4 | **unfreez_text_model() 中 eval() 应为 train()** | 🟡 MEDIUM | modeling.py 第167行，解冻后仍设置为 eval 模式，影响 LoRA 训练中的 dropout 和 batchnorm |
| 5 | **训练数据规模偏小** | 🟡 MEDIUM | LLaVA-Pretrain 558K + LLaVA-Instruct 150K，相比现代 VLM 的百万级数据偏少 |
| 6 | **LoRA alpha/r 比例过低** | 🟡 MEDIUM | alpha=16, r=64，有效缩放比 alpha/r=0.25，LoRA 贡献被过度压缩 |
| 7 | **未利用 H20 大显存** | 🟢 LOW | Stage1 仅需 ~5-8GB 显存，H20 有 95GB 可用，可大幅提升 batch size |

---

## 一、核心架构（优化后）

### 1.1 技术路线选择：SiQ-VL 架构（保留，但需深度改造）

仍然选择 SiQ-VL 架构路线（SigLIP2 + Pixel Shuffle Projector + Qwen3-0.6B），理由不变：
1. 架构简洁，训练收敛快
2. 社区有参考实现
3. 不需要 mRoPE 等复杂适配

**但必须进行以下关键改造**：
- 基类从 `Qwen2ForCausalLM` 改为 `Qwen3ForCausalLM`
- 基类配置从 `Qwen2Config` 改为 `Qwen3Config`
- **不能直接使用 siq-vl stage1 权重**（因 Qwen2/Qwen3 不兼容），必须从头训练 Projector

### 1.2 优化后架构

```
┌───────────────────────────────────────────────────────────┐
│                    SiQ_VLForCausalLM (v2)                 │
│                                                           │
│  ┌──────────────┐  ┌──────────────────────┐  ┌─────────┐ │
│  │  Vision Model │  │  Projector (v2)      │  │ Qwen3   │ │
│  │  (SigLIP2)   │→│  Pixel Shuffle +      │→│ 0.6B    │ │
│  │  1152-dim     │  │  2-Layer MLP +       │  │ 1024-dim│ │
│  │  27 layers    │  │  LayerNorm +         │  │ 28层    │ │
│  │  ~430M params │  │  Residual            │  │ ~600M   │ │
│  └──────────────┘  │  ~11.5M params       │  └─────────┘ │
│                     └──────────────────────┘               │
│                                                           │
│  输入: pixel_values (B, C, H, W) + input_ids             │
│  输出: logits (B, seq_len, vocab_size=151936)             │
│  总参数: ~1.2B                                            │
└───────────────────────────────────────────────────────────┘
```

### 1.3 数据流（优化后）

```
Image (PIL)
  → SiQ_VLImageProcessor: resize/normalize → pixel_values (B, 3, 224, 224)
  → SigLIP2 Vision Encoder: 27层 ViT → vision_features (B, 256, 1152)
  → Pixel Shuffle: (B, 256, 1152) → (B, 64, 4608)    # factor=2, 256/4=64 tokens
  → Linear1: (B, 64, 4608) → (B, 64, 1024)            # 映射到 LLM 维度
  → GELU Activation
  → Linear2: (B, 64, 1024) → (B, 64, 1024)            # 非线性精炼
  → LayerNorm: 归一化视觉 token 分布
  → + Residual (Linear1 output)                          # 稳定训练
  → 替换 input_ids 中的 <|image_pad|> token embeddings
  → Qwen3-0.6B: 自回归生成文本
```

### 1.4 关键参数（优化后）

| 组件 | 参数量 | 维度 | 说明 |
|------|--------|------|------|
| SigLIP2-so400m-patch14-224 | ~430M | hidden=1152, 27层 | 视觉编码器，冻结 |
| **2-Layer MLP Projector** | **~11.5M** | 4608→1024→1024 | **v2 优化：2层 MLP + LayerNorm + Residual** |
| Qwen3-0.6B | ~600M | hidden=1024, 28层 | 语言模型，Stage 1 冻结，Stage 2 LoRA |
| LM Head | ~155M | 1024→151936 | 与 LLM 共享（tie_word_embeddings=true） |
| **总计** | **~1.2B** | | |

---

## 二、Projector 设计优化详解

### 2.1 v1 vs v2 对比

| 方面 | v1 (单层 Linear) | v2 (2-Layer MLP + Residual + LN) |
|------|------------------|----------------------------------|
| 结构 | PixelShuffle → Linear(4608→1024) | PixelShuffle → Linear(4608→1024) → GELU → Linear(1024→1024) → LN + Residual |
| 参数量 | 4.7M | 11.5M (+6.8M) |
| 非线性 | ❌ 无 | ✅ GELU |
| 归一化 | ❌ 无 | ✅ LayerNorm |
| 残差连接 | ❌ 无 | ✅ Linear1 输出跳连 |
| 对齐能力 | 线性映射，仅能仿射变换 | 可学习非线性映射，对齐不同拓扑的嵌入空间 |

### 2.2 v2 Projector 代码

```python
class SiQ_VLProjector(PreTrainedModel):
    config: SiQ_VLProjectorConfig = None

    def __init__(self, config: SiQ_VLProjectorConfig = None):
        super().__init__(config)
        self.config = config
        self.vision_pixel_shuffle_factor = config.vision_pixel_shuffle_factor
        input_dim = config.vision_hidden_size * (config.vision_pixel_shuffle_factor ** 2)

        # 2-Layer MLP with residual connection
        self.linear1 = nn.Linear(input_dim, config.text_hidden_size, bias=False)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(config.text_hidden_size, config.text_hidden_size, bias=False)
        self.norm = nn.LayerNorm(config.text_hidden_size)

        self.apply(self._init_weights)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Kaiming init for first layer (post-PixelShuffle inputs may have large values)
            if module is self.linear1:
                nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _pixel_shuffle(self, x):
        # ... (same as v1, omitted for brevity)

    def forward(self, x):
        x = self._pixel_shuffle(x)
        residual = self.linear1(x)       # (B, 64, 1024)
        h = self.act(residual)            # GELU
        h = self.linear2(h)              # (B, 64, 1024)
        h = self.norm(h + residual)      # Residual + LayerNorm
        return h
```

### 2.3 为什么 2-Layer MLP 更好

1. **非线性变换**：视觉特征按视觉相似性聚类，语言特征按语义相似性聚类，两者拓扑结构不同。单层线性只能做仿射变换，2 层 MLP + GELU 可以学习非线性映射
2. **残差连接**：保证最坏情况下退化为线性映射，不会比 v1 差
3. **LayerNorm**：稳定视觉 token 进入 LLM 的分布，防止分布偏移导致训练不稳定
4. **额外开销极小**：仅增加 6.8M 参数（0.5%），训练时间增加 <10%

---

## 三、训练流程优化

### 3.1 Stage 1: Projector Pre-training（对齐阶段）

**训练策略**（不变）:
- Vision Encoder: **冻结** ❌
- Projector: **训练** ✅
- Text Model (Qwen3-0.6B): **冻结** ❌

**超参数**（优化后）:
```python
TrainingArguments(
    per_device_train_batch_size=16,       # v1: 4 → v2: 16 (利用 H20 大显存)
    gradient_accumulation_steps=4,        # 有效 batch_size = 16 × 1 × 4 = 64 (v1: 16)
    learning_rate=1e-3,                   # 保持不变
    lr_scheduler_type="cosine",
    warmup_steps=200,                     # v1: warmup_ratio=0.03 → v2: 固定步数
    max_steps=10000,                      # v1: 5000 → v2: 10000 (更多数据需要更多步)
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    max_grad_norm=1.0,
    save_steps=1000,                      # v1: 500 → v2: 1000
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=10,
    load_best_model_at_end=True,          # v2 新增：自动加载最优检查点
    metric_for_best_model="loss",
    remove_unused_columns=False,
    report_to="none",
)
```

**训练数据**（优化后）:
- LLaVA-Pretrain (558K) — 基础图文对
- ShareGPT4V-Pretrain (1.2M) — 高质量描述性标题
- **合计: ~1.76M 图文对** (v1: 558K)

**预计时间**: ~3-4 小时（单卡 H20，bs=64）

### 3.2 Stage 2: Visual Instruction Tuning（指令微调阶段）

**训练策略**（不变）:
- Vision Encoder: **冻结** ❌
- Projector: **训练** ✅
- Text Model (Qwen3-0.6B): **LoRA 微调** ✅

**LoRA 配置**（优化后）:
```python
LoraConfig(
    r=128,                               # v1: 64 → v2: 128 (更高秩)
    lora_alpha=128,                      # v1: 16 → v2: 128 (alpha/r=1.0，v1=0.25)
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)
# v2 新增：考虑添加 modules_to_save=["embed_tokens"] 用于视觉 token 适配
```

**超参数**（优化后）:
```python
TrainingArguments(
    per_device_train_batch_size=4,       # v1: 2 → v2: 4
    gradient_accumulation_steps=4,       # 有效 batch_size = 4 × 1 × 4 = 16 (v1: 16)
    learning_rate=5e-5,                  # v1: 2e-5 → v2: 5e-5 (更高 LoRA alpha 支持更高 lr)
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    num_train_epochs=2,                  # v1: 1 → v2: 2 (更多 epoch)
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    weight_decay=0.01,
    max_grad_norm=1.0,
    save_steps=500,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    remove_unused_columns=False,
)
```

**训练数据**（优化后）:
- LLaVA-Instruct-150K (通用 VQA)
- ShareGPT4V-Instruct (100K, 高质量详细描述)
- ALLaVA-Instruct-4M (4M, 多样化任务包括 OCR、图表、推理)
- **使用策略**: 每个 epoch 随机采样 20% 数据，跨 epoch 洗牌
- **合计可用: ~4.25M 指令样本** (v1: 150K)

**预计时间**: ~6-10 小时（单卡 H20）

---

## 四、关键代码修改清单

### 4.1 configuration.py 修改

```python
# 修改前:
from transformers import AutoConfig, PretrainedConfig, Qwen2Config, SiglipVisionConfig

class SiQ_VLTextConfig(Qwen2Config):
    model_type = "siq_vl"
    base_config_key = "text_config"

# 修改后:
from transformers import AutoConfig, PretrainedConfig, Qwen3Config, SiglipVisionConfig

class SiQ_VLTextConfig(Qwen3Config):     # ← Qwen2Config → Qwen3Config
    model_type = "siq_vl"
    base_config_key = "text_config"
```

### 4.2 modeling.py 修改

```python
# 修改1: 导入替换
# 修改前:
from transformers import Qwen2ForCausalLM, Qwen2TokenizerFast
# 修改后:
from transformers import Qwen3ForCausalLM, AutoTokenizer

# 修改2: 基类替换
# 修改前:
class SiQ_VLTextModel(Qwen2ForCausalLM):
# 修改后:
class SiQ_VLTextModel(Qwen3ForCausalLM):    # ← 关键修改

# 修改3: unfreez_text_model() 修复
# 修改前:
def unfreez_text_model(self):
    for param in self.text_model.parameters():
        param.requires_grad_(True)
    self.text_model.eval()        # ← BUG: 解冻后应为 train()
# 修改后:
def unfreez_text_model(self):
    for param in self.text_model.parameters():
        param.requires_grad_(True)
    self.text_model.train()       # ← 修复

# 修改4: Projector 升级为 2-Layer MLP (见 §2.2)

# 修改5: get_stage1_model_and_processor 默认路径更新
def get_stage1_model_and_processor(
    pretrained_vision_model_path: str = "google/siglip2-so400m-patch14-224",
    pretrained_text_model_path: str = "/workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B",
    vision_pixel_shuffle_factor: int = 2,
    enable_dynamic_tiling: bool = False,
) -> tuple[SiQ_VLForCausalLM, SiQ_VLProcessor]:
```

### 4.3 processing.py 修改

```python
# 修改1: 导入替换
# 修改前:
from transformers import Qwen2TokenizerFast
# 修改后:
from transformers import AutoTokenizer  # Qwen3 兼容 Qwen2TokenizerFast

# 修改2: 系统提示词更新
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. "
    "You can perceive, understand, and analyze images provided by the user. "
    "When answering questions about images, provide detailed, accurate, and helpful responses. "
    "If the image details are unclear, admit it rather than hallucinating."
)
# 注意：移除了 "SiQ-VL" 和 "Qwen2" 的自引用，使其更通用

# 修改3: processor 类名保持不变但 tokenizer_class 更新
tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")  # 保持，Qwen3 兼容
```

### 4.4 config.json 修改

```json
{
    "eos_token_id": [151643, 151645],     // ← 修复: 添加正确的 eos_token_id
    "text_config": {
        "eos_token_id": [151643, 151645],  // ← 修复
        ...
    }
}
```

---

## 五、GPU 资源分配（优化后）

### 推荐方案: 单卡 GPU 3（95GB 空闲）

```bash
CUDA_VISIBLE_DEVICES=3 python stage1_train.py
CUDA_VISIBLE_DEVICES=3 python stage2_train.py
```

**显存估算**:
- Stage 1: Projector ~11.5M + 优化器状态 → 约 8-12GB (H20 95GB 充裕)
- Stage 2: Projector ~11.5M + LoRA ~70M + 优化器 → 约 25-35GB (H20 95GB 充裕)
- 推理: 全模型 bf16 ~2.4GB + KV cache → 约 5-10GB

### 可选: 多卡加速

```bash
# 使用 4 卡训练 (GPU 0,2,3 可用)
CUDA_VISIBLE_DEVICES=0,2,3 accelerate launch \
    --num_processes 3 \
    --mixed_precision bf16 \
    stage1_train.py
```

---

## 六、推理与部署优化

### 6.1 EOS Token 处理

```python
# 推理时必须显式传入正确的 eos_token_id
output_ids = model.generate(
    input_ids=inputs.input_ids,
    pixel_values=inputs.pixel_values,
    attention_mask=inputs.attention_mask,
    max_new_tokens=512,
    eos_token_id=[151643, 151645],  # ← 显式指定，防止不停止
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2,
)
```

### 6.2 Thinking 模式处理

```python
# 推理时禁用 thinking 模式（简化输出）
# 保持 siq-vl 的简化 chat template 即可
# 如果需要 thinking 模式，需要修改 chat template 支持 enable_thinking 参数
```

### 6.3 Flash Attention 2 加速

```python
# 加载模型时启用 Flash Attention 2
model = SiQ_VLForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # ← 加速推理
)
```

### 6.4 多轮对话支持

当前架构的 forward 方法在 `past_key_values is not None` 时跳过视觉处理，这是正确的。但需确保多轮对话中：
- 之前的图像 token 已在 KV cache 中
- 新的 input_ids 不包含旧的 `<|image_pad|>` token

---

## 七、训练数据准备

### 7.1 Stage 1 数据

```bash
mkdir -p ./data/stage1

# 方案 A: 最小化数据集（快速验证）
python3 -c "
from datasets import load_dataset
ds = load_dataset('liuhaotian/LLaVA-Pretrain', split='train')
ds.save_to_disk('./data/stage1/llava_pretrain')
print(f'LLaVA-Pretrain: {len(ds)} samples')
"

# 方案 B: 增强数据集（推荐）
# 在方案 A 基础上添加 ShareGPT4V-Pretrain
python3 -c "
from datasets import load_dataset
ds = load_dataset('Lin-Chen/ShareGPT4V', split='train')  # 或具体子集
ds.save_to_disk('./data/stage1/sharegpt4v_pretrain')
print(f'ShareGPT4V: {len(ds)} samples')
"
```

### 7.2 Stage 2 数据

```bash
mkdir -p ./data/stage2

# 核心数据集
python3 -c "
from datasets import load_dataset
ds = load_dataset('liuhaotian/LLaVA-Instruct-150K', split='train')
ds.save_to_disk('./data/stage2/llava_instruct')
print(f'LLaVA-Instruct: {len(ds)} samples')
"

# 补充数据集（按可用性添加）
# ShareGPT4V-Instruct, ALLaVA-Instruct-4M 等
```

---

## 八、完整实施步骤

### Step 0: 环境准备

```bash
cd /workspace2/cy/Qwen3-0.6B

# 安装缺失依赖
pip install torchmetrics gradio  # gradio 用于后续界面

# 创建项目结构
mkdir -p siq_vl/model
mkdir -p data/stage1 data/stage2
mkdir -p outputs/stage1 outputs/stage2
mkdir -p scripts
mkdir -p tests
mkdir -p demo
```

### Step 1: 下载和组织模型代码

```bash
# 1.1 下载 siq-vl 源码（从 HuggingFace 缓存）
SNAPSHOT="7b39d66ee0eb2c40e7af66a5d600f621dd90c024"
SIQ_DIR="/home/caiyi/.cache/huggingface/hub/models--duoan--siq-vl_siglip2-so400m-patch14-224_qwen3-0.6b-base_stage1/snapshots/$SNAPSHOT"

cp "$SIQ_DIR/configuration.py" ./siq_vl/model/
cp "$SIQ_DIR/modeling.py" ./siq_vl/model/
cp "$SIQ_DIR/processing.py" ./siq_vl/model/

# 1.2 创建 __init__.py
echo "from siq_vl.model.modeling import *
from siq_vl.model.processing import *
from siq_vl.model.configuration import *" > ./siq_vl/__init__.py
echo "from siq_vl.model.modeling import *
from siq_vl.model.processing import *
from siq_vl.model.configuration import *" > ./siq_vl/model/__init__.py
```

### Step 2: 应用代码修改

按照 §4 的修改清单，修改 configuration.py、modeling.py、processing.py

### Step 3: 下载模型权重

```bash
# 3.1 下载 SigLIP2 视觉编码器
python3 -c "
from transformers import SiglipVisionModel
model = SiglipVisionModel.from_pretrained('google/siglip2-so400m-patch14-224')
model.save_pretrained('./models/siglip2-so400m-patch14-224')
"

# 3.2 Qwen3-0.6B 已在本地
# ln -s /workspace1/chenliang/SmartResume-main/models/Qwen3-0.6B ./models/Qwen3-0.6B
```

### Step 4: 准备训练数据

按照 §7 下载和准备数据集

### Step 5: Stage 1 训练

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/stage1_train.py
```

### Step 6: Stage 2 训练

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/stage2_train.py
```

### Step 7: 推理验证 + Gradio 界面

```bash
python demo/gradio_demo.py
```

---

## 九、评估方案

### 9.1 自动评估

| 评估维度 | 数据集 | 指标 | 说明 |
|----------|--------|------|------|
| 图像描述 | COCO Captions | CIDEr, BLEU-4 | 基础描述能力 |
| 视觉问答 | VQAv2 / GQA | Accuracy | 通用视觉理解 |
| OCR/文本理解 | TextVQA | Accuracy | 细粒度视觉理解 |
| 指令遵循 | MMBench | Accuracy | 指令遵循能力 |
| 幻觉检测 | POPE | F1 Score | 幻觉控制能力 |

### 9.2 人工评估

- 图像描述准确性（1-5 分）
- 指令遵循度（1-5 分）
- 幻觉率（越低越好）
- 整体可用性（1-5 分）

### 9.3 对比基线

| 模型 | 参数量 | 参考性能 |
|------|--------|----------|
| LLaVA-1.5-7B | 7B | MMBench 58+ |
| InternVL2-1B | 1B | MMBench 60+ |
| **我们的 Qwen3-VL-0.6B** | **1.2B** | **目标: MMBench 45+** |

> 注：0.6B 基础模型能力有限，目标不是追赶 7B 模型，而是在同等参数量下达到最优效果

---

## 十、风险与应对（更新）

| 风险 | 影响 | 应对措施 | 状态 |
|------|------|----------|------|
| Qwen2/Qwen3 不兼容 | 无法使用 stage1 预训练权重 | 从头训练 Projector，~3-4h | ✅ 已识别，方案明确 |
| eos_token_id 错误 | 生成不停止 | 显式指定 eos_token_id=[151643, 151645] | ✅ 已修复 |
| 单层 Projector 性能差 | 对齐质量低 | 升级为 2-Layer MLP + Residual + LN | ✅ 已优化 |
| 训练数据不足 | 泛化能力弱 | 增加至 ~1.76M (S1) + ~4.25M (S2) | ✅ 已优化 |
| LoRA alpha 过低 | 视觉适应不足 | alpha/r=1.0, r=128 | ✅ 已优化 |
| Qwen3 thinking 干扰 | 训练信号泄漏 | 简化 chat template，禁用 thinking | ✅ 已识别 |
| 大数据集下载慢 | 流程阻塞 | 分步下载，优先核心数据集 | 🟡 需监控 |

---

## 十一、时间估算（更新）

| 阶段 | 操作 | 预计时间 |
|------|------|----------|
| 环境准备 | 安装依赖、创建结构 | 10 分钟 |
| 代码下载+修改 | 下载 siq-vl + Qwen2→Qwen3 适配 | 30 分钟 |
| 模型下载 | SigLIP2 权重 | 20 分钟 |
| 数据下载 | LLaVA-Pretrain + Instruct | 30-60 分钟 |
| Stage 1 训练 | 10000 steps (单卡 H20) | 3-4 小时 |
| Stage 2 训练 | 2 epochs (单卡 H20) | 6-10 小时 |
| 评估 + Demo | 测试 + Gradio 界面 | 1-2 小时 |
| **总计** | | **~12-18 小时** |
