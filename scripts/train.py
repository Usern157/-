# scripts/train.py
"""
训练脚本：使用 LoRA + CAA (Character-Aware Adapter) 微调 Qwen3 模型用于字谜任务

主要功能：
1. 加载并预处理字谜数据集
2. 使用 LoRA 进行参数高效微调
3. 使用 CAA 注入字符结构信息
4. 训练并保存模型

作者：NLP 项目组
日期：2024
"""
import os
import sys
import torch

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    WEIGHTS_NAME,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import numpy as np

from scripts.char_adapter import CharacterAwareAdapter, load_decompose_map

# === 路径配置 ===
MODEL_PATH = "./.hf_cache_Qwen3-0.6B"
DATA_DIR = "data/processed"
OUTPUT_DIR = "outputs/models/qwen3-riddle-caa-lora"
SRC_DATA_DIR = "data/raw/src_data"

# === 训练配置 ===
# 重复抑制参数（用于训练时避免学习重复模式，默认关闭）
# 注意：重复抑制主要应在生成时使用，训练时通常不需要
REPETITION_PENALTY = 1.0  # 1.0 表示不应用重复抑制，>1.0 表示惩罚重复

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 自定义模型包装器 ===
class QwenWithCAA(torch.nn.Module):
    """
    将 Qwen 模型与 CAA (Character-Aware Adapter) 结合的包装器
    
    在生成答案的位置注入字符结构信息，帮助模型更好地理解汉字结构
    """
    def __init__(self, base_model, caa_adapter, tokenizer):
        super().__init__()
        self.model = base_model
        self.caa = caa_adapter
        self.tokenizer = tokenizer

    def forward(self, input_ids, labels=None, target_chars=None, repetition_penalty=1.0, **kwargs):
        """前向传播，支持CAA注入"""
        # 获取 hidden states
        outputs = self.model.model(
            input_ids=input_ids,
            output_hidden_states=True,
            **kwargs
        )
        
        # 提取最后一层 hidden states
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs

        # 如果提供了 target_chars，注入 CAA 信息
        if target_chars is not None:
            answer_positions = []
            for label_row in labels:
                non_ignore = (label_row != -100).nonzero(as_tuple=True)[0]
                pos = non_ignore[-1].item() if len(non_ignore) > 0 else label_row.size(0) - 1
                answer_positions.append(pos)
            hidden_states = self.caa.inject_at_positions(hidden_states, target_chars, answer_positions)

        logits = self.model.lm_head(hidden_states)
        loss = None

        if labels is not None:
            # 🔧 关键修复：不要 shift！labels 与 input_ids 对齐（位置 t 的 label 是 xt）
            # 使用 ignore_index=-100 自动忽略 prompt 部分
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            # 🧪 调试：仅在第一个 batch 打印一次（避免刷屏）
            if not hasattr(self, "_debug_logged"):
                self._debug_logged = True
                num_valid_labels = (labels != -100).sum().item()
                total_tokens = labels.numel()
                print(f"[DEBUG] Batch label stats:")
                print(f"  - Total tokens: {total_tokens}")
                print(f"  - Valid (non -100) labels: {num_valid_labels}")
                print(f"  - Loss: {loss.item():.4f}")
                if num_valid_labels > 0:
                    # 解码前几个有效 token 用于验证
                    valid_mask = labels[0] != -100
                    if valid_mask.any():
                        valid_ids = labels[0][valid_mask]
                        decoded = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
                        print(f"  - Decoded valid labels (first sample): '{decoded}'")

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# === 数据加载函数 ===
def load_dataset(split):
    """加载训练/验证数据集"""
    data_path = os.path.join(DATA_DIR, f"{split}.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts, answers = [], []
    for item in data:
        prompt = (
            "<|im_start|>system\n你是一个字谜专家。<|im_end|>\n"
            "<|im_start|>user\n"
            f"谜面：{item['riddle']}\n"
            f"线索：{item['clue']}\n"
            "请直接回答一个汉字。<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{item['answer']}<|im_end|>"
        )
        texts.append(prompt)
        answers.append(item["answer"])
    
    print(f"✅ 已加载 {split} 数据集: {len(texts)} 个样本")
    return Dataset.from_dict({"text": texts, "answer_char": answers})

# === 数据标记化函数 ===
def tokenize_fn(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors=None
    )
    
    input_ids = tokenized["input_ids"]
    labels = []
    
    # 获取 <|im_end|> 的 token ID
    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if end_token_id == tokenizer.unk_token_id:
        # fallback: try encoding
        end_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]

    for ids in input_ids:
        label = [-100] * len(ids)
        try:
            # 从后往前找 <|im_end|>
            end_idx = -1
            for i in range(len(ids) - 1, -1, -1):
                if ids[i] == end_token_id:
                    end_idx = i
                    break
            if end_idx != -1 and end_idx >= 1:
                # 假设答案是 <|im_end|> 前的一个 token
                label[end_idx - 1] = ids[end_idx - 1]
                # 可选：也训练倒数第二个（防多字或空格）
                if end_idx >= 2:
                    label[end_idx - 2] = ids[end_idx - 2]
        except Exception as e:
            # 保持全 -100，表示忽略该位置
            pass
        labels.append(label)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "answer_char": examples["answer_char"]
    }

# === 主训练函数 ===
def main():
    """主训练函数"""
    print("=" * 60)
    print("开始训练流程")
    print("=" * 60)
    
    # 检查必要的目录和文件
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型路径不存在: {MODEL_PATH}")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"数据目录不存在: {DATA_DIR}")
    if not os.path.exists(SRC_DATA_DIR):
        raise FileNotFoundError(f"源数据目录不存在: {SRC_DATA_DIR}")
    
    # 🔧 修复1: tokenizer 加载时启用 regex 修复
    print("\n[1/6] 加载 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            fix_mistral_regex=True  # ← 关键！
        )
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer 加载成功")
    except Exception as e:
        raise RuntimeError(f"加载 tokenizer 失败: {e}")

    # 🔧 修复2: 使用 dtype 替代 torch_dtype
    print("[2/6] 加载基础模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            dtype=torch.float16,  # ← 替换 torch_dtype
            device_map="auto",
            use_cache=False
        )
        model.gradient_checkpointing_enable()
        print("✅ 基础模型加载成功")
    except Exception as e:
        raise RuntimeError(f"加载基础模型失败: {e}")

    # LoRA 配置（统一配置，确保训练和推理一致）
    # 注意：如需修改，请同步更新 infer.py 和 evaluate.py 中的配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    print(f"📌 LoRA 配置: r={lora_config.r}, alpha={lora_config.lora_alpha}, target_modules={lora_config.target_modules}")
    print("[3/6] 应用 LoRA 适配器...")
    model = get_peft_model(model, lora_config)
    print("✅ LoRA 适配器已应用")

    # CAA (Character-Aware Adapter) 初始化
    print("[4/6] 初始化 CAA 适配器...")
    try:
        decompose_map = load_decompose_map(SRC_DATA_DIR)
        print(f"✅ 已加载 {len(decompose_map)} 个字符的拆解映射")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        caa = CharacterAwareAdapter(
            decompose_map,
            embed_dim=256,
            hidden_size=model.config.hidden_size,
            device=device
        ).to(device)
        print("✅ CAA 适配器初始化成功")
    except Exception as e:
        raise RuntimeError(f"初始化 CAA 适配器失败: {e}")

    print("[5/6] 包装模型...")
    model_with_caa = QwenWithCAA(model, caa, tokenizer)
    model_with_caa.train()
    print("✅ 模型包装完成")

    # 加载数据
    print("[6/6] 加载数据集...")
    try:
        train_ds = load_dataset("train")
        val_ds_full = load_dataset("valid")
    except Exception as e:
        raise RuntimeError(f"加载数据集失败: {e}")

    # 🔧 优化：训练过程中只使用极少部分验证集（加快验证速度）
    # 只保留前 10 个样本用于训练过程中的验证（可根据需要调整）
    VAL_SUBSET_SIZE = 10
    if len(val_ds_full) > VAL_SUBSET_SIZE:
        val_ds = val_ds_full.select(range(VAL_SUBSET_SIZE))
        print(f"📊 验证集已缩减: {len(val_ds_full)} -> {len(val_ds)} 个样本（仅用于训练过程验证）")
    else:
        val_ds = val_ds_full
        print(f"📊 验证集大小: {len(val_ds)} 个样本")

    train_ds = train_ds.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        remove_columns=["text"]  # 保留 answer_char，因为训练时需要用到
    )
    val_ds = val_ds.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        remove_columns=["text"]  # 保留 answer_char，因为训练时需要用到
    )

    class CustomDataCollator:
        """
        自定义数据整理器，处理 answer_char 字段
        
        在标准的数据整理基础上，保留 answer_char 字段用于 CAA 注入
        """
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        
        def __call__(self, features):
            # 提取 answer_char（用于 CAA 注入）
            answer_chars = [f.pop("answer_char") for f in features]
            # 使用标准 collator 处理其他字段
            batch = self.collator(features)
            # 将 answer_chars 添加回批次
            batch["answer_char"] = answer_chars
            return batch
    
    def compute_metrics(eval_pred):
        """
        计算验证指标
        
        返回验证集上的损失、困惑度等指标
        """
        predictions, labels = eval_pred
        
        # predictions 是 logits，labels 是真实的 token IDs
        # 计算准确率：预测的 token ID 是否与真实标签匹配
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # 确保是 numpy 数组
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        # 获取预测的 token ID（argmax）
        if predictions.ndim == 3:
            # 标准格式：[batch_size, seq_len, vocab_size]
            pred_ids = np.argmax(predictions, axis=-1)
        else:
            # 如果形状不对，返回 0
            return {"eval_accuracy": 0.0}
        
        # 只计算非忽略位置（labels != -100）的准确率
        mask = labels != -100
        if mask.sum() > 0:
            correct = (pred_ids[mask] == labels[mask]).sum()
            total = mask.sum()
            accuracy = correct / total
        else:
            accuracy = 0.0
        
        return {
            "eval_accuracy": accuracy,
        }
    
    class CustomTrainer(Trainer):
        """
        自定义 Trainer，支持 CAA 注入
        
        在计算损失时，将 answer_char 传递给模型以进行 CAA 注入
        """
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            answer_chars = inputs.pop("answer_char", None)
            
            # 确保输入在正确的设备上（Trainer 通常会自动处理，但为了安全起见）
            input_ids = inputs["input_ids"]
            if isinstance(labels, torch.Tensor):
                # 确保 labels 在正确的设备上
                if labels.device != input_ids.device:
                    labels = labels.to(input_ids.device)
            
            # 传递 answer_chars 用于 CAA 注入
            outputs = model(
                input_ids=input_ids, 
                labels=labels, 
                target_chars=answer_chars,
                repetition_penalty=REPETITION_PENALTY
            )
            return (outputs["loss"], outputs) if return_outputs else outputs["loss"]
        
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """
            重写预测步骤，支持 CAA 注入和 answer_char 传递
            🔧 优化：减少内存使用，立即将 logits 移到 CPU
            """
            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)
            
            # 提取 answer_char（如果存在）
            answer_chars = inputs.pop("answer_char", None)
            labels = inputs.pop("labels", None) if has_labels else None
            
            # 前向传播
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    labels=labels,
                    target_chars=answer_chars,
                    repetition_penalty=REPETITION_PENALTY
                )
                loss = outputs["loss"] if has_labels else None
                logits = outputs["logits"]
            
            if prediction_loss_only:
                # 🔧 修复：如果只需要损失，立即释放 logits 以节省内存
                del logits
                torch.cuda.empty_cache()  # 清理 GPU 缓存
                return (loss, None, None)
            
            # 🔧 修复：立即将 logits 移到 CPU 以节省 GPU 内存
            logits = logits.detach().cpu()
            if labels is not None:
                labels = labels.detach().cpu()
            
            return (loss, logits, labels)
        
        def _save(self, output_dir: str = None, state_dict=None):
            """重写保存方法，禁用 safetensors 以处理权重共享问题"""
            from transformers.modeling_utils import unwrap_model
            
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取要保存的模型
            model_to_save = unwrap_model(self.model)
            
            # 保存模型配置
            if hasattr(model_to_save, 'config'):
                model_to_save.config.save_pretrained(output_dir)
            
            # 保存模型权重，使用 pickle 格式而不是 safetensors
            if state_dict is None:
                state_dict = model_to_save.state_dict()
            
            # 使用 torch.save 保存权重（pickle 格式）
            weights_file = os.path.join(output_dir, WEIGHTS_NAME)
            torch.save(state_dict, weights_file)
            
            # 保存 tokenizer（如果有）
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,  # 🔧 修复：减小评估批次大小以避免内存溢出
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=50,
        # 更频繁的验证：每 200 步验证一次（可根据实际情况调整）
        eval_strategy="steps",
        eval_steps=200,
        # 保存策略：保存最佳模型和定期保存
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,  # 只保留最近3个checkpoint，节省空间
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
        metric_for_best_model="eval_loss",  # 使用验证损失作为最佳模型指标
        greater_is_better=False,  # 损失越小越好
        fp16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # 🔧 修复：禁用 pin_memory 以节省内存
    )

    # 早停回调配置
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,  # 如果验证损失连续5次没有改善，则停止训练
        early_stopping_threshold=0.001,  # 改善幅度小于0.001认为没有改善
    )
    
    trainer = CustomTrainer(
        model=model_with_caa,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=CustomDataCollator(tokenizer),
        compute_metrics=compute_metrics,  # 添加指标计算
        callbacks=[early_stopping_callback],  # 添加早停回调
    )

    # 打印训练配置信息
    print("\n" + "=" * 60)
    print("训练配置信息")
    print("=" * 60)
    print(f"📊 验证策略: 每 {training_args.eval_steps} 步验证一次")
    print(f"💾 保存策略: 每 {training_args.save_steps} 步保存一次")
    print(f"📈 最佳模型指标: {training_args.metric_for_best_model}")
    print(f"⏹️  早停机制: patience={early_stopping_callback.early_stopping_patience}, threshold={early_stopping_callback.early_stopping_threshold}")
    print(f"📦 最多保留 {training_args.save_total_limit} 个 checkpoint")
    print(f"🔄 训练轮数: {training_args.num_train_epochs}")
    print("=" * 60)
    print("开始训练...")
    print("=" * 60)
    trainer.train()
    
    # 保存模型
    print("\n正在保存模型...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "lora"))
    torch.save(caa.state_dict(), os.path.join(OUTPUT_DIR, "caa.bin"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 模型已保存至: {OUTPUT_DIR}")
    print("✅ 训练完成！")

if __name__ == "__main__":
    main()