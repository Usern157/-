# scripts/evaluate.py
import os
import sys
import re
import torch
import json
import argparse
import random
import numpy as np
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from scripts.char_adapter import CharacterAwareAdapter, load_decompose_map

# === 路径配置 ===
MODEL_PATH = "./.hf_cache_Qwen3-0.6B"
FINETUNED_MODEL_DIR = "outputs/models/qwen3-riddle-caa-lora"
TEST_DATA_PATH = "data/processed/test.json"
SRC_DATA_DIR = "data/raw/src_data"
OUTPUT_DIR = "outputs/evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 生成配置 ===
REPETITION_PENALTY = 1.2

# === 模型包装器（参考 train.py） ===
class QwenWithCAA(torch.nn.Module):
    """
    将 Qwen 模型与 CAA (Character-Aware Adapter) 结合的包装器
    参考 train.py 的实现
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
        if target_chars is not None and labels is not None:
            answer_positions = []
            for label_row in labels:
                non_ignore = (label_row != -100).nonzero(as_tuple=True)[0]
                pos = non_ignore[-1].item() if len(non_ignore) > 0 else label_row.size(0) - 1
                answer_positions.append(pos)
            hidden_states = self.caa.inject_at_positions(hidden_states, target_chars, answer_positions)

        logits = self.model.lm_head(hidden_states)
        loss = None

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

    def generate(self, input_ids=None, **kwargs):
        """委托给基础模型的generate方法"""
        if input_ids is not None:
            return self.model.generate(input_ids=input_ids, **kwargs)
        else:
            return self.model.generate(**kwargs)


def load_models():
    """加载原始模型和微调后的模型，参考 train.py 的实现方式"""
    # 检查必要的路径
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型路径不存在: {MODEL_PATH}")
    if not os.path.exists(FINETUNED_MODEL_DIR):
        raise FileNotFoundError(f"微调模型目录不存在: {FINETUNED_MODEL_DIR}")
    if not os.path.exists(SRC_DATA_DIR):
        raise FileNotFoundError(f"源数据目录不存在: {SRC_DATA_DIR}")
    
    print("正在加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            fix_mistral_regex=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer 加载成功")
    except Exception as e:
        raise RuntimeError(f"加载 tokenizer 失败: {e}")

    print("正在加载原始模型...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto",
            use_cache=True
        )
        base_model.eval()
        print("✅ 原始模型加载成功")
    except Exception as e:
        raise RuntimeError(f"加载原始模型失败: {e}")

    print("正在加载微调后的模型...")
    try:
        finetuned_base = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto",
            use_cache=True
        )
    except Exception as e:
        raise RuntimeError(f"加载微调模型失败: {e}")

    # 配置并加载LoRA（必须与训练时配置一致）
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    finetuned_model = get_peft_model(finetuned_base, lora_config)
    
    # 加载LoRA权重
    lora_dir = os.path.join(FINETUNED_MODEL_DIR, "lora")
    
    if os.path.exists(lora_dir):
        try:
            # 优先尝试使用 PeftModel.from_pretrained 加载
            adapter_config_path = os.path.join(lora_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                print(f"从 {lora_dir} 加载LoRA适配器...")
                finetuned_model = PeftModel.from_pretrained(finetuned_base, lora_dir)
                print("✅ LoRA适配器加载成功（使用PeftModel.from_pretrained）")
            else:
                # 手动加载权重
                weights_file = os.path.join(lora_dir, "pytorch_model.bin")
                if os.path.exists(weights_file):
                    print(f"从 {weights_file} 加载LoRA权重...")
                    state_dict = torch.load(weights_file, map_location="cpu")
                    missing_keys, unexpected_keys = finetuned_model.load_state_dict(state_dict, strict=False)
                    loaded_lora_keys = [k for k in state_dict.keys() 
                                      if any(lora_key in k for lora_key in ["lora_A", "lora_B", "lora_embedding"])]
                    if loaded_lora_keys:
                        print(f"✅ 已加载 {len(loaded_lora_keys)} 个LoRA参数")
                    else:
                        print("⚠️  警告: 未找到LoRA权重")
                else:
                    # 尝试从最新的checkpoint加载
                    checkpoint_dirs = []
                    for item in os.listdir(FINETUNED_MODEL_DIR):
                        checkpoint_path = os.path.join(FINETUNED_MODEL_DIR, item)
                        if os.path.isdir(checkpoint_path) and item.startswith("checkpoint-"):
                            checkpoint_dirs.append((item, checkpoint_path))
                    
                    if checkpoint_dirs:
                        checkpoint_dirs.sort(key=lambda x: int(x[0].split("-")[1]), reverse=True)
                        latest_checkpoint = checkpoint_dirs[0][1]
                        checkpoint_weights = os.path.join(latest_checkpoint, "pytorch_model.bin")
                        if os.path.exists(checkpoint_weights):
                            print(f"从 {checkpoint_weights} 加载LoRA权重...")
                            state_dict = torch.load(checkpoint_weights, map_location="cpu")
                            missing_keys, unexpected_keys = finetuned_model.load_state_dict(state_dict, strict=False)
                            loaded_lora_keys = [k for k in state_dict.keys() 
                                              if any(lora_key in k for lora_key in ["lora_A", "lora_B", "lora_embedding"])]
                            if loaded_lora_keys:
                                print(f"✅ 已加载 {len(loaded_lora_keys)} 个LoRA参数")
                            else:
                                print("⚠️  警告: 未找到LoRA权重")
                        else:
                            print(f"⚠️  警告: 未找到LoRA权重文件")
                    else:
                        print(f"⚠️  警告: 未找到LoRA权重文件")
        except Exception as e:
            print(f"❌ 加载LoRA权重时出错: {e}")
            import traceback
            traceback.print_exc()
            print("将使用未加载权重的LoRA模型进行评估")
    else:
        print(f"⚠️  警告: LoRA目录不存在: {lora_dir}")
    
    finetuned_model.eval()

    # 加载CAA适配器
    print("正在加载CAA适配器...")
    caa_path = os.path.join(FINETUNED_MODEL_DIR, "caa.bin")
    decompose_map = load_decompose_map(SRC_DATA_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    caa = CharacterAwareAdapter(
        decompose_map,
        embed_dim=256,
        hidden_size=finetuned_model.config.hidden_size,
        device=device
    )
    
    if os.path.exists(caa_path):
        caa.load_state_dict(torch.load(caa_path, map_location="cpu"))
        print(f"✅ 已加载CAA权重: {caa_path}")
    else:
        print(f"⚠️  警告: 未找到CAA权重文件 {caa_path}")
    
    caa = caa.to(device)
    caa.eval()
    
    # 包装模型（参考 train.py）
    model_with_caa = QwenWithCAA(finetuned_model, caa, tokenizer)
    model_with_caa.eval()

    return tokenizer, base_model, model_with_caa


def prepare_input(tokenizer, item):
    """准备模型输入，参考 train.py 的格式"""
    prompt = (
        "<|im_start|>system\n你是一个字谜专家。<|im_end|>\n"
        "<|im_start|>user\n"
        f"谜面：{item['riddle']}\n"
        f"线索：{item['clue']}\n"
        "请直接回答一个汉字。<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    # 编码输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    
    # 创建完整序列（包含答案）用于计算loss
    full_text = prompt + item['answer'] + "<|im_end|>"
    full_inputs = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    
    return inputs, full_inputs, item['answer']


def extract_answer(generated_text: str) -> str:
    """从生成的文本中提取答案汉字（改进版）"""
    # 保存原始文本用于调试
    original_text = generated_text
    
    # 清理生成的文本：移除特殊标记
    generated_text = generated_text.strip()
    
    # 🔧 关键修复：首先移除推理相关的XML标签及其内容（使用DOTALL模式）
    # 这样可以移除整个推理块，避免影响答案提取
    generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL)
    generated_text = re.sub(r'<reasoning>.*?</reasoning>', '', generated_text, flags=re.DOTALL)
    
    # 移除其他XML风格的标记（单个标签）
    generated_text = re.sub(r'<[^>]+>', '', generated_text)
    
    # 移除常见的特殊标记词
    special_markers = ['think', 'reasoning', 'redacted', 'assistant', 'user', 'system']
    for marker in special_markers:
        generated_text = generated_text.replace(marker, '')
    
    # 清理多余的空白行
    generated_text = re.sub(r'\n\s*\n+', '\n', generated_text)
    generated_text = generated_text.strip()
    
    # 提取所有中文字符
    chinese_chars = [char for char in generated_text if '\u4e00' <= char <= '\u9fff']
    
    if not chinese_chars:
        return ""
    
    prediction = ""
    
    # 策略1: 查找答案提示词后的第一个中文字符（更全面的模式）
    answer_patterns = [
        r'最终答案为[：:：]?\s*[：:：\n]*\s*([\u4e00-\u9fff])',  # "最终答案为："、"最终答案为：\n\n牛"
        r'最终答案[：:：]?\s*[：:：\n]*\s*([\u4e00-\u9fff])',  # "最终答案："
        r'答案是[：:：]?\s*([\u4e00-\u9fff])',  # "答案是："、"答案是：牛"
        r'答案[：:：]?\s*[：:：\n]*\s*([\u4e00-\u9fff])',  # "答案："、"答案：\n牛"
        r'为[：:：]?\s*([\u4e00-\u9fff])',  # "为："、"为 牛"
        r'应为[：:：]?\s*([\u4e00-\u9fff])',  # "应为："
        r'是[：:：]?\s*([\u4e00-\u9fff])',  # "是："
        r'：\s*([\u4e00-\u9fff])',  # "：牛"（冒号后的第一个汉字）
    ]
    
    for pattern in answer_patterns:
        matches = re.finditer(pattern, generated_text)
        for match in matches:
            char = match.group(1)
            # 确保提取的是单个汉字
            if len(char) == 1 and '\u4e00' <= char <= '\u9fff':
                prediction = char
                break
        if prediction:
            break
    
    # 策略2: 查找markdown格式的答案（**答案**、*答案*、【答案】等）
    if not prediction:
        markdown_patterns = [
            r'\*\*([\u4e00-\u9fff])\*\*',  # **牛**
            r'\*([\u4e00-\u9fff])\*',  # *牛*
            r'【([\u4e00-\u9fff])】',  # 【牛】
            r'\[([\u4e00-\u9fff])\]',  # [牛]
            r'（([\u4e00-\u9fff])）',  # （牛）
            r'\(([\u4e00-\u9fff])\)',  # (牛)
        ]
        for pattern in markdown_patterns:
            matches = re.finditer(pattern, generated_text)
            for match in matches:
                char = match.group(1)
                if len(char) == 1 and '\u4e00' <= char <= '\u9fff':
                    prediction = char
                    break
            if prediction:
                break
    
    # 策略3: 查找文本末尾的答案（通常答案在最后）
    if not prediction:
        # 获取文本的最后100个字符，答案通常在这里
        text_tail = generated_text[-100:] if len(generated_text) > 100 else generated_text
        tail_chinese = [char for char in text_tail if '\u4e00' <= char <= '\u9fff']
        if tail_chinese:
            # 定义分隔符集合（标点、空格、换行等）
            separators = set([' ', '\n', '\t', '：', ':', '，', ',', '。', '.', '、', '；', ';', 
                            '！', '!', '？', '?', '【', '[', '】', ']', '（', '(', '）', ')', 
                            '<', '|', '|', '>', '《', '》', '「', '」'])
            # 尝试找到最后一个独立的汉字（前后有分隔符）
            for i in range(len(text_tail) - 1, -1, -1):
                char = text_tail[i]
                if '\u4e00' <= char <= '\u9fff':
                    # 检查前后字符，确保是独立的汉字
                    prev_char = text_tail[i-1] if i > 0 else ' '
                    next_char = text_tail[i+1] if i < len(text_tail) - 1 else ' '
                    # 如果前后是分隔符，可能是答案
                    if prev_char in separators or next_char in separators:
                        prediction = char
                        break
            # 如果没找到独立的，就用最后一个汉字
            if not prediction and tail_chinese:
                prediction = tail_chinese[-1]
    
    # 策略4: 如果还是没找到，使用所有中文字符中的最后一个
    if not prediction and chinese_chars:
        prediction = chinese_chars[-1]
    
    # 策略5: 如果只有一个中文字符，直接返回它
    if not prediction and len(chinese_chars) == 1:
        prediction = chinese_chars[0]
    
    return prediction


def calculate_perplexity(model, tokenizer, inputs, labels):
    """计算困惑度"""
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        else:
            loss = outputs.loss
        if loss is not None:
            perplexity = torch.exp(loss).item()
        else:
            perplexity = None
    return perplexity


def generate_answer(model, tokenizer, inputs, max_new_tokens=1000, ground_truth=None):
    """生成答案（优化版）"""
    with torch.no_grad():
        # 获取设备
        if hasattr(model, 'model'):
            device = next(model.model.parameters()).device
        elif hasattr(model, 'parameters'):
            device = next(model.parameters()).device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        input_ids = inputs["input_ids"].to(device)
        input_length = input_ids.shape[1]
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # 调用模型的generate方法（优化生成参数）
        if hasattr(model, 'generate') and callable(getattr(model, 'generate', None)):
            generate_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.5,  # 降低temperature，使生成更稳定
                "top_p": 0.9,  # 添加nucleus sampling
                "repetition_penalty": REPETITION_PENALTY,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "no_repeat_ngram_size": 3,  # 避免重复的3-gram
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            
            generated_ids = model.generate(**generate_kwargs)
        else:
            raise AttributeError(f"模型 {type(model)} 没有 generate 方法")
        
        # 只解码新生成的部分
        new_tokens = generated_ids[0][input_length:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        # 提取答案（先清理再提取）
        # 移除特殊token标记，但保留其他内容用于提取
        cleaned_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        prediction = extract_answer(cleaned_text)
        
        # 如果第一次提取失败，尝试从原始文本中提取（可能包含特殊token）
        if not prediction:
            prediction = extract_answer(generated_text)
        
        # 如果还是失败，尝试从完整输出中提取（包含输入部分）
        if not prediction:
            full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # 只提取assistant部分之后的内容
            assistant_marker = "<|im_start|>assistant\n"
            if assistant_marker in full_output:
                assistant_part = full_output.split(assistant_marker)[-1]
                prediction = extract_answer(assistant_part)
        
        return prediction


def evaluate_model(model, tokenizer, test_data, model_name, use_caa=False):
    """评估模型"""
    print(f"\n正在评估 {model_name}...")
    
    correct = 0
    total = 0
    perplexities = []
    all_predictions = []
    all_ground_truths = []
    
    # 获取设备
    if hasattr(model, 'model'):
        device = next(model.model.parameters()).device
    elif hasattr(model, 'parameters'):
        device = next(model.parameters()).device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for item in tqdm(test_data, desc=f"评估{model_name}"):
        inputs, full_inputs, ground_truth = prepare_input(tokenizer, item)
        
        # 移动到正确的设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
        
        # 创建labels用于计算困惑度
        labels = full_inputs["input_ids"].clone()
        # 只计算assistant部分的loss
        prompt_length = inputs["input_ids"].shape[1]
        labels[:, :prompt_length] = -100
        
        # 计算困惑度
        try:
            if use_caa:
                # 对于带CAA的模型，需要传入target_chars
                outputs = model(
                    input_ids=full_inputs["input_ids"],
                    labels=labels,
                    target_chars=[ground_truth]
                )
            else:
                outputs = model(**full_inputs, labels=labels)
            
            if isinstance(outputs, dict):
                loss = outputs.get("loss")
            else:
                loss = outputs.loss
            
            if loss is not None:
                perplexity = torch.exp(loss).item()
                # 过滤inf和nan值
                if np.isfinite(perplexity) and not np.isnan(perplexity):
                    perplexities.append(perplexity)
        except Exception as e:
            print(f"计算困惑度时出错: {e}")
            perplexity = None
        
        # 生成答案
        try:
            prediction = generate_answer(model, tokenizer, inputs, ground_truth=ground_truth)
            
            all_predictions.append(prediction)
            all_ground_truths.append(ground_truth)
            
            if prediction == ground_truth:
                correct += 1
            total += 1
        except Exception as e:
            print(f"生成答案时出错: {e}")
            import traceback
            traceback.print_exc()
            all_predictions.append("")
            all_ground_truths.append(ground_truth)
            total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    # 过滤inf和nan值后计算平均困惑度
    valid_perplexities = [p for p in perplexities if np.isfinite(p) and not np.isnan(p)]
    avg_perplexity = np.mean(valid_perplexities) if valid_perplexities else None
    
    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_perplexity": avg_perplexity,
        "perplexities": perplexities,
        "predictions": all_predictions,
        "ground_truths": all_ground_truths
    }
    
    print(f"{model_name} 准确率: {accuracy:.4f} ({correct}/{total})")
    if avg_perplexity:
        print(f"{model_name} 平均困惑度: {avg_perplexity:.4f}")
    
    return results




def save_detailed_results(base_results, finetuned_results, save_dir):
    """保存详细结果到JSON文件"""
    results_summary = {
        "base_model": {
            "accuracy": base_results["accuracy"],
            "correct": base_results["correct"],
            "total": base_results["total"],
            "avg_perplexity": base_results["avg_perplexity"]
        },
        "finetuned_model": {
            "accuracy": finetuned_results["accuracy"],
            "correct": finetuned_results["correct"],
            "total": finetuned_results["total"],
            "avg_perplexity": finetuned_results["avg_perplexity"]
        },
        "improvement": {
            "accuracy_delta": finetuned_results["accuracy"] - base_results["accuracy"],
            "accuracy_relative_improvement": (finetuned_results["accuracy"] - base_results["accuracy"]) / base_results["accuracy"] * 100 if base_results["accuracy"] > 0 else 0
        }
    }
    
    # 保存预测结果（只保存前100个样本，避免文件过大）
    predictions_data = []
    for i in range(min(100, len(base_results["predictions"]))):
        predictions_data.append({
            "index": i,
            "ground_truth": base_results["ground_truths"][i],
            "base_prediction": base_results["predictions"][i],
            "finetuned_prediction": finetuned_results["predictions"][i],
            "base_correct": base_results["predictions"][i] == base_results["ground_truths"][i],
            "finetuned_correct": finetuned_results["predictions"][i] == finetuned_results["ground_truths"][i]
        })
    
    results_summary["sample_predictions"] = predictions_data
    
    save_path = os.path.join(save_dir, "evaluation_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存至: {save_path}")


def main():
    # 声明全局变量
    global MODEL_PATH, FINETUNED_MODEL_DIR, TEST_DATA_PATH, OUTPUT_DIR
    
    parser = argparse.ArgumentParser(description="评估微调后的模型")
    parser.add_argument("--num_samples", type=int, default=30, help="随机抽取的测试样本数量（默认30）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认42）")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="基础模型路径")
    parser.add_argument("--finetuned_dir", type=str, default=FINETUNED_MODEL_DIR, help="微调模型目录")
    parser.add_argument("--test_data", type=str, default=TEST_DATA_PATH, help="测试数据路径")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="输出目录")
    args = parser.parse_args()
    
    # 更新全局配置
    MODEL_PATH = args.model_path
    FINETUNED_MODEL_DIR = args.finetuned_dir
    TEST_DATA_PATH = args.test_data
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("开始模型评估")
    print("=" * 60)
    
    # 加载测试数据
    print(f"\n正在加载测试数据: {TEST_DATA_PATH}")
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"测试样本总数: {len(test_data)}")
    
    # 随机抽取指定数量的样本
    random.seed(args.seed)
    if len(test_data) > args.num_samples:
        sampled_data = random.sample(test_data, args.num_samples)
        print(f"随机抽取 {args.num_samples} 个样本（随机种子: {args.seed}）")
    else:
        sampled_data = test_data
        print(f"测试数据量少于请求数量，使用全部 {len(test_data)} 个样本")
    
    # 加载模型
    tokenizer, base_model, finetuned_model = load_models()
    
    # 评估原始模型
    base_results = evaluate_model(
        base_model, 
        tokenizer, 
        sampled_data, 
        "原始模型",
        use_caa=False
    )
    
    # 评估微调后的模型
    finetuned_results = evaluate_model(
        finetuned_model,
        tokenizer,
        sampled_data,
        "微调后模型",
        use_caa=True
    )
    
    # 保存详细结果
    save_detailed_results(base_results, finetuned_results, OUTPUT_DIR)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("评估总结")
    print("=" * 60)
    print(f"原始模型准确率: {base_results['accuracy']:.4f}")
    print(f"微调后模型准确率: {finetuned_results['accuracy']:.4f}")
    
    accuracy_delta = finetuned_results['accuracy'] - base_results['accuracy']
    if base_results['accuracy'] > 0:
        relative_improvement = (accuracy_delta / base_results['accuracy']) * 100
        print(f"准确率提升: {accuracy_delta:+.4f} ({relative_improvement:+.2f}%)")
    else:
        print(f"准确率提升: {accuracy_delta:+.4f} (原始模型准确率为0，无法计算相对改进)")
    
    if base_results['avg_perplexity'] is not None and finetuned_results['avg_perplexity'] is not None:
        print(f"原始模型平均困惑度: {base_results['avg_perplexity']:.4f}")
        print(f"微调后模型平均困惑度: {finetuned_results['avg_perplexity']:.4f}")
        print(f"困惑度变化: {finetuned_results['avg_perplexity'] - base_results['avg_perplexity']:+.4f}")
    
    print(f"\n所有结果已保存至: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
