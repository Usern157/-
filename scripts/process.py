# scripts/process.py
import os
import json
import random
from collections import defaultdict

RAW_DATA_DIR = "data/raw/src_data"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# Step 1: 构建汉字拆解图谱
# ======================
decompose_map = {}

# 加载 radical_table（一级拆解）
with open(os.path.join(RAW_DATA_DIR, "radical_table.txt"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            char = parts[0]
            components = parts[1:]
            decompose_map[char] = components

# 加载 once_table（二级拆解，用于丰富线索）
with open(os.path.join(RAW_DATA_DIR, "once_table.txt"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            char = parts[0]
            # 如果该字未在 radical_table 中出现，也加入
            if char not in decompose_map:
                decompose_map[char] = parts[1:]

# ======================
# Step 2: 加载合法汉字集合（来自 word.json）
# ======================
valid_chars = set()
with open(os.path.join(RAW_DATA_DIR, "word.json"), "r", encoding="utf-8") as f:
    words = json.load(f)
    for item in words:
        if "word" in item and len(item["word"]) == 1:
            valid_chars.add(item["word"])

print(f"Loaded {len(valid_chars)} valid single-character words.")
print(f"Decomposition map size: {len(decompose_map)}")


# ======================
# Step 3: 生成结构化线索
# ======================
def generate_clue(riddle: str, answer: str) -> str:
    clues = []

    # 1. 若答案可拆解，加入拆字信息
    if answer in decompose_map:
        comps = decompose_map[answer]
        clues.append("拆字：" + " + ".join(comps))

        # 尝试二级拆解（若部件也在 map 中）
        sub_clues = []
        for comp in comps:
            if comp in decompose_map:
                sub_comps = decompose_map[comp]
                sub_clues.append(f"{comp}→{''.join(sub_comps)}")
        if sub_clues:
            clues.append("；".join(sub_clues))

    # 2. 检查谜面是否包含常见部件（简单关键词匹配）
    common_components = ["口", "木", "日", "月", "水", "火", "土", "人", "手", "目", "牛", "马", "村", "寨", "旧", "改"]
    found = [c for c in common_components if c in riddle]
    if found:
        clues.append(f"谜面含：{'、'.join(found)}")

    # 3. 默认兜底
    if not clues:
        clues.append("无明确结构线索")

    return "；".join(clues)


# ======================
# Step 4: 读取 CC-Riddle.jsonl 并过滤
# ======================
samples_by_split = defaultdict(list)

with open("data/raw/CC-Riddle.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        answer = item["answer"]
        riddle = item["question"]
        split = item["split"]

        # 过滤：仅保留单字答案且在合法词表中
        if len(answer) == 1 and answer in valid_chars:
            clue = generate_clue(riddle, answer)
            samples_by_split[split].append({
                "riddle": riddle,
                "answer": answer,
                "clue": clue
            })

# ======================
# Step 5: 保存各 split
# ======================
for split_name, samples in samples_by_split.items():
    # 若原始数据无 validation/test，可手动划分 train
    if split_name == "train" and len(samples_by_split) == 1:
        n = len(samples)
        val_n = max(100, int(0.05 * n))
        test_n = max(100, int(0.05 * n))
        random.seed(42)
        random.shuffle(samples)
        test_samples = samples[:test_n]
        val_samples = samples[test_n:test_n + val_n]
        train_samples = samples[test_n + val_n:]

        for name, data in [("train", train_samples), ("validation", val_samples), ("test", test_samples)]:
            with open(os.path.join(OUTPUT_DIR, f"{name}.json"), "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, ensure_ascii=False, indent=2)
        break
    else:
        with open(os.path.join(OUTPUT_DIR, f"{split_name}.json"), "w", encoding="utf-8") as out_f:
            json.dump(samples, out_f, ensure_ascii=False, indent=2)

print("✅ Data processing completed.")
for split in ["train", "validation", "test"]:
    path = os.path.join(OUTPUT_DIR, f"{split}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            print(f"{split}: {len(json.load(f))} samples")