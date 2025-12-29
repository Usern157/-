# scripts/char_adapter.py
import torch
import torch.nn as nn
import os
from collections import defaultdict


def load_decompose_map(src_dir: str):
    """从 radical_table.txt 和 once_table.txt 构建字符拆解映射"""
    decompose_map = {}

    # 加载 radical_table
    path1 = os.path.join(src_dir, "radical_table.txt")
    if os.path.exists(path1):
        with open(path1, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    char = parts[0]
                    decompose_map[char] = parts[1:]

    # 加载 once_table（补充或覆盖）
    path2 = os.path.join(src_dir, "once_table.txt")
    if os.path.exists(path2):
        with open(path2, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    char = parts[0]
                    decompose_map[char] = parts[1:]  # 覆盖更细粒度

    return decompose_map


class CharacterAwareAdapter(nn.Module):
    def __init__(self, decompose_map, embed_dim=256, hidden_size=1536, device="cuda"):
        super().__init__()
        self.decompose_map = decompose_map
        self.hidden_size = hidden_size
        self.device = device

        # 收集所有部件
        all_comps = set()
        for comps in decompose_map.values():
            for c in comps:
                if len(c) == 1:  # 只保留单字符部件
                    all_comps.add(c)
        self.comp_to_id = {c: i for i, c in enumerate(sorted(all_comps))}
        self.num_comps = len(self.comp_to_id)
        print(f"[CAA] Loaded {len(decompose_map)} chars, {self.num_comps} unique components.")

        # 可学习的部件 embedding 表
        self.comp_embed = nn.Embedding(self.num_comps, embed_dim)
        self.fusion_proj = nn.Linear(embed_dim, hidden_size)

        # 预计算 char -> comp_ids 张量（CPU 缓存）
        self.char_to_comp_ids = {}
        for char, comps in decompose_map.items():
            ids = []
            for c in comps:
                if c in self.comp_to_id:
                    ids.append(self.comp_to_id[c])
            if ids:
                self.char_to_comp_ids[char] = torch.tensor(ids, dtype=torch.long)
            else:
                self.char_to_comp_ids[char] = None  # 无法分解

    def get_component_repr(self, char: str) -> torch.Tensor:
        """返回该汉字的部件融合表示 (hidden_size,)"""
        if char not in self.char_to_comp_ids or self.char_to_comp_ids[char] is None:
            return torch.zeros(self.hidden_size, device=self.device)

        comp_ids = self.char_to_comp_ids[char].to(self.device)  # (n,)
        emb = self.comp_embed(comp_ids)  # (n, embed_dim)
        mean_emb = emb.mean(dim=0)  # (embed_dim,)
        fused = self.fusion_proj(mean_emb)  # (hidden_size,)
        return fused

    def inject_at_positions(self, hidden_states: torch.Tensor, target_chars: list, answer_positions: list):
        """
        在指定位置注入部件表示
        - hidden_states: (B, L, H)
        - target_chars: [char1, char2, ...] 长度 B
        - answer_positions: [pos1, pos2, ...] 长度 B
        """
        for i, (char, pos) in enumerate(zip(target_chars, answer_positions)):
            if pos < hidden_states.size(1):
                comp_repr = self.get_component_repr(char)
                hidden_states[i, pos] += comp_repr
        return hidden_states