'''
Q-Hitter: 通过稀疏-量化 KV Cache 优化 LLM 推理

本文件实现了 Q-Hitter 方法，通过结合注意力分数和量化友好度来选择 KV Cache 中的关键 tokens，
并对选中的 tokens 进行量化，从而在保持模型性能的同时显著减少内存占用。

主要组件：
1. QH2OKVCache: KV Cache 管理类，负责 token 选择和量化
2. QH2OLlamaAttention: 修改后的 LLaMA 注意力机制，集成 Q-Hitter 优化
3. QH2OLlamaForCausalLM: 修改后的 LLaMA 模型，替换所有注意力层
'''

import os
import sys
import pdb
import math
import copy
import types
import torch
from typing import Optional, Tuple

from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaForCausalLM,
)

__all__ = ['QH2OLlamaForCausalLM', 'QH2OLlamaAttention']

def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    创建因果掩码（Causal Mask），用于自回归模型的注意力机制
    
    因果掩码确保每个 token 只能看到它之前的 tokens，不能看到未来的 tokens。
    这对于生成任务（如文本生成）是必需的。
    
    Args:
        bsz: batch size（批次大小）
        tgt_len: 目标序列长度（当前生成的 tokens 数量）
        past_key_values_length: 过去 KV Cache 的长度（已生成的 tokens 数量）
        dtype: 数据类型
        device: 设备（CPU/GPU）
    
    Returns:
        形状为 [bsz, 1, tgt_len, tgt_len + past_key_values_length] 的掩码张量
        -inf 表示被掩码的位置（不能看到）
        0 表示可见的位置
    """
    # 创建下三角掩码矩阵，初始化为最小值（-inf）
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    # 将下三角部分（包括对角线）设置为 0，表示可见
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    # 如果有过去的 KV Cache，需要在前面添加全零掩码（因为过去的 tokens 对当前都可见）
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    # 扩展到 batch 维度
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    """
    应用旋转位置编码（RoPE, Rotary Position Embedding）到单个张量
    
    RoPE 是一种相对位置编码方法，通过旋转矩阵将位置信息编码到 query/key 向量中。
    这个函数是原始 apply_rotary_pos_emb 的修改版本，用于处理单个张量。
    
    Args:
        x: 输入张量，形状为 [bsz, num_heads, seq_len, head_dim]
        cos: 余弦值，形状为 [1, 1, seq_len, head_dim]
        sin: 正弦值，形状为 [1, 1, seq_len, head_dim]
        position_ids: 位置 ID，用于索引对应的 cos/sin 值
    
    Returns:
        应用了旋转位置编码后的张量
    """
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

## Simulation Quantizationl; Real Implementation are based on https://github.com/FMInference/H2O/blob/main/h2o_flexgen/flexgen/compression.py
def Quantization_simulated(w, n_bit=8, inplace=False):
    """
    模拟量化过程，将浮点数转换为 n_bit 整数，然后转换回浮点数
    
    使用对称量化（symmetric quantization）方法：
    1. 计算每个 token 的最大值和最小值
    2. 计算量化 scale
    3. 将浮点数量化为整数，再反量化回浮点数
    
    Args:
        w: 输入张量，形状为 [bsz, num_heads, num_tokens, head_dim] 或类似
        n_bit: 量化位数（如 4-bit, 8-bit）
        inplace: 是否原地修改输入张量
    
    Returns:
        w: 量化后的张量（浮点数形式）
        scales: 量化 scale，用于后续反量化
    """
    # BS, HEADS, TOKEN, DIMS
    org_w_shape = w.shape

    max_val = w.amax(dim=-1, keepdim=True)
    min_val = w.amin(dim=-1, keepdim=True)
    max_int = 2 ** n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                         zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w, scales


def Qerror(w, n_bit=8, inplace=False, normalize=False, norm=2):
    """
    计算量化误差（Quantization Error），用于评估某个 token 的量化友好度
    
    量化误差越小，说明该 token 越适合量化（量化后损失小）。
    这个指标用于 Q-Hitter 的 token 选择策略。
    
    Args:
        w: 输入张量，通常是 KV Cache 的一部分
        n_bit: 量化位数
        inplace: 是否原地修改
        normalize: 是否归一化误差到 [0, 1] 区间
        norm: 使用的范数类型（2 表示 L2 范数）
    
    Returns:
        error: 量化误差，形状为 [num_heads, num_tokens]
        如果 normalize=True，返回归一化后的误差（值域 [0, 1]）
    """
    # 保存原始值
    w_copy = copy.deepcopy(w)
    # 进行量化模拟
    qw, _ = Quantization_simulated(w, n_bit, inplace)
    # 计算量化前后的差异（L2 范数）
    error = (qw - w_copy).norm(p=norm, dim=-1)

    # 可选：归一化误差到 [0, 1] 区间
    if normalize:
        min_error, max_error = error.min(), error.max()
        # 避免除零
        if max_error - min_error > 1e-8:
            n_error = (error - min_error) / (max_error - min_error)
        else:
            n_error = error * 0  # 如果所有误差相同，返回零
        return n_error

    return error


class QH2OKVCache:
    """
    Q-Hitter KV Cache 管理类
    
    核心功能：
    1. 维护三种分数：注意力分数（hh_score）、量化友好度分数（q_score）、组合分数（combination_score）
    2. 根据组合分数选择要保留的 tokens（Heavy Hitters + Recent + Default）
    3. 对选中的 KV Cache 进行量化
    
    KV Cache 选择策略：
    - Heavy Hitters (hh_size): 根据组合分数选择的重要 tokens
    - Recent tokens (recent_size): 最近生成的 tokens（通常对当前生成很重要）
    - Default tokens (default_ratio): 默认保留的 tokens（作为保底）
    """
    def __init__(
        self,
        hh_size=0.2,
        recent_size=0,
        k_seq_dim=2,
        v_seq_dim=2,
        kbits=4,
        vbits=4,
        lambda_hh=1,
        default_ratio=0.07,
    ):  
        """
        初始化 Q-Hitter KV Cache
        
        Args:
            hh_size: Heavy Hitter 保留比例（相对于新生成的 tokens）
            recent_size: 最近 tokens 保留比例
            k_seq_dim: Key 张量的序列维度索引
            v_seq_dim: Value 张量的序列维度索引
            kbits: Key Cache 量化位数
            vbits: Value Cache 量化位数
            lambda_hh: 注意力分数在组合分数中的权重（alpha 参数）
            default_ratio: 默认保留的 tokens 比例（保底策略）
        """
        self.default_ratio = default_ratio
        # 计算实际的 Heavy Hitter 比例（减去 default_ratio，因为它们是分开计算的）
        self.hh_size_ratio = hh_size - self.default_ratio
        self.recent_size_ratio = recent_size
        # 总缓存大小比例
        self.cache_size_ratio = hh_size + recent_size + self.default_ratio
        
        # 这些值会在第一次调用时根据实际 token 数量计算
        self.cache_size = None  # 总缓存大小（token 数量）
        self.hh_size = None     # Heavy Hitter 数量
        self.default_size = None  # Default tokens 数量
        self.recent_size = None   # Recent tokens 数量
        
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_bits = kbits  # Key 量化位数
        self.v_bits = vbits  # Value 量化位数
        
        # 三种分数（会在 forward 过程中更新）
        self.hh_score = None          # 注意力分数（Heavy Hitter score）
        self.q_score = None          # 量化友好度分数（Quantization friendliness score）
        self.combination_score = None # 组合分数 = lambda_hh * hh_score + (1-lambda_hh) * q_score
        
        self.lambda_hh = lambda_hh  # 组合权重（alpha）

        assert self.cache_size_ratio > 0

    def __call__(self, past_key_values, attn_score_cache):
        """
        核心函数：处理 KV Cache，选择并量化关键 tokens
        
        Args:
            past_key_values: 过去的 KV Cache，元组 (key_states, value_states)
            attn_score_cache: 注意力分数缓存，形状为 [bsz, num_heads, q_len, kv_len]
        
        Returns:
            处理后的 KV Cache，只包含选中的 tokens
        """
        # 步骤 1: 更新三种分数
        self._update_hh_score(attn_score_cache)  # 更新注意力分数
        self._update_q_score(past_key_values)    # 更新量化友好度分数
        self._update_combination_score()          # 计算组合分数

        # 如果 KV Cache 为空，直接返回
        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0].size(self.k_seq_dim)
        # 如果序列长度还没超过缓存大小，不需要压缩，直接返回
        if seq_len <= self.cache_size:
            return past_key_values

        # hh-selection
        bsz, num_heads, _, head_dim = past_key_values[0].shape
        if self.default_size > 0:
            select_default_scores = self.hh_score[:, :seq_len - self.recent_size]
            _, keep_topk_default = torch.topk(select_default_scores, self.default_size, dim=-1)
            keep_topk_default = keep_topk_default.sort().values
        else:
            keep_topk_default = None

        if self.hh_size > 0:
            select_hh_scores = self.combination_score[:, :seq_len - self.recent_size]

            if keep_topk_default is not None:
                select_hh_scores = select_hh_scores.scatter(-1, keep_topk_default, -1)

            _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
            keep_topk = keep_topk.sort().values
        else:
            keep_topk = None

        if keep_topk_default is not None:
            keep_topk = torch.cat([keep_topk, keep_topk_default], dim=-1)

        if self.recent_size > 0:
            keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=past_key_values[0].device).repeat(num_heads, 1)
            if keep_topk is not None:
                keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)
            else:
                keep_idx = keep_recent
        else:
            keep_idx = keep_topk

        mask = torch.zeros(self.hh_score.shape, dtype=torch.bool).to(past_key_values[0].device)
        mask = mask.scatter(-1, keep_idx, 1)

        k_hh_recent = past_key_values[0].squeeze()[mask].view(bsz, num_heads, -1, head_dim)
        v_hh_recent = past_key_values[1].squeeze()[mask].view(bsz, num_heads, -1, head_dim)

        self.hh_score= self.hh_score[mask].view(num_heads, self.cache_size)
        self.q_score= self.q_score[mask].view(num_heads, self.cache_size)

        return (k_hh_recent, v_hh_recent)

    def _update_hh_score(self, attn_score_cache):
        """
        更新 Heavy Hitter 分数（注意力分数）
        
        Heavy Hitter 分数 = 每个 token 在所有 query 上的注意力分数之和
        分数越高，说明该 token 被关注得越多，越重要。
        
        Args:
            attn_score_cache: 注意力分数，形状为 [bsz, num_heads, q_len, kv_len]
        """
        num_new_tokens = attn_score_cache.shape[2]  # 新生成的 tokens 数量

        if self.hh_score is None:
            # set-up cache size
            self.hh_size = int(self.hh_size_ratio * num_new_tokens)
            self.recent_size = int(self.recent_size_ratio * num_new_tokens)
            self.default_size = int(self.default_ratio * num_new_tokens)
            self.cache_size = self.hh_size + self.recent_size + self.default_size

            self.hh_score = attn_score_cache.sum(0).sum(1)
        else:
            attn_score_cache = attn_score_cache.sum(0).sum(1)
            attn_score_cache[:, :-num_new_tokens] += self.hh_score
            self.hh_score = attn_score_cache

    ## Using raw quantization error
    def _update_q_score(self, past_key_values):
        """
        更新量化友好度分数（Quantization Friendliness Score）
        
        量化友好度 = 1 - 量化误差（归一化后）
        分数越高，说明该 token 越适合量化（量化后损失小）。
        
        这是 Q-Hitter 的核心创新：不仅考虑注意力分数，还考虑量化友好度。
        
        Args:
            past_key_values: KV Cache，元组 (key_states, value_states)
        """
        inplace_k = True
        inplace_v = True

        if self.q_score is None:
            kerror = Qerror(past_key_values[0][0], n_bit=self.k_bits, inplace=inplace_k, normalize=True)
            verror = Qerror(past_key_values[1][0], n_bit=self.v_bits, inplace=inplace_v, normalize=True)
            self.q_score = (2 - kerror - verror)/2
        else:
            kerror = Qerror(past_key_values[0][0,:,-1:,:], n_bit=self.k_bits, inplace=inplace_k, normalize=True)
            verror = Qerror(past_key_values[1][0,:,-1:,:], n_bit=self.v_bits, inplace=inplace_v, normalize=True)
            new_q_score = (2 - kerror - verror)/2
            self.q_score = torch.cat([self.q_score, new_q_score], dim=-1)

    def _update_combination_score(self):
        """
        计算组合分数（Combination Score）
        
        组合分数 = lambda_hh * 归一化注意力分数 + (1 - lambda_hh) * 量化友好度分数
        
        这是 Q-Hitter 的核心：同时考虑 token 的重要性和量化友好度。
        - lambda_hh (alpha) = 1: 只考虑注意力分数（退化为 H2O）
        - lambda_hh < 1: 同时考虑量化友好度（Q-Hitter）
        
        通过这个组合分数来选择既重要又适合量化的 tokens。
        """
        min_hh, max_hh = self.hh_score.min(), self.hh_score.max()
        self.combination_score = self.lambda_hh * (self.hh_score - min_hh) / (max_hh - min_hh) + (1 - self.lambda_hh) * self.q_score

    def _clean_scores(self):
        """
        清理所有分数和缓存大小
        
        在每个请求结束后调用，避免跨请求的干扰。
        """
        self.hh_score = None
        self.q_score = None
        self.combination_score = None
        self.default_size = None


class QH2OLlamaAttention(nn.Module):
    """
    Q-Hitter 版本的 LLaMA 注意力机制
    
    这是对标准 LlamaAttention 的修改版本，集成了 Q-Hitter KV Cache 优化。
    主要改动：
    1. 使用 QH2OKVCache 管理 KV Cache，实现稀疏化和量化
    2. 在 forward 过程中，对 KV Cache 进行动态选择和压缩
    3. 保持与原始 LLaMA 注意力机制的兼容性
    """

    def __init__(self, config: LlamaConfig):
        """
        初始化 Q-Hitter 注意力层
        
        Args:
            config: LLaMA 配置对象，包含模型架构参数和 Q-Hitter 参数
        """
        super().__init__()
        self.config = config
        
        # 从配置中读取标准 LLaMA 参数
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # 每个头的维度
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads  # GQA (Grouped Query Attention) 中的 KV heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 每个 KV head 对应的 Q heads 数量
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta  # RoPE 的基频率

        # 验证参数有效性
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # 定义投影层（与标准 LLaMA 相同）
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # 初始化旋转位置编码（RoPE）
        self._init_rope()

        # 创建 Q-Hitter KV Cache 管理器（核心组件）
        self.kv_cache = QH2OKVCache(
            hh_size=config.hh_size,          # Heavy Hitter 比例
            recent_size=config.recent_size,   # Recent tokens 比例
            k_seq_dim=2,                      # Key 的序列维度
            v_seq_dim=2,                      # Value 的序列维度
            kbits=config.kbits,               # Key 量化位数
            vbits=config.vbits,               # Value 量化位数
            lambda_hh=config.alpha            # 组合权重（alpha）
        )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _clean_cache(self):
        """
        清理 KV Cache 中的分数
        
        在每个请求结束后调用，重置所有分数和缓存大小。
        """
        self.kv_cache._clean_scores()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播函数
        
        这是 Q-Hitter 注意力机制的核心，与标准 LLaMA 注意力的主要区别在于：
        在计算注意力后，会使用 QH2OKVCache 对 KV Cache 进行选择和压缩。
        
        Args:
            hidden_states: 输入隐藏状态，形状 [bsz, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置 ID
            past_key_value: 过去的 KV Cache（用于增量生成）
            output_attentions: 是否输出注意力权重
            use_cache: 是否使用缓存
        
        Returns:
            attn_output: 注意力输出
            attn_weights: 注意力权重（如果 output_attentions=True）
            past_key_value: 处理后的 KV Cache（如果 use_cache=True）
        """
        bsz, q_len, _ = hidden_states.size()

        # 步骤 1: 计算 Query, Key, Value（支持张量并行）
        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # 步骤 2: 重塑 Q/K/V 为多头格式 [bsz, num_heads, seq_len, head_dim]
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # remake causal mask
        attention_mask = _make_causal_mask(
            bsz=bsz,
            tgt_len=q_len,
            past_key_values_length=past_key_value[0].shape[-2] if past_key_value is not None else 0,
            dtype=query_states.dtype,
            device=query_states.device,
        )

        # 步骤 4: 计算 KV 序列长度（包括过去的缓存）
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        # 步骤 5: 准备旋转位置编码（RoPE）
        position_length = kv_seq_len
        if not position_ids.nelement() > 1:
            if position_length < position_ids.item()+1:
                position_length = position_ids.item()+1

        cos, sin = self.rotary_emb(value_states, seq_len=position_length)

        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
        key_states = apply_rotary_pos_emb_single(key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # key/value are already rotated
        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 步骤 9: 计算注意力分数 Q @ K^T / sqrt(head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # 核心步骤：使用 Q-Hitter 处理 KV Cache
        # 这里会进行 token 选择和压缩，返回处理后的 KV Cache
        # attn_weights 用于计算 Heavy Hitter 分数
        past_key_value = self.kv_cache(past_key_value, attn_weights.detach().clone())

        # 使用处理后的注意力权重计算输出
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QH2OLlamaForCausalLM(LlamaForCausalLM):
    """
    Q-Hitter 版本的 LLaMA 因果语言模型
    
    这是对标准 LlamaForCausalLM 的修改版本，将所有注意力层替换为 QH2OLlamaAttention。
    这样整个模型在推理时都会使用 Q-Hitter 优化。
    """
    def __init__(self, config):
        """
        初始化 Q-Hitter LLaMA 模型
        
        Args:
            config: LLaMA 配置对象，必须包含 Q-Hitter 相关参数：
                - hh_size: Heavy Hitter 比例
                - recent_size: Recent tokens 比例
                - kbits: Key 量化位数
                - vbits: Value 量化位数
                - alpha: 组合权重
        """
        # 先调用父类初始化，加载标准 LLaMA 模型
        super().__init__(config)
        
        # 替换所有层的注意力机制为 Q-Hitter 版本
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            # 将标准 LlamaAttention 替换为 QH2OLlamaAttention
            self.model.layers[layer_idx].self_attn = QH2OLlamaAttention(config)



