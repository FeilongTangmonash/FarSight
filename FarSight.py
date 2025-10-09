import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional, Tuple
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from typing import Optional, Tuple, Union
import copy
#from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding

class AttnAdapter(LlamaAttention):
    def __init__(self, config, layer_idx: int, *args, **kwargs):
        if hasattr(config, "text_config"):
            bias_flag = config.text_config.attention_bias
        else:
            bias_flag = getattr(config, "attention_bias", False)

        super().__init__(config)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias_flag)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias_flag)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias_flag)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias_flag)

        rope_base = getattr(config, "rope_theta", 10000)
        max_pos = getattr(config, "max_position_embeddings", 2048)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=max_pos, base=rope_base)

        #ALiBi slopes
        def get_slopes(n):
            m = 2.0 ** (-8.0 / n)
            slopes = torch.pow(m, torch.arange(1, n+1, dtype=torch.float32))
            if n & (n - 1) != 0:
                n_small = 2 ** math.floor(math.log2(n))
                m2 = 2.0 ** (-4.0 / n_small)
                extra_slopes = torch.pow(m2, torch.arange(1, 2*(n - n_small) + 1, 2, dtype=torch.float32))
                slopes = torch.cat([slopes[:n_small], extra_slopes])
            return slopes
        slopes = get_slopes(self.num_heads)
        self.register_buffer("alibi_slopes", slopes.view(self.num_heads, 1, 1), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_value: tuple = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, seq_len, _ = hidden_states.size()

        
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        
        kv_seq_len = key_states.size(-2)
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].size(-2)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        
        if past_key_value is not None:
            pk, pv = past_key_value
            key_states_raw   = torch.cat([pk, key_states], dim=2)
            value_states_raw = torch.cat([pv, value_states], dim=2)
        else:
            key_states_raw, value_states_raw = key_states, value_states

        
        present_key_value = (key_states_raw, value_states_raw) if use_cache else None

        #推理计算用的kv
        key_states   = repeat_kv(key_states_raw, self.num_key_value_groups)
        value_states = repeat_kv(value_states_raw, self.num_key_value_groups)

        #原始打分
        scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores_baseline = scores.clone()

        #一阶段：因果mask/核心识别/σ缩放/ALiBi/
        K_total = key_states.size(-2)
        C_full = torch.tril(torch.ones((K_total, K_total), device=scores.device, dtype=torch.bool), diagonal=0)
        C_mask = C_full[-seq_len:].unsqueeze(0).unsqueeze(0)
        neg_inf = torch.finfo(scores.dtype).min

        #核心 token 选择
        neg_inf = torch.finfo(scores.dtype).min

        #不可见位置做-inf
        scores_masked = scores.masked_fill(C_mask == 0, neg_inf)             

        #对“行/步”维度取列强度：任意一步的最大|qk|
        cols_strength = scores_masked.abs().amax(dim=-2, keepdim=True)        

        #用分位数近似τ_m（逐 head）
        tau = torch.quantile(cols_strength.float(), 0.98, dim=-1, keepdim=True)  
        core_mask = cols_strength >= tau                                       


        need_fallback = ~torch.isfinite(tau)
        if need_fallback.any():
            _, top_idx = torch.topk(cols_strength, k=1, dim=-1)              
            fb = torch.zeros_like(core_mask, dtype=torch.bool)
            fb.scatter_(-1, top_idx, True)
            core_mask = torch.where(need_fallback.unsqueeze(-1), fb, core_mask)

        full_core_mask = core_mask.expand(bsz, self.num_heads, seq_len, K_total)


        #σ缩放
        sigma = 0.9
        scale_mat = torch.full_like(scores, sigma)
        scale_mat = torch.where(full_core_mask, torch.ones_like(scale_mat), scale_mat)
        scores = scores * scale_mat

        #ALiBi
        idx_i = torch.arange(K_total - seq_len, K_total, device=scores.device).unsqueeze(1)
        idx_j = torch.arange(K_total, device=scores.device).unsqueeze(0)
        distance = (idx_j - idx_i).clamp(min=0).to(scores.dtype)
        alibi_bias = -distance.unsqueeze(0) * self.alibi_slopes

        alibi_lambda_core = 0.0
        alibi_lambda_noncore = 1.0
        alibi_lambda = torch.where(full_core_mask,
                                   torch.full_like(scores, alibi_lambda_core),
                                   torch.full_like(scores, alibi_lambda_noncore))
        scores = scores + alibi_lambda * alibi_bias

        scores_sm = scores.masked_fill(C_mask == 0, neg_inf)
        attn_probs_new = torch.softmax(scores_sm, dim=-1)

        scores_base_sm = scores_baseline.masked_fill(C_mask == 0, neg_inf)
        attn_probs_base = torch.softmax(scores_base_sm, dim=-1)

        #δ保护
        delta = 0.1
        attn_probs = torch.where(full_core_mask,
                                 (1.0 - delta) * attn_probs_new + delta * attn_probs_base,
                                 attn_probs_new)

        #输出
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        attn_weights = attn_probs if output_attentions else None
        return attn_output, attn_weights, present_key_value
