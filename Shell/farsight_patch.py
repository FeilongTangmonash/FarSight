# farsight_patch.py
import math
import torch
import torch.nn.functional as F
import types


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary position embedding to query and key tensors."""
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def farsight_attention_forward_v2(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,         # 兼容 KV 缓存接口
    output_attentions=False,     # 是否需要返回注意力矩阵
    use_cache=False,             # 兼容 generate 传参；本实现不支持
    **kwargs
):
    """
    FarSight: W = (QK^T/√d) ⊙ C  +  P
              Â = softmax(W) ⊙ C
    - 上三角作为“寄存槽”，softmax 后再乘因果下三角保持因果性
    - padding-safe：只允许往未来的有效列泄露
    - 可选 ALiBi 头相关衰减 + 简单的 progressive 衰减
    注意：此版本未实现 past_key_value 的 KV cache；使用时建议 use_cache=False。
    """
    B, L, _ = hidden_states.size()
    dtype = hidden_states.dtype
    device = hidden_states.device

    # Q K V projections
    # Handle both standard attention and grouped-query attention (GQA)
    num_key_value_heads = getattr(self, 'num_key_value_heads', self.num_heads)
    
    q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(B, L, num_key_value_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(B, L, num_key_value_heads, self.head_dim).transpose(1, 2)

    # Apply rotary position embeddings (critical for LLaMA!)
    if hasattr(self, 'rotary_emb'):
        kv_seq_len = k.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        if position_ids is None:
            position_ids = torch.arange(L, device=device).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    
    # Handle grouped query attention (GQA) - repeat KV heads if needed
    num_key_value_groups = getattr(self, 'num_key_value_groups', 1)
    if num_key_value_groups > 1:
        k = k.repeat_interleave(num_key_value_groups, dim=1)
        v = v.repeat_interleave(num_key_value_groups, dim=1)

    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,L,L]

    # 因果下三角 C（两次用）
    C = torch.tril(torch.ones((L, L), device=device, dtype=dtype)).view(1, 1, L, L)

    # 上三角（未来）与距离
    idx = torch.arange(L, device=device)
    j_idx = idx.view(1, -1)
    i_idx = idx.view(-1, 1)
    delta = (j_idx - i_idx).to(dtype)
    upper = torch.triu(torch.ones((L, L), device=device, dtype=dtype), diagonal=1)

    # 基础线性衰减 P_basic：-(sigma * ReLU(j-i)) 仅上三角
    base_sigma = getattr(self, "sigma", float(math.log(256) / math.log(1024)))
    P_basic = -(base_sigma * F.relu(delta)) * upper  # [L,L]

    # 简单 progressive 衰减：随 i 递减 0.5
    prog_factor = 1.0 - (i_idx.to(dtype) / max(L - 1, 1)) * 0.5  # [L,1]
    P_prog = -(base_sigma * prog_factor) * F.relu(delta) * upper  # [L,L]

    # ALiBi 风格头相关斜率（可关）
    use_alibi = getattr(self, "farsight_use_alibi", True)
    if use_alibi:
        H = self.num_heads
        slopes = torch.tensor([2.0 ** (-8.0 * (h + 1) / H) for h in range(H)], device=device, dtype=dtype)  # [H]
        slopes = slopes.view(1, H, 1, 1)  # [1,H,1,1]
        P_alibi = -(F.relu(delta) * upper).to(dtype).view(1, 1, L, L) * slopes  # [1,H,L,L]
    else:
        P_alibi = None

    # ===== padding-safe：自适应解析 attention_mask（兼容 [B,L] / [B,1,1,L] / [B,1,L,L] / [B,L,L]） =====
    def _derive_valid_from_attention_mask(m, B, L, device, out_dtype):
        """
        输出 [B, L] 的有效列向量（1=有效token，0=padding）：
        - [B, L] 或 [B, 1, 1, L]（0/1 语义）：>0 为有效
        - [B, 1, L, L] / [B, L, L]（加性 0/-inf 语义）：看对角线，> -1e8 视为有效
        其余形状：保守全有效
        """
        if m.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            m_f = m.to(torch.float32)
        else:
            m_f = m

        if m.dim() == 2 and m.shape[-1] == L:
            valid = (m_f > 0).to(out_dtype)                              # [B,L]
        elif m.dim() == 3 and m.shape[-1] == L and m.shape[-2] == 1:
            valid = (m_f.squeeze(-2) > 0).to(out_dtype)                  # [B,L]
        elif m.dim() >= 3 and m.shape[-1] == L and m.shape[-2] == L:
            diag = torch.diagonal(m_f, dim1=-2, dim2=-1)                 # [B, ?, L]
            if diag.dim() == 3:
                diag = diag.squeeze(-2)                                  # [B,L]
            valid = (diag > -1e8).to(out_dtype)                          # [B,L]
        else:
            valid = torch.ones((B, L), device=device, dtype=out_dtype)   # [B,L]
        return valid

    if attention_mask is not None:
        valid_1d = _derive_valid_from_attention_mask(attention_mask, B, L, device, dtype)  # [B,L]
    else:
        valid_1d = torch.ones((B, L), device=device, dtype=dtype)                           # [B,L]

    # 按列/行扩展到与 P 的 [B,1,L,L] 兼容的形状
    valid_cols_4d = valid_1d.view(B, 1, 1, L)   # 作用于列 j
    valid_rows_4d = valid_1d.view(B, 1, L, 1)   # 可选作用于行 i（防止从 padding 行泄出）

    # 组合静态 P 并做 padding-safe（仍只对上三角有效，因为 P_basic/P_prog 自带 upper）
    P_static = (0.5 * P_basic + 0.5 * P_prog).to(dtype).view(1, 1, L, L)  # [1,1,L,L]
    P_static = (P_static * valid_cols_4d * valid_rows_4d).expand(B, self.num_heads, L, L)  # [B,H,L,L]

    if P_alibi is not None:
        P_alibi_b = (P_alibi * valid_cols_4d * valid_rows_4d).expand(B, self.num_heads, L, L)  # [B,H,L,L]
        P_total = P_static + P_alibi_b
    else:
        P_total = P_static
    # ===== padding-safe 结束 =====

    # 组装 W 与 softmax
    # FarSight formula: W = (QK^T/√d) ⊙ C + P
    # where C is causal mask (lower triangular 1s), P is FarSight upper triangular penalties
    
    # Create a large negative value for masking (using -1e9 instead of -inf to avoid NaN issues)
    NEG_INF = -1e9
    
    # For the lower triangular (causal): keep original attention scores
    # For the upper triangular (future): apply FarSight penalties (allows limited look-ahead)
    # The original code: attn_scores * C zeros out upper triangular completely,
    # then adds P_total which has penalties only in upper triangular.
    # This effectively gives: lower_tri = original_scores, upper_tri = penalties
    
    W = attn_scores * C + P_total
    
    # Apply attention mask for padding tokens if provided
    # The attention_mask from HuggingFace models is typically in additive format (0 for attend, -inf for mask)
    if attention_mask is not None:
        # Handle different attention mask formats from HuggingFace
        if attention_mask.dim() == 4 and attention_mask.shape[-2] == L and attention_mask.shape[-1] == L:
            # [B, 1, L, L] format - additive mask with 0 or -inf values
            W = W + attention_mask.to(dtype)
        elif attention_mask.dim() == 4 and attention_mask.shape[-2] == 1:
            # [B, 1, 1, L] format - broadcast over query dimension
            W = W + attention_mask.to(dtype)
        elif attention_mask.dim() == 2:
            # [B, L] format - binary mask (1 for attend, 0 for mask)
            # Convert to additive mask format [B, 1, 1, L]
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)
            additive_mask = (1.0 - expanded_mask) * NEG_INF
            W = W + additive_mask
    
    attn_probs = torch.softmax(W, dim=-1) * C

    # 输出
    context = torch.matmul(attn_probs, v)  # [B,H,L,D]
    context = context.transpose(1, 2).reshape(B, L, self.hidden_size)  # [B,L,HD]
    output = self.o_proj(context)                     # [B, L, HD]

    # 是否返回注意力矩阵（与 HF 接口一致）
    attn_weights = attn_probs if output_attentions else None

    # 本实现未做 KV cache，强制不返回 present_key_value
    present_key_value = None

    return output, attn_weights, present_key_value

class FarSightContext:
    """
    仅在 with 块内启用 FarSight 前向替换；训练态不启用。
    提示：此版本未处理 KV cache，建议 generate(use_cache=False)。
    """
    def __init__(self, model, decay_factor=None, use_alibi=True):
        self.model = model
        self.original_forwards = []
        self.decay_factor = decay_factor
        self.use_alibi = use_alibi

    def __enter__(self):
        if not self.model.training:
            for layer in self.model.model.layers:
                attn = layer.self_attn
                if not hasattr(attn, 'num_heads'):
                    hidden_size = attn.q_proj.in_features
                    head_dim = getattr(attn, 'head_dim', 128)
                    attn.num_heads = hidden_size // head_dim
                    attn.head_dim = head_dim
                    attn.hidden_size = hidden_size
                attn.sigma = self.decay_factor if self.decay_factor is not None else float(math.log(256) / math.log(1024))
                attn.farsight_use_alibi = bool(self.use_alibi)
                self.original_forwards.append(attn.forward)
                attn.forward = types.MethodType(farsight_attention_forward_v2, attn)
        return self.model

    def __exit__(self, exc_type, exc, tb):
        if self.original_forwards:
            for (layer, orig) in zip(self.model.model.layers, self.original_forwards):
                layer.self_attn.forward = orig
        self.original_forwards = []
