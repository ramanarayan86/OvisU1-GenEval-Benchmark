# Minimal fallback adapter using PyTorch SDPA
import torch
import torch.nn.functional as F

def make_flash_attn_kwargs(is_causal: bool = False,
                           dropout_p: float = 0.0,
                           softmax_scale: float | None = None,
                           **kwargs):
    # Map to torch SDPA kwargs
    return dict(is_causal=is_causal, dropout_p=dropout_p, scale=softmax_scale)

def flash_attn_varlen_func(q, k, v, attn_bias=None,
                           is_causal: bool = False,
                           dropout_p: float = 0.0,
                           softmax_scale: float | None = None,
                           **kwargs):
    """
    Minimal adapter. Treats inputs as standard [B, H, L, D] (or [B, L, H, D]) tensors
    and ignores varlen packing (seq offsets). Good enough to unblock imports;
    performance will rely on PyTorch SDPA/Flash kernels.
    """
    def _bhld(t):
        if t.dim() == 3:  # [B, L, D] -> [B,1,L,D]
            B, L, D = t.shape
            return t.view(B, 1, L, D)
        if t.dim() == 4 and t.shape[1] != t.shape[2]:  # [B, L, H, D] -> [B, H, L, D]
            return t.permute(0, 2, 1, 3).contiguous()
        return t  # assume [B,H,L,D]
    q, k, v = _bhld(q), _bhld(k), _bhld(v)

    return F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_bias,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=softmax_scale
    )

