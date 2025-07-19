import torch
from torch import Tensor


class ScaledDotProductAttention(torch.nn.Module):
    def forward(self, q_proj: Tensor, k_proj: Tensor, v_proj: Tensor):
        # q_proj: (batch, seq_len, dim_model) => (bqd)
        # k_proj: (batch, seq_len, dim_model) => (bkd)
        # v_proj: (batch, seq_len, dim_model) => (bvd) or (bkd)
        logits = torch.einsum("bqd, bkd -> bqk", q_proj, k_proj)
        # OR
        # logits = torch.bmm(q_proj, k_proj.transpose(-1, -2))
        dim_k = k_proj.shape[-1]
        logits /= dim_k**0.5
        attention = torch.softmax(logits, dim=-1)
        output = torch.einsum("bqk, bkd -> bqd", attention, v_proj)
        # OR
        # output = torch.bmm(attention, v_proj)
        return output
