import torch
from torch import Tensor


class ScaledDotProductAttention(torch.nn.Module):
    def forward(
        self,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        causal_mask: bool = False,
    ):
        # q_proj: (batch, seq_len, dim_model) => (bqd)
        # k_proj: (batch, seq_len, dim_model) => (bkd)
        # v_proj: (batch, seq_len, dim_model) => (bvd) or (bkd)
        logits = torch.einsum("bqd, bkd -> bqk", q_proj, k_proj)
        # OR
        # logits = torch.bmm(q_proj, k_proj.transpose(-1, -2))
        batch_size, _, q_dim = q_proj.shape
        dim_k = k_proj.shape[-1]
        logits /= dim_k**0.5

        mask = torch.zeros(batch_size, q_dim, dim_k)
        if causal_mask:
            mask = torch.masked_fill(
                mask,
                mask=torch.ones_like(mask).tril().logical_not(),
                value=-torch.inf,
            )

        logits += mask
        attention = torch.softmax(logits, dim=-1)
        output = torch.einsum("bqk, bkd -> bqd", attention, v_proj)
        # OR
        # output = torch.bmm(attention, v_proj)
        return output
