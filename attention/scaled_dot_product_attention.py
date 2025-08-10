import torch
from torch import Tensor


class ScaledDotProductAttention(torch.nn.Module):
    """
    Computes Scaled Dot-Product Attention.

    This is a critical component of the Transformer model, as described in the
    "Attention is All You Need" paper. It calculates attention scores for a set
    of queries, keys, and values.

    The formula is as follows:
    Attention(Q, K, V) = softmax( (Q @ K.T) / sqrt(d_k) ) @ V
    where:
    - Q: Queries tensor
    - K: Keys tensor
    - V: Values tensor
    - d_k: The dimension of the keys.

    Args:
        q_proj (Tensor): The queries tensor of shape
            (batch, seq_len, dim_model).
        k_proj (Tensor): The keys tensor of shape (batch, seq_len, dim_model).
        v_proj (Tensor): The values tensor of shape
            (batch, seq_len, dim_model).
        causal_mask (bool, optional): If True, applies a causal mask to prevent
            attention to future positions. Defaults to False.

    Returns:
        Tensor: The output of the attention mechanism, with the same shape as
            the queries and values.
    """

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

        # Calculate the dot product of queries and keys
        logits = torch.einsum("bqd, bkd -> bqk", q_proj, k_proj)
        # OR
        # logits = torch.bmm(q_proj, k_proj.transpose(-1, -2))

        # Scale the logits by the square root of the key dimension
        batch_size, _, q_dim = q_proj.shape
        dim_k = k_proj.shape[-1]
        logits /= dim_k**0.5

        # Apply causal mask if required
        mask = torch.zeros(batch_size, q_dim, dim_k)
        if causal_mask:
            mask = torch.masked_fill(
                mask,
                mask=torch.ones_like(mask).tril().logical_not(),
                value=-torch.inf,
            )

        logits += mask

        # Apply softmax to get attention weights
        attention = torch.softmax(logits, dim=-1)

        # Multiply attention weights by values
        output = torch.einsum("bqk, bkd -> bqd", attention, v_proj)
        # OR
        # output = torch.bmm(attention, v_proj)
        return output
