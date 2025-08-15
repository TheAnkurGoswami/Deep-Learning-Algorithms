import math

import torch
from torch import Tensor


class FlashAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size_q = 4
        self.block_size_kv = 4

    def forward(
        self,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for Flash Attention.

        Args:
            query (torch.Tensor): The query tensor of shape
                (batch_size, seq_len_q, d_model).
            key (torch.Tensor): The key tensor of shape
                (batch_size, seq_len_k, d_model).
            value (torch.Tensor): The value tensor of shape
                (batch_size, seq_len_v, d_model).

        Returns:
            torch.Tensor: The output tensor of shape
                (batch_size, seq_len_q, d_model).
        """

        #  Assuming no batch dimension for simplicity
        batch_size, seq_len, _ = q_proj.shape
        # Use ceiling division to get number of blocks
        n_blocks_q = int(math.ceil(seq_len / self.block_size_q))
        n_blocks_kv = int(math.ceil(seq_len / self.block_size_kv))

        # Initialize output tensor, and running stats for online softmax
        output = torch.zeros_like(q_proj)
        softmax_normalizers = torch.zeros((batch_size, seq_len, 1))
        max_logits = torch.full((batch_size, seq_len, 1), -torch.inf)

        # Outer loop over key/value blocks
        for block_kv_ix in range(n_blocks_kv):
            start_kv = block_kv_ix * self.block_size_kv
            end_kv = min(start_kv + self.block_size_kv, seq_len)

            k_block = k_proj[:, start_kv:end_kv, :]
            # Shape: (batch_size, block_size_kv, d_model)
            v_block = v_proj[:, start_kv:end_kv, :]
            # Shape: (batch_size, block_size_kv, d_model)

            # Inner loop over query blocks
            for block_q_ix in range(n_blocks_q):
                start_q = block_q_ix * self.block_size_q
                end_q = min(start_q + self.block_size_q, seq_len)

                q_block = q_proj[:, start_q:end_q, :]
                # Shape: (batch_size, block_size_q, d_model)

                logits = torch.einsum("bqd, bkd -> bqk", q_block, k_block)
                # Shape: (batch_size, block_size_q, block_size_kv)

                dim_k = k_block.shape[-1]  # basically d_model
                logits /= dim_k**0.5

                # Optionally, apply causal mask

                ########## Calculate Online Softmax ##########
                block_max_logit, _ = torch.max(logits, dim=-1, keepdim=True)
                # Shape: (batch_size, block_size_q, 1)

                # Numerically stable softmax for the current block
                attn_wt_unnormalized = torch.exp(logits - block_max_logit)
                # Shape: (batch_size, block_size_q, block_size_kv)

                block_softmax_normalizer = torch.sum(
                    attn_wt_unnormalized, dim=-1, keepdim=True
                )
                # Shape: (batch_size, block_size_q, 1)

                # Load old statistics
                prev_max_logit = max_logits[:, start_q:end_q, :]
                # Shape: (batch_size, block_size_q, 1)

                # Compute new max
                new_max_logit = torch.maximum(prev_max_logit, block_max_logit)
                # Shape: (batch_size, block_size_q, 1)

                # Update running statistics
                rescale_factor_prev = torch.exp(prev_max_logit - new_max_logit)
                rescale_factor_block = torch.exp(
                    block_max_logit - new_max_logit
                )

                block_rowsum_old = softmax_normalizers[:, start_q:end_q, :]
                # Shape: (batch_size, block_size_q, 1)
                block_rowsum_new = (rescale_factor_prev * block_rowsum_old) + (
                    rescale_factor_block * block_softmax_normalizer
                )

                # Update output

                # Optionally, apply dropout.

                output_block_prev = output[:, start_q:end_q, :]
                # Shape: (batch_size, block_size_q, d_model)

                output_block_curr = torch.einsum(
                    "bqk, bkd -> bqd", attn_wt_unnormalized, v_block
                )
                # Shape: (batch_size, block_size_q, d_model)

                numerator = (
                    block_rowsum_old * rescale_factor_prev * output_block_prev
                ) + (rescale_factor_block * output_block_curr)
                output[:, start_q:end_q, :] = numerator / block_rowsum_new

                # Store updated statistics
                max_logits[:, start_q:end_q, :] = new_max_logit
                softmax_normalizers[:, start_q:end_q, :] = block_rowsum_new
        return output
