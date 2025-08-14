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
        rowsum = torch.zeros((batch_size, seq_len, 1))
        rowmax = torch.full((batch_size, seq_len, 1), -torch.inf)

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
                block_rowmax, _ = torch.max(logits, dim=-1, keepdim=True)
                # Shape: (batch_size, block_size_q, 1)

                # Numerically stable softmax for the current block
                block_P = torch.exp(logits - block_rowmax)
                # Shape: (batch_size, block_size_q, block_size_kv)

                block_rowsum = torch.sum(block_P, dim=-1, keepdim=True)
                # Shape: (batch_size, block_size_q, 1)

                # Load old statistics
                block_rowmax_old = rowmax[:, start_q:end_q, :]
                # Shape: (batch_size, block_size_q, 1)

                # Compute new max
                block_rowmax_new = torch.maximum(
                    block_rowmax_old, block_rowmax
                )
                # Shape: (batch_size, block_size_q, 1)

                # Update running statistics
                exp_diff_old = torch.exp(block_rowmax_old - block_rowmax_new)
                exp_diff_new = torch.exp(block_rowmax - block_rowmax_new)

                block_rowsum_old = rowsum[:, start_q:end_q, :]
                # Shape: (batch_size, block_size_q, 1)
                block_rowsum_new = (exp_diff_old * block_rowsum_old) + (
                    exp_diff_new * block_rowsum
                )

                # Update output

                # Optionally, apply dropout.

                block_out_old = output[:, start_q:end_q, :]
                # Shape: (batch_size, block_size_q, d_model)

                partial_out = torch.einsum("bqk, bkd -> bqd", block_P, v_block)
                # Shape: (batch_size, block_size_q, d_model)

                numerator = (
                    block_rowsum_old * exp_diff_old * block_out_old
                ) + (exp_diff_new * partial_out)
                output[:, start_q:end_q, :] = numerator / block_rowsum_new

                # Store updated statistics
                rowmax[:, start_q:end_q, :] = block_rowmax_new
                rowsum[:, start_q:end_q, :] = block_rowsum_new
        return output
