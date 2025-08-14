import torch
from torch import Tensor


class FlashAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size_q = 4
        self.block_size_kv = 4
        self.block_size_v = 4

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
        batch_size, seq_len_q, d_model = q_proj.shape
        n_blocks_q = torch.ceil(seq_len_q / self.block_size_q)
        n_blocks_kv = torch.ceil(k_proj.shape[1] / self.block_size_kv)

        output = torch.zeros((batch_size, seq_len_q, d_model))
        rowsum = torch.zeros((batch_size, seq_len_q))
        rowmax = torch.empty((batch_size, seq_len_q)).fill_(-torch.inf)

        for block_kv_ix in range(int(n_blocks_kv)):
            start_kv = block_kv_ix * self.block_size_kv
            end_kv = min(start_kv + self.block_size_kv, k_proj.shape[1])

            # Process the key block
            k_block = k_proj[:, start_kv:end_kv, :]
            #  Shape: (batch_size, block_size_kv, d_model)
            v_block = v_proj[:, start_kv:end_kv, :]
            #  Shape: (batch_size, block_size_kv, d_model)
            for block_q_ix in range(int(n_blocks_q)):
                start_q = block_q_ix * self.block_size_q
                end_q = min(start_q + self.block_size_q, seq_len_q)

                # Process the query block
                q_block = q_proj[:, start_q:end_q, :]
                #  Shape: (batch_size, block_size_q, d_model)

                logits = torch.einsum("bqd, bkd -> bqk", q_block, k_block)
                # Shape: (batch_size, block_size_q, block_size_kv)

                dim_k = k_block.shape[-1]  # basically d_model
                logits /= dim_k**0.5

                ########## Calculate Online Softmax ##########
                block_rowmax = torch.max(logits, dim=-1)
                #  Shape: (batch_size, block_size_q)

                block_P = torch.exp(logits - block_rowmax)
                #  Shape: (batch_size, block_size_kv, d_model)

                block_rowsum = torch.sum(block_P, dim=-1)
                # Shape: (batch_size, block_size_q)

                block_rowmax_old = rowmax[:, start_q:end_q]
                # Shape: (batch_size, block_size_q)

                block_rowmax_new = rowmax[:, start_q:end_q] = torch.maximum(
                    block_rowmax_old, block_rowmax
                )
                # block_rowmax_new = rowmax[:, start_q:end_q]
                # Shape: (batch_size, block_size_q)

                block_rowsum_old = rowsum[:, start_q:end_q]
                # Shape: (batch_size, block_size_q)
                block_rowsum_new = (
                    torch.exp(block_rowmax_old - block_rowmax_new)
                    * block_rowsum_old
                    + torch.exp(block_rowmax - block_rowmax_new) * block_rowsum
                )
                # Optional: dropout
                # block_P -> dropout
                partial_out = torch.einsum("bqk, bkd -> bqd", block_P, v_block)
                # Shape: (batch_size, block_size_q, d_model)

                block_out_old = output[:, start_q:end_q, :]
                # Shape: (batch_size, block_size_q, d_model)

                output[:, start_q:end_q, :] = (
                    (
                        block_rowsum_old
                        * block_out_old
                        * torch.exp(block_rowmax_old - block_rowmax_new)
                    )
                    + (
                        partial_out
                        * torch.exp(block_rowmax - block_rowmax_new)
                    )
                ) / block_rowsum_new
        return output
