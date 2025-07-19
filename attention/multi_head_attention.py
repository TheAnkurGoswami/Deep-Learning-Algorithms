from typing import Optional

import torch
from torch import Tensor

from attention.projection import Projection
from attention.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_q: Optional[int] = None,
        dim_k: Optional[int] = None,
        dim_v: Optional[int] = None,
        add_bias: bool = True,
    ):
        super().__init__()
        self.d_model: int = d_model  # Store as int, not tensor
        self.num_heads: int = num_heads
        self.add_bias: bool = add_bias

        # If specific dimensions for Q, K, V are not provided,
        # default them to d_model
        self.dim_q: int = dim_q if dim_q is not None else d_model
        self.dim_k: int = dim_k if dim_k is not None else d_model
        self.dim_v: int = dim_v if dim_v is not None else d_model

        if not (
            self.dim_q % num_heads == 0
            and self.dim_k % num_heads == 0
            and self.dim_v % num_heads == 0
        ):
            raise ValueError(
                "Total dimensions for Q, K, V must be divisible by num_heads."
            )

        self.q_proj_layer = Projection(
            in_features=self.d_model,
            out_features=self.dim_q,
            add_bias=add_bias,
        )
        self.k_proj_layer = Projection(
            in_features=self.d_model,
            out_features=self.dim_k,
            add_bias=add_bias,
        )
        self.v_proj_layer = Projection(
            in_features=self.d_model,
            out_features=self.dim_v,
            add_bias=add_bias,
        )

        self.out_proj_layer = Projection(
            in_features=self.dim_v,
            out_features=self.d_model,
            add_bias=add_bias,
        )

    def get_head_size(self, param: str):
        match param:
            case "query":
                return self.dim_q // self.num_heads
            case "key":
                return self.dim_k // self.num_heads
            case "value":
                return self.dim_v // self.num_heads

    def forward(self, inputs: Tensor):
        # print("inputs", inputs)
        q_proj = self.q_proj_layer.forward(inputs)
        # print("q_w", self.q_proj_layer._weights)
        # print("q_proj", q_proj)
        k_proj = self.k_proj_layer.forward(inputs)
        # print("k_w", self.k_proj_layer._weights)
        # print("k_proj", k_proj)
        v_proj = self.v_proj_layer.forward(inputs)

        head_outputs = []

        for head_ix in range(self.num_heads):
            sliced_projs = []
            for param, proj_mat in zip(
                ["query", "key", "value"],
                [q_proj, k_proj, v_proj],
                strict=False,
            ):
                head_size = self.get_head_size(param)
                start, end = (head_size * head_ix, head_size * (head_ix + 1))
                sliced_projs.append(proj_mat[:, :, start:end])
            head_output = ScaledDotProductAttention().forward(*sliced_projs)
            head_outputs.append(head_output)

        concat_heads = torch.concat(head_outputs, dim=-1)
        out = self.out_proj_layer.forward(concat_heads)
        return out
