import torch

from attention.layernorm import LayerNorm
from attention.multi_head_attention import MultiHeadAttention
from attention.projection import Projection


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = Projection(d_model, d_ff)
        self.activation = torch.nn.ReLU()
        self.linear2 = Projection(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> (batch_size, seq_len, d_model)
        x = self.linear1(x)  # (batch_size, seq_len, d_ff)
        x = self.activation(x)  # (batch_size, seq_len, d_ff)
        x = self.linear2(x)  # (batch_size, seq_len, d_model)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm_1 = LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
        self.layer_norm_2 = LayerNorm(d_model)

    def forward(self, in_embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings -> (batch_size, seq_len, d_model)
        # in_embeddings
        mha_out = self.mha(in_embeddings, in_embeddings, in_embeddings)
        norm_out = self.layer_norm_1(mha_out + in_embeddings)
        ffn_out = self.ffn(norm_out)
        output = self.layer_norm_2(ffn_out + norm_out)
        return output
