import torch

from attention.layernorm import LayerNorm
from attention.multi_head_attention import MultiHeadAttention
from transformers.encoder import FeedForwardNetwork


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048):
        super().__init__()
        self.mmha = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, causal_mask=True
        )
        self.layer_norm_1 = LayerNorm(d_model=d_model)
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm_2 = LayerNorm()
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
        self.layer_norm_3 = LayerNorm(d_model=d_model)

    def forward(
        self, in_embeddings: torch.Tensor, encoder_out: torch.Tensor
    ) -> torch.Tensor:
        mmha_out = self.mmha(in_embeddings, in_embeddings, in_embeddings)
        norm_out_1 = self.layer_norm_1(mmha_out + in_embeddings)
        mha_out = self.mha(encoder_out, encoder_out, norm_out_1)
        norm_out_2 = self.layer_norm_2(mha_out + norm_out_1)
        ffn_out = self.ffn(norm_out_2)
        output = self.layer_norm_3(ffn_out + norm_out_2)
        return output
