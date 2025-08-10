import torch

from attention.layernorm import LayerNorm
from attention.multi_head_attention import MultiHeadAttention
from attention.projection import Projection


class FeedForwardNetwork(torch.nn.Module):
    """
    Implements the Position-wise Feed-Forward Network (FFN).

    The FFN is applied to each position separately and identically. It consists of
    two linear transformations with a ReLU activation in between.

    The formula is:
    FFN(x) = max(0, x @ W1 + b1) @ W2 + b2

    This network is a key component of both the encoder and decoder layers in the
    Transformer model.
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Initializes the FeedForwardNetwork module.

        Args:
            d_model (int): The dimensionality of the input and output.
            d_ff (int): The dimensionality of the inner-layer.
        """
        super().__init__()
        self.linear1 = Projection(d_model, d_ff)
        self.activation = torch.nn.ReLU()
        self.linear2 = Projection(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Position-wise Feed-Forward Network.

        Args:
            x (torch.Tensor): The input tensor of shape
                (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor of the same shape as the input.
        """
        # x -> (batch_size, seq_len, d_model)
        x = self.linear1(x)  # (batch_size, seq_len, d_ff)
        x = self.activation(x)  # (batch_size, seq_len, d_ff)
        x = self.linear2(x)  # (batch_size, seq_len, d_model)
        return x


class EncoderLayer(torch.nn.Module):
    """
    Implements a single layer of the Transformer Encoder.

    An encoder layer consists of two main sub-layers:
    1.  A multi-head self-attention mechanism.
    2.  A position-wise fully connected feed-forward network.

    Each of these sub-layers has a residual connection around it, followed by
    layer normalization. The output of each sub-layer is `LayerNorm(x + Sublayer(x))`,
    where `Sublayer(x)` is the function implemented by the sub-layer itself.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048):
        """
        Initializes the EncoderLayer module.

        Args:
            d_model (int): The dimensionality of the input and output.
            num_heads (int): The number of heads in the multi-head attention.
            d_ff (int, optional): The dimensionality of the inner-layer of the
                feed-forward network. Defaults to 2048.
        """
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm_1 = LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
        self.layer_norm_2 = LayerNorm(d_model)

    def forward(self, in_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EncoderLayer.

        Args:
            in_embeddings (torch.Tensor): The input tensor of shape
                (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor of the same shape as the input.
        """
        # Multi-Head Attention sub-layer
        mha_out = self.mha(in_embeddings)
        # Residual connection and layer normalization
        norm_out_1 = self.layer_norm_1(mha_out + in_embeddings)

        # Feed-Forward Network sub-layer
        ffn_out = self.ffn(norm_out_1)
        # Residual connection and layer normalization
        output = self.layer_norm_2(ffn_out + norm_out_1)

        return output
