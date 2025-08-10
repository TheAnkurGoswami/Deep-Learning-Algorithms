import torch

from positional_encoding.sinusoidal import SinusoidalPositionalEncoding
from transformers.decoder import DecoderLayer
from transformers.encoder import EncoderLayer


class Transformer(torch.nn.Module):
    """
    Implements the Transformer model, as described in
    "Attention is All You Need".

    The Transformer model is a sequence-to-sequence architecture that relies
    entirely on self-attention mechanisms, without using recurrent or
    convolutional layers. It consists of an encoder and a decoder, each
    composed of a stack of identical layers.

    This implementation includes:
    -   Sinusoidal Positional Encoding to inject positional information.
    -   A stack of Encoder layers.
    -   A stack of Decoder layers.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
    ):
        """
        Initializes the Transformer model.

        Args:
            num_layers (int): The number of encoder and decoder layers.
            d_model (int): The dimensionality of the model.
            num_heads (int): The number of heads in the multi-head attention.
            d_ff (int): The dimensionality of the inner-layer of the
                feed-forward network.
        """
        super().__init__()
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model
        )
        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.dec_layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(
        self, input_embedding: torch.Tensor, output_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the Transformer model.

        Args:
            input_embedding (torch.Tensor): The input embeddings for the
                encoder, of shape (batch_size, seq_len_in, d_model).
            output_embedding (torch.Tensor): The output embeddings for the
                decoder, of shape (batch_size, seq_len_out, d_model).

        Returns:
            torch.Tensor: The output of the decoder stack, of shape
                (batch_size, seq_len_out, d_model).
        """
        # Add positional encoding to the input embeddings
        pe = self.positional_encoding(input_embedding)
        encoder_in = input_embedding + pe
        # Pass through the encoder stack
        for enc_layer in self.enc_layers:
            encoder_in = enc_layer(encoder_in)

        # Add positional encoding to the output embeddings
        pe = self.positional_encoding(output_embedding)
        decoder_in = output_embedding + pe
        # Pass through the decoder stack
        for dec_layer in self.dec_layers:
            decoder_in = dec_layer(decoder_in, encoder_in)

        return decoder_in
