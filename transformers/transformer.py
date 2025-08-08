import torch

from positional_encoding.sinusoidal import SinusoidalPositionalEncoding
from transformers.decoder import DecoderLayer
from transformers.encoder import EncoderLayer


class Transformer(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
    ):
        super().__init__()
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model
        )
        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
        )
        self.dec_layers = torch.nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff) for _ in range(num_layers)]
        )

    def forward(
        self, input_embedding: torch.Tensor, output_embedding: torch.Tensor
    ) -> torch.Tensor:
        pe = self.positional_encoding(input_embedding)
        encoder_in = input_embedding + pe
        for enc_layer in self.enc_layers:
            encoder_in = enc_layer(encoder_in)

        pe = self.positional_encoding(output_embedding)
        decoder_in = output_embedding + pe
        for dec_layer in self.dec_layers:
            decoder_in = dec_layer(decoder_in, encoder_in)

        return decoder_in
