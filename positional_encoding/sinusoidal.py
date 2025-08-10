import torch


class SinusoidalPositionalEncoding(torch.nn.Module):
    """
    Implements Sinusoidal Positional Encoding.

    This module generates positional encodings that are added to the input embeddings
    to provide the model with information about the position of tokens in a sequence.
    The positional encodings are based on sine and cosine functions of different
    frequencies.

    The formulas for the positional encodings are:
    PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    where:
    - pos: is the position of the token in the sequence.
    - i: is the dimension index.
    - d_model: is the dimensionality of the model.

    The use of sine and cosine functions allows the model to learn relative
    positional information, as the encoding for any position can be represented as a
    linear function of the encoding of other positions.
    """

    def __init__(self, d_model: int):
        """
        Initializes the SinusoidalPositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the model.
        """
        super().__init__()
        self.d_model = d_model
        self.wavelength = 10_000

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Generates sinusoidal positional encodings.

        Note: The input tensor is only used to determine the sequence length and
        device. Its values are not used in the computation.

        Args:
            input_ (torch.Tensor): The input tensor of shape
                (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The positional encodings of the same shape as the input.
        """
        # input_ -> (batch_size, seq_len, d_model)
        seq_len = input_.shape[1]
        positions = torch.arange(seq_len, device=input_.device).reshape(
            -1, 1
        )  # (seq_len, 1)

        # Calculate the dimension indices for the denominator
        index_vals = torch.arange(self.d_model, device=input_.device)
        index_vals[1::2] = index_vals[1::2] - 1

        encoding = torch.zeros_like(input_)

        # Calculate the positional encodings for even and odd dimensions
        denominator = self.wavelength ** (index_vals[::2] / self.d_model)
        encoding[:, :, ::2] = torch.sin(positions / denominator)
        encoding[:, :, 1::2] = torch.cos(positions / denominator)

        return encoding
