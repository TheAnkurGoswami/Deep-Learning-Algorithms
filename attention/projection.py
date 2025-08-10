import torch
from torch import Tensor


class Projection(torch.nn.Module):
    """
    Implements a linear projection layer.

    This module applies a linear transformation to the input data. It is a fundamental
    building block for many neural network architectures, including Transformers.

    The linear transformation is defined as:
    y = x @ W + b
    where:
    - x: is the input tensor.
    - W: is the weight matrix.
    - b: is the bias vector (optional).

    The weights are initialized from a normal distribution, and the bias (if used)
    is initialized to zeros.
    """

    def __init__(
        self, in_features: int, out_features: int, add_bias: bool = True
    ) -> None:
        """
        Initializes the Projection module.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            add_bias (bool, optional): Whether to include a bias term.
                Defaults to True.
        """
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.add_bias = add_bias

        self._weights = torch.normal(
            mean=0.0,
            std=1.0,
            size=(self._in_features, self._out_features),
            requires_grad=True,
        )
        if add_bias:
            self._bias = torch.zeros(
                size=(1, self._out_features), requires_grad=True
            )
        else:
            self._bias = None

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass for the linear projection.

        Args:
            inputs (Tensor): The input tensor of shape (batch_size, seq_len,
                in_features).

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, out_features).
        """
        # bsi -> (batch, seq_len, in_features)
        # io -> (in_features, out_features)
        output = torch.einsum("bsi,io->bso", inputs, self._weights)
        if self.add_bias:
            output += self._bias
        # output: (batch, seq_len, out_features)
        return output
