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

