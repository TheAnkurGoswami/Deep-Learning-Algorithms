import torch
from torch import Tensor


class Projection(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, add_bias: bool = True
    ) -> None:
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.add_bias = add_bias

        self._weights = torch.normal(
            mean=0.0, std=1.0, size=(self._in_features, self._out_features)
        ).requires_grad_(True)
        if add_bias:
            self._bias = torch.zeros(
                size=(1, self._out_features)
            ).requires_grad_(True)
        else:
            self._bias = None

    def forward(self, inputs: Tensor):
        # bsi -> (batch, seq_len, in_features)
        # io -> (in_features, out_features)
        output = torch.einsum("bsi,io->bso", inputs, self._weights)
        if self.add_bias:
            output += self._bias
        # output: (batch, seq_len, out_features)
        return output
