import torch


class LayerNorm(torch.nn.Module):
    """
    Implements Layer Normalization.

    Layer Normalization is a technique to normalize the distributions of intermediate
    layers. It helps in stabilizing the training of deep neural networks. Unlike
    Batch Normalization, Layer Normalization is independent of the batch size.

    The formula for Layer Normalization is:
    y = ((x - E[x]) / sqrt(Var[x] + eps)) * gamma + beta
    where:
    - x: is the input tensor.
    - E[x]: is the mean of the input tensor.
    - Var[x]: is the variance of the input tensor.
    - eps: is a small value added for numerical stability.
    - gamma: is a learnable gain parameter.
    - beta: is a learnable bias parameter.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Initializes the LayerNorm module.

        Args:
            d_model (int): The dimensionality of the input.
            eps (float, optional): A small value added to the denominator for
                numerical stability. Defaults to 1e-5.
        """
        super().__init__()
        self.eps = eps
        self.gain = torch.ones((d_model,), requires_grad=True)
        self.bias = torch.zeros((d_model,), requires_grad=True)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Layer Normalization.

        Args:
            input_ (torch.Tensor): The input tensor of shape (batch_size,
                seq_len, d_model).

        Returns:
            torch.Tensor: The normalized tensor of the same shape as the input.
        """
        mean = torch.mean(input_, dim=-1, keepdim=True)
        var = torch.var(input_, dim=-1, keepdim=True, unbiased=False)

        normalized = (input_ - mean) / torch.sqrt(var + self.eps)
        output = normalized * self.gain + self.bias
        return output
