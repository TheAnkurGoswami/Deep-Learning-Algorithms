import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gain = torch.ones(d_model, 1, requires_grad=True)
        self.bias = torch.zeros(d_model, 1, requires_grad=True)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(input_, dim=-1, keepdim=True)
        var = torch.var(input_, dim=-1, keepdim=True)

        normalized = (input_ - mean) / torch.sqrt(var + self.eps)
        output = normalized * self.gain + self.bias
        return output
