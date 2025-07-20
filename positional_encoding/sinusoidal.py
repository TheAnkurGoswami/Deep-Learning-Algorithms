import torch


class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.wavelength = 10_000

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_ -> (batch_size, seq_len, d_model)
        positions = torch.arange(input_.shape[1]).reshape(
            -1, 1
        )  # (seq_len, 1)
        index_vals = torch.arange(self.d_model)
        index_vals[1::2] = index_vals[1::2] - 1

        encoding = torch.zeros_like(input_)

        encoding[:, :, ::2] = torch.sin(
            positions / self.wavelength ** (index_vals[::2] / self.d_model)
        )
        encoding[:, :, 1::2] = torch.cos(
            positions / self.wavelength ** (index_vals[1::2] / self.d_model)
        )

        return encoding
