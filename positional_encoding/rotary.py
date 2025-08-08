import torch


class RotaryPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.wavelength = 10_000
        self.d_model = d_model

        ix = torch.arange(self.d_model) // 2
        # ix -> [0, 0, 1, 1, .... d/2 - 1, d/2 - 1]
        self.theta = self.wavelength ** (-2 * ix / self.d_model)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_ -> (batch_size, seq_len, d_model)
        seq_len = input_.shape[1]
        m = torch.arange(seq_len).view(-1, 1)
        theta_m = self.theta * m
        # theta_m -> (seq_len, d_model)

        first_comp = input_ * torch.cos(theta_m)
        second_comp = torch.stack(
            (-input_[:, :, 1::2], input_[:, :, ::2]), dim=-1
        ).reshape(input_.shape) * torch.sin(theta_m)

        return first_comp + second_comp
        # (batch_size, seq_len, d_model)
