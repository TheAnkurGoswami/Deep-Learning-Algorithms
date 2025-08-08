import torch


class RotaryPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.wavelength = 10_000
        self.d_model = d_model

    def get_rotation_matrix(self, seq_len: int) -> torch.Tensor:
        ix = torch.arange(0, self.d_model) // 2
        # ix -> [0, 0, 1, 1, .... d/2 - 1, d/2 - 1]
        theta = self.wavelength ** (-2 * (ix) / self.d_model)
        m = torch.arange(0, seq_len).view(-1, 1)
        return theta * m
        # theta * m -> (seq_len, d_model)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        # input_ -> (batch_size, seq_len, d_model)
        first_comp = input_
        second_comp = torch.stack(
            (-input_[:, :, 1::2], input_[:, :, ::2]), dim=-1
        ).reshape(input_.shape)

        rotation_mat = self.get_rotation_matrix(seq_len=input_.shape[1])

        rotated_pe = first_comp * torch.cos(
            rotation_mat
        ) + second_comp * torch.sin(rotation_mat)

        return rotated_pe
        # rotated_pe -> (batch_size, seq_len, d_model)
