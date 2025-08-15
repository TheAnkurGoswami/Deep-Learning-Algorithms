import numpy as np
import pytest
import torch

from attention.flash_attention import FlashAttention
from attention.scaled_dot_product_attention import ScaledDotProductAttention
from utils import check_closeness

torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)
np.random.seed(69)
torch.manual_seed(69)


@pytest.mark.parametrize(
    "batch_size, seq_len, d_model",
    [
        (1, 4, 4),
        (2, 8, 4),
        (4, 16, 8),
    ],
)
def test_flash_attention(batch_size, seq_len, d_model):
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    flash_attention = FlashAttention()
    sdpa = ScaledDotProductAttention()

    flash_output = flash_attention(query, key, value)
    sdpa_output = sdpa(query, key, value)
    assert check_closeness(
        flash_output.detach().numpy(), sdpa_output.detach().numpy()
    ), "Flash Attention output does not match SDPA output"
