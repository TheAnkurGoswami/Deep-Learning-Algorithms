# Attention Module

The `attention` module provides the building blocks for the attention mechanism in the Transformer model.

## Scaled Dot-Product Attention

The `ScaledDotProductAttention` class implements the scaled dot-product attention mechanism.

The attention score is calculated as follows:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:
- $Q$ is the query matrix.
- $K$ is the key matrix.
- $V$ is the value matrix.
- $d_k$ is the dimension of the key vectors.

A causal mask can be applied to prevent positions from attending to subsequent positions.

## Multi-Head Attention

The `MultiHeadAttention` class implements the multi-head attention mechanism. It allows the model to jointly attend to information from different representation subspaces at different positions.

The input is projected into $h$ different subspaces, and scaled dot-product attention is applied in each subspace. The outputs are then concatenated and projected back to the original dimension.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.

The projection matrices $W_i^Q$, $W_i^K$, $W_i^V$ and $W^O$ are learned during training.

## Projection

The `Projection` class is a simple linear layer that projects the input from `in_features` to `out_features`. It can optionally include a bias term.

The operation is:

$$
\text{output} = \text{input} \cdot W + b
$$

Where:
- $W$ is the weight matrix.
- $b$ is the bias vector.

## Layer Normalization

The `LayerNorm` class implements layer normalization. It normalizes the inputs across the features.

The operation is:

$$
\text{output} = \frac{\text{input} - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

Where:
- $\mu$ is the mean of the input.
- $\sigma^2$ is the variance of the input.
- $\epsilon$ is a small value to prevent division by zero.
- $\gamma$ is a learnable gain parameter.
- $\beta$ is a learnable bias parameter.
