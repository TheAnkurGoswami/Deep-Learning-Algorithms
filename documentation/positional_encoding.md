# Positional Encoding Module

The `positional_encoding` module provides methods for incorporating positional information into the input embeddings.

## Sinusoidal Positional Encoding

The `SinusoidalPositionalEncoding` class implements the sinusoidal positional encoding method. This method adds a unique positional encoding to each input embedding based on its position in the sequence.

The positional encodings are calculated as follows:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

Where:
- $pos$ is the position of the token in the sequence.
- $i$ is the dimension index.
- $d_{\text{model}}$ is the dimension of the model.

## Rotary Positional Encoding (RoPE)

The `RotaryPositionalEncoding` class implements Rotary Positional Encoding (RoPE). RoPE encodes absolute positional information with a rotation matrix and naturally incorporates explicit relative position dependency in self-attention.

Instead of adding positional information to the embeddings, RoPE rotates the existing embeddings based on their position. The rotation is applied to pairs of features. For a pair of features $(x_{2j-1}, x_{2j})$ at position $m$, the transformation is given by:

$$
\left( \begin{matrix} x'_{2j-1} \\ x'_{2j} \end{matrix} \right) = \left( \begin{matrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{matrix} \right) \left( \begin{matrix} x_{2j-1} \\ x_{2j} \end{matrix} \right)
$$

where $\theta_j = 10000^{-2(j-1)/d_{\text{model}}}$. This rotation is applied for $j=1, \dots, d_{\text{model}}/2$.
