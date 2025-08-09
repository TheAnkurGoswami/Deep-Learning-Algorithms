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

Instead of adding positional information to the embeddings, RoPE rotates the existing embeddings based on their position. This is achieved by multiplying the embedding by a rotation matrix that depends on its position. For a d-dimensional vector, the rotation matrix $\mathbf{R}_{m}^{d}$ is a block-diagonal matrix:

$$
\mathbf{R}_{m}^{d} =
\Bigg[
\begin{array}{ccccccc}
\cos(m\theta_{1}) & -\sin(m\theta_{1}) & 0 & 0 & \dots & 0 & 0 \\
\sin(m\theta_{1}) & \cos(m\theta_{1}) & 0 & 0 & \dots & 0 & 0 \\
0 & 0 & \cos(m\theta_{2}) & -\sin(m\theta_{2}) & \dots & 0 & 0 \\
0 & 0 & \sin(m\theta_{2}) & \cos(m\theta_{2}) & \dots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \dots & \cos(m\theta_{d/2}) & -\sin(m\theta_{d/2}) \\
0 & 0 & 0 & 0 & \dots & \sin(m\theta_{d/2}) & \cos(m\theta_{d/2})
\end{array}
\Bigg]
$$

The rotated embedding is then $f(\mathbf{x}, m) = \mathbf{R}_{m}^{d} \mathbf{x}$.

This rotation is applied to pairs of features. For a pair of features $(x_{2j-1}, x_{2j})$ at position $m$, this is equivalent to the following transformation:

$$
x'_{2j-1} = x_{2j-1} \cos(m\theta_j) - x_{2j} \sin(m\theta_j)
$$
$$
x'_{2j} = x_{2j-1} \sin(m\theta_j) + x_{2j} \cos(m\theta_j)
$$

where $\theta_j = 10000^{-2(j-1)/d_{\text{model}}}$. This rotation is applied for $j=1, \dots, d_{\text{model}}/2$.
