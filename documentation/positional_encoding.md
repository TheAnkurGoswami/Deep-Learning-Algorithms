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

For a token at position $m$, its embedding $x_m$ is rotated by an angle $\theta_m$. The rotation is applied to pairs of features.

Given an input $x = (x_1, x_2, \dots, x_d)$, the rotated vector is:

$$
\begin{pmatrix}
x_1' \\
x_2' \\
\vdots \\
x_{d-1}' \\
x_d'
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & \dots & 0 & 0 \\
\sin(m\theta_1) & \cos(m\theta_1) & \dots & 0 & 0 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \dots & \cos(m\theta_{d/2}) & -\sin(m\theta_{d/2}) \\
0 & 0 & \dots & \sin(m\theta_{d/2}) & \cos(m\theta_{d/2})
\end{pmatrix}
\begin{pmatrix}
x_1 \\
x_2 \\
\vdots \\
x_{d-1} \\
x_d
\end{pmatrix}
$$

where $\theta_i = 10000^{-2(i-1)/d}$.
