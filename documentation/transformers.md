# Transformers Module

The `transformers` module contains the core components of the Transformer model, including the encoder, decoder, and the full Transformer architecture.

## Feed-Forward Network

The `FeedForwardNetwork` class implements the position-wise feed-forward network used in the Transformer encoder and decoder layers. It consists of two linear transformations with a ReLU activation in between.

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

## Encoder

The `EncoderLayer` class represents a single layer of the Transformer encoder. It consists of two sub-layers:
1.  A multi-head self-attention mechanism.
2.  A position-wise fully connected feed-forward network.

Each sub-layer is followed by a residual connection and layer normalization.

The output of each sub-layer is $\text{LayerNorm}(x + \text{Sublayer}(x))$.

## Decoder

The `DecoderLayer` class represents a single layer of the Transformer decoder. It consists of three sub-layers:
1.  A masked multi-head self-attention mechanism.
2.  A multi-head attention mechanism over the output of the encoder stack.
3.  A position-wise fully connected feed-forward network.

Each sub-layer is followed by a residual connection and layer normalization.

## Transformer

The `Transformer` class implements the full Transformer model. It consists of a stack of encoder layers and a stack of decoder layers.

The input sequence is first passed through the positional encoding layer to inject positional information. Then, the encoded sequence is passed to the encoder stack. The decoder takes the target sequence (shifted right) and the encoder output to produce the final output.
