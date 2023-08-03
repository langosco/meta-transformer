import dataclasses
from typing import Optional
import jax
import flax.linen as nn

from meta_transformer.transformer import Transformer
from jax.typing import ArrayLike


@dataclasses.dataclass
class MetaModelClassifier(nn.Module):
    """A simple meta-model."""
    d_model: int
    num_heads: int
    num_layers: int
    dropout_rate: int
    widening_factor: int = 4
    num_classes: int
    name: Optional[str] = None
    use_embedding: Optional[bool] = False

    @nn.compact
    def __call__(
        self,
        inputs: ArrayLike,  # dict
        *,
        is_training: bool = True,
    ) -> jax.Array:
        """Forward pass. Returns a sequence of logits."""
        #net_embed = NetEmbedding(embed_dim=self.d_model)
        #inputs = vmap(net_embed)(inputs)
        if self.use_embedding:
            inputs = nn.Dense(self.transformer.d_model)(inputs)
        _, seq_len, _ = inputs.shape

        positional_embeddings = self.param(
            'positional_embeddings',
            nn.initializers.zeros(),
#            nn.initializers.normal(stddev=0.02),  # From BERT
            (seq_len, self.d_model)
        )
        inputs = inputs + positional_embeddings  # [B, T, D]

        # Run the transformer over the inputs.
        # Run the transformer over the inputs.
        transformer = Transformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            widening_factor=self.widening_factor,
        )
        outputs = transformer(inputs, is_training=is_training)

        first_out = outputs[:, 0, :]  # [B, D]
        return nn.Dense(self.num_classes, name="linear_output")(first_out)  # [B, V]


@dataclasses.dataclass
class MetaModel(nn.Module):
    """A meta-model that returns neural network parameters."""

    d_model: int
    num_heads: int
    num_layers: int
    dropout_rate: int
    widening_factor: int = 4
    name: Optional[str] = None
    use_embedding: Optional[bool] = False

    @nn.compact
    def __call__(
            self,
            inputs: ArrayLike,
            *,
            is_training: bool = True,
        ) -> jax.Array:
        """Forward pass. Returns a sequence of output embeddings."""
        input_shape = inputs.shape[-1]
        if self.use_embedding:
            inputs = nn.Dense(self.d_model)(inputs)
        _, seq_len, _ = inputs.shape

        # Add positional embeddings.
        positional_embeddings = self.param(
            'positional_embeddings',
            nn.initializers.zeros,
#            nn.initializers.normal(stddev=0.02),  # From BERT
            (seq_len, self.d_model)
        )
        inputs = inputs + positional_embeddings

        # Run the transformer over the inputs.
        transformer = Transformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            widening_factor=self.widening_factor,
        )
        outputs = transformer(inputs, is_training=is_training)

        if self.use_embedding:
            outputs = nn.Dense(input_shape)(outputs)
        return outputs