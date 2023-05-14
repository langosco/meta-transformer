import dataclasses
from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp
from meta_transformer.transformer import Transformer, TransformerConfig
from meta_transformer import utils
from jax.typing import ArrayLike
from typing import Dict, Sequence
import functools
import chex
import numpy as np
from meta_transformer.preprocessing import NetEmbedding

@dataclasses.dataclass
class MetaModelClassifier(hk.Module):
    """A simple meta-model."""

    transformer: Transformer
    model_size: int
    num_classes: int
    name: Optional[str] = None
    use_embedding: Optional[bool] = False

    def __call__(
        self,
        inputs: ArrayLike,  # dict
        *,
        is_training: bool = True,
    ) -> jax.Array:
        """Forward pass. Returns a sequence of logits."""
        #net_embed = NetEmbedding(embed_dim=self.model_size)
        #inputs = hk.vmap(net_embed, split_rng=False)(inputs)
        if self.use_embedding:
            inputs = hk.Linear(self.model_size)(inputs)
        _, seq_len, _ = inputs.shape

        # Add positional embeddings.
        positional_embeddings = hk.get_parameter(
            'positional_embeddings', [seq_len, self.model_size], init=jnp.zeros)
        inputs = inputs + positional_embeddings  # [B, T, D]

        # Run the transformer over the inputs.
        outputs = self.transformer(inputs, is_training=is_training)

        first_out = outputs[:, 0, :]  # [B, D]
        return hk.Linear(self.num_classes, name="linear_output")(first_out)  # [B, V]


@chex.dataclass
class MetaModelClassifierConfig(TransformerConfig):
    """Hyperparameters for the model."""
    num_classes: int = 4
    use_embedding: bool = False


def create_meta_model_classifier(
        config: MetaModelClassifierConfig) -> hk.Transformed:
    @hk.transform
    def model(params_batch: dict, 
              is_training: bool = True) -> ArrayLike:
        net = MetaModelClassifier(
            model_size=config.model_size,
            num_classes=config.num_classes,
            use_embedding=config.use_embedding,
            transformer=Transformer(
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                key_size=config.key_size,
                dropout_rate=config.dropout_rate,
            ))
        return net(params_batch, is_training=is_training)
    return model


@dataclasses.dataclass
class MetaModel(hk.Module):
    """A meta-model that returns neural network parameters."""

    transformer: Transformer
    model_size: int
    name: Optional[str] = None
    use_embedding: Optional[bool] = False

    def __call__(
            self,
            inputs: ArrayLike,
            *,
            is_training: bool = True,
        ) -> jax.Array:
        """Forward pass. Returns a sequence of output embeddings."""
        if self.use_embedding:
            inputs = hk.Linear(self.model_size)(inputs)
        _, seq_len, _ = inputs.shape

        # Add positional embeddings.
        positional_embeddings = hk.get_parameter(
        'positional_embeddings', [seq_len, self.model_size], init=jnp.zeros)
        inputs = inputs + positional_embeddings  # [B, T, D]

        # Run the transformer over the inputs.
        outputs = self.transformer(
            inputs, is_training=is_training)  # [B, T, D]

        if self.use_embedding:
            outputs = hk.Linear(inputs.shape[-1])(outputs)
        return outputs


MetaModelConfig = TransformerConfig


def create_meta_model(
        config: MetaModelConfig) -> hk.Transformed:
    @hk.transform
    def model(input_batch: dict,
              is_training: bool = True) -> ArrayLike:
        net = MetaModel(
            model_size=config.model_size,
            transformer=Transformer(
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                key_size=config.key_size,
                dropout_rate=config.dropout_rate,
            ))
        return net(input_batch, is_training=is_training)
    return model
