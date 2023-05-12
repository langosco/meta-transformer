import dataclasses
from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp
from meta_transformer.transformer import Transformer
from meta_transformer import utils
from jax.typing import ArrayLike
from typing import Dict, Sequence
import functools
import chex
import numpy as np


def chunk_array(x: ArrayLike, chunk_size: int) -> jax.Array:
    """Split an array into chunks of size chunk_size. 
    If not divisible by chunk_size, pad with zeros."""
    x = x.flatten()
    x = utils.pad_to_chunk_size(x, chunk_size)
    return x.reshape(-1, chunk_size)


# Chunking:
# 1. Input: dict of params, eg {'conv_1': {'w': [3, 3, 3, 64], 'b': [64]}, ...}
# 2. Flatten each layer into a single vector, eg {'conv_1': [3*3*3*64 + 64], ...}
# 3. Pad each vector with zeros to a multiple of chunk_size.
# 4. Split each vector into chunks of size chunk_size, eg {'conv_1_chunk_0': [chunk_size], ...}
# 5. Done!


@dataclasses.dataclass
class ChunkCNN:
    linear_chunk_size: int
    conv_chunk_size: int

    def __call__(self, params: Dict[str, Dict[str, ArrayLike]]) -> Dict[str, ArrayLike]:
        """Split a CNN into nice little weight chunks."""
        # First, reshape all the layers into a single vector per layer
        # (this is the part that's really a pain to compute the inverse of)
        params = {k: jnp.concatenate([v.flatten() for v in layer.values()])
                    for k, layer in params.items()}

        # Then, split the layers into chunks
        def chunk_size(layer_name):
            return self.conv_chunk_size if 'conv' in layer_name else \
                   self.linear_chunk_size
        params = {k: chunk_array(v, chunk_size(k)) for k, v in params.items()}
        params = {f"{k}_chunk_{i}": v for k, vs in params.items() 
                    for i, v in enumerate(vs)}
        return params


def unchunk_layers(chunked_params: Dict[str, ArrayLike]) -> Dict[str, jax.Array]:
    """Input: dictionary of chunked parameters (one-level). 
    Returns: flat dict of 'unchunked' parameters, ie combined into layers.
    """
    unchunked_params = {}
    for k, v in chunked_params.items():
        layer, _ = k.split("_chunk_")
        if layer not in unchunked_params:
            unchunked_params[layer] = [v]
        else:
            unchunked_params[layer].append(v)
    unchunked_params = {k: jnp.concatenate(v) for k, v in unchunked_params.items()}
    return unchunked_params


def get_layer_sizes(params: Dict[str, Dict[str, ArrayLike]]) -> Dict[str, ArrayLike]:
    """Get the size (weights.size + bias.size) of each layer in params.)"""
    return {k: sum([v.size for v in layer.values()]) 
            for k, layer in params.items()}


def unflatten_layer(
        layer: ArrayLike, 
        shapes: Dict[str, ArrayLike]) -> Dict[str, jax.Array]:
    """Unflatten a layer vector into a dict of weights and biases."""
    unflattened = {}
    i = 0
    for k, v in shapes.items():
        size = np.prod(v)  # np.prod instead?
        unflattened[k] = layer[i:i+size].reshape(v)
        i += size
    return unflattened


def nest_params(
        params: Dict[str, ArrayLike],
        nested_shapes: Dict[str, ArrayLike]) -> Dict[str, Dict[str, jax.Array]]:
    """Transform a flat dictionary of concatenated layer params into 
    a nested dictionary, with the correct shapes."""
    assert len(params) == len(nested_shapes), \
        f"Number of layers ({len(params)}) does " \
        f"not match number of shapes ({len(nested_shapes)})."
    nested_params = {}
    for layer, shapes in nested_shapes.items():
        nested_params[layer] = unflatten_layer(params[layer], shapes)
    return nested_params


def skip_layer(layer_name: str) -> bool:
    """Skip certain layers when chunking and unchunking."""
    skip_list = ['layer_norm', 'dropout', 'batch_norm']
    return any([s in layer_name for s in skip_list])


@dataclasses.dataclass
class UnChunkCNN:
    """Inverse of ChunkCNN."""
    original_shapes: Dict[str, Dict[str, Sequence[int]]]

    def __call__(self, params: Dict[str, ArrayLike]) -> Dict[str, Dict[str, jax.Array]]:
        """Unchunk a chunked CNN."""
        # First, unchunk the layers
        params = unchunk_layers(params)

        # Then, unflatten the layers
        params = nest_params(params, self.original_shapes)

        return params


@dataclasses.dataclass
class NetEmbedding(hk.Module):
    """A module that creates embedding vectors from neural network params.
    Not batched."""
    embed_dim: int
    linear_chunk_size: int = 1024
    conv_chunk_size: int = 256

    def __call__(
            self,
            input_params: Dict[str, Dict[str, ArrayLike]],
    ) -> jax.Array:
        chunk = ChunkCNN(self.linear_chunk_size, self.conv_chunk_size)
        conv_embed = hk.Linear(self.embed_dim)
        linear_embed = hk.Linear(self.embed_dim)
        chunked_params = chunk(input_params)  # dict

        embeddings = []
        for k, v in chunked_params.items():
            if 'conv' in k:
                embeddings.append(conv_embed(v))
            elif 'linear' in k:
                embeddings.append(linear_embed(v))
            elif skip_layer(k):
                continue
            else:
                raise ValueError(f"Invalid layer: {k}.")

        embeddings = jnp.stack(embeddings, axis=0)  # [T, D]
        chex.assert_shape(embeddings, [None, self.embed_dim])
        return embeddings


@dataclasses.dataclass
class NetUnEmbedding(hk.Module):
    """A module that maps embedding vectors back to neural network params.
    Not batched."""
    original_param_shapes: Dict[str, Dict[str, ArrayLike]]  
    linear_chunk_size: int = 1024
    conv_chunk_size: int = 256

    def get_layers(self):
        # for every layer, get the size of weights + biases (vectorized):
        flat_param_sizes = {k: sum([np.prod(v) for v in layer.values()])
                                  for k, layer in self.original_param_shapes.items()}

        # Now we need to figure out how many chunks each layer has
        n_chunks = {
            k: int(np.ceil(size / self.conv_chunk_size)) if 'conv' in k else \
                            int(np.ceil(size / self.linear_chunk_size))
                            for k, size in flat_param_sizes.items()}

        # list with layer names repeated n_chunks times:
        # (this is so we know which embedding vector corresponds to which layer)
        layers = [f"{k}_chunk_{i}" for k, nc in n_chunks.items()
                       for i in range(nc)]
        return layers

    def __call__(
            self,
            embeddings: ArrayLike,  # [T, D]
    ) -> dict:
        conv_unembed = hk.Linear(self.conv_chunk_size)
        linear_unembed = hk.Linear(self.linear_chunk_size)
        layers = self.get_layers()
        assert len(layers) == embeddings.shape[0], \
        f" Received embedding sequence length ({embeddings.shape[0]})." \
        f" does not match expected length ({len(layers)})."
        chex.assert_rank(embeddings, 2)  # no batch dim

        unembeddings = {}
        for layer_name, emb in zip(layers, embeddings):
            if 'conv' in layer_name:
                unembeddings[layer_name] = conv_unembed(emb)
            elif 'linear' in layer_name:
                unembeddings[layer_name] = linear_unembed(emb)
            elif skip_layer(layer_name):
                continue
            else:
                raise ValueError(f"Invalid layer: {layer_name}.")
        
        # Now we need to unchunk the layers
        unchunk = UnChunkCNN(self.original_param_shapes)
        unchunked_params = unchunk(unembeddings)  # dict
        return unchunked_params


@dataclasses.dataclass
class MetaModelClassifier(hk.Module):
  """A simple meta-model."""

  transformer: Transformer
  model_size: int
  num_classes: int
  name: Optional[str] = None
  chunk_size: Optional[int] = 4

  def __call__(
      self,
      input_params: dict,
      *,
      is_training: bool = True,
  ) -> jax.Array:
    """Forward pass. Returns a sequence of logits."""
    net_embed = NetEmbedding(embed_dim=self.model_size)
    embeddings = hk.vmap(net_embed, split_rng=False)(input_params)  # [B, T, D]
    _, seq_len, _ = embeddings.shape

    # Add positional embeddings.
    positional_embeddings = hk.get_parameter(
        'positional_embeddings', [seq_len, self.model_size], init=jnp.zeros)
    input_embeddings = embeddings + positional_embeddings  # [B, T, D]

    # Run the transformer over the inputs.
    embeddings = self.transformer(
        input_embeddings,
        is_training=is_training,
    )  # [B, T, D]

    first_out = embeddings[:, 0, :]  # [B, V]
    return hk.Linear(self.num_classes, name="linear_output")(first_out)  # [B, V]


@chex.dataclass
class MetaModelClassifierConfig:
    """Hyperparameters for the model."""
    num_heads: int = 4
    num_layers: int = 2
    dropout_rate: float = 0.1
    model_size: int = 128
    num_classes: int = 4

    def __post_init__(self):
        self.key_size = self.model_size // self.num_heads
        if self.model_size % self.num_heads != 0:
            raise ValueError(
                f"model_size ({self.model_size}) must be "
                "divisible by num_heads ({self.num_heads})")


def create_meta_model_classifier(
        config: MetaModelClassifierConfig) -> hk.Transformed:
    @hk.transform
    def model(params_batch: dict, 
              is_training: bool = True) -> ArrayLike:
        net = MetaModelClassifier(
            model_size=config.model_size,
            num_classes=config.num_classes,
            transformer=Transformer(
                num_heads=config.num_heads,
                num_layers=config.num_layers,
                key_size=config.key_size,
                dropout_rate=config.dropout_rate,
            ))
        return net(params_batch, is_training=is_training)
    return model