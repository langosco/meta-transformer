import dataclasses
from typing import Optional
import haiku as hk
import jax
import jax.numpy as jnp
from meta_transformer.transformer import Transformer
from meta_transformer import utils
from jax.typing import ArrayLike
from typing import Dict, Sequence, Tuple, Callable, List
import functools
import chex
import numpy as np


def skip_layer(layer_name: str) -> bool:
    """Skip certain layers when chunking and unchunking."""
    skip_list = ['dropout', 'norm']
    return any([s in layer_name.lower() for s in skip_list])


def pad_to_chunk_size(arr: ArrayLike, chunk_size: int) -> np.ndarray:
    pad_size = -len(arr) % chunk_size
    padded = np.pad(arr, (0, pad_size))
    return padded


def filter_layers(
        params: Dict[str, Dict[str, ArrayLike]]
        ) -> Tuple[Dict[str, Dict[str, ArrayLike]], Callable]:
    """
    Filters out layers from params and provides a callable to retrieve them.
    """
    if not params:
        raise ValueError("Empty parameter dict.")

    output_layers = {}
    removed_layers = {}
    approved_layers = ['conv', 'linear', 'head', 'mlp']
    for k, v in params.items():
        if skip_layer(k):
            removed_layers[k] = v
        elif any([l in k.lower() for l in approved_layers]):
            output_layers[k] = v
        else:
            raise ValueError(f"Invalid layer: {k}.")

    original_order = list(params.keys())
    
    def unfilter(filtered_params) -> Dict[str, Dict[str, ArrayLike]]:
        """Inverse of filter_layers."""
        return {k: (filtered_params[k] if k in filtered_params
                     else removed_layers[k]) for k in original_order}
        
    return output_layers, unfilter


def preprocess(
        params: Dict[str, Dict[str, ArrayLike]], 
        chunk_size: int
        ) -> Tuple[jax.Array, Callable]:
    """Preprocess a pytree of parameters into a flat array of chunks."""
    params, unfilter = filter_layers(params)
    flat_params, unflatten = jax.flatten_util.ravel_pytree(params)
    padded = pad_to_chunk_size(flat_params, chunk_size)
    chunks = padded.reshape(-1, chunk_size)
    
    def unpreprocess(chunks: ArrayLike) -> Dict[str, Dict[str, ArrayLike]]:
        """Inverse of preprocess."""
        flat_params_new = chunks.flatten()[:len(flat_params)]
        return unfilter(unflatten(flat_params_new))

    return chunks, unpreprocess
    

def get_param_shapes(
        params: Dict[str, Dict[str, ArrayLike]]) -> Dict[str, Dict[str, Tuple[int, ...]]]:
    return {
        k: {subk: v[0].shape for subk, v in layer.items()}
        for k, layer in params.items()
    }


def get_unpreprocess_fn(
        params: Dict[str, Dict[str, ArrayLike]],
        chunk_size: int,
        verbose: bool = True,
        ) -> Tuple[Dict[str, Dict[str, Tuple[int, ...]]], int]:
    """Extra function that preprocess once just 
    to get the unpreprocess function."""
    params, _ = filter_layers(params)
    chunks, unpreprocess = preprocess(params, chunk_size)
    if verbose:
        raveled_params = jax.flatten_util.ravel_pytree(params)[0]
        print()
        print(f"Number of (relevant) layers per net: {len(params)}")
        print(f"Number of parameters per net: "
            f"{raveled_params.shape[0]}")
        print(f"Chunk size: {chunk_size}")
        print(f"Number of chunks per net: {chunks.shape[0]}")
        print()
    return unpreprocess


# Check for high variance or mean of params
def flatten(x):
    return jax.flatten_util.ravel_pytree(x)[0]


def is_fine(params: dict):
    """Return false if std or mean is too high."""
    flat = flatten(params)
    if flat.std() > 5.0 or jnp.abs(flat.mean()) > 5.0:
        return False
    else:
        return True


# TODO untested!
def filter_data(*arrays):
    """Given a list of net arrays, filter out those
    with very large means or stds."""
    chex.assert_equal_shape_prefix(arrays, 1)  # equal len

    def all_fine(elements):
        return all([is_fine(x) for x in elements])

    arrays_filtered = zip(*[x for x in zip(*arrays) if all_fine(x)])
    num_filtered = len(arrays[0]) - len(arrays_filtered[0])
    print(f"Filtered out {num_filtered} nets.")
    return arrays_filtered



#####################
## old stuff below ##


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
    conv_chunk_size: int = 1024

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
            else:
                raise ValueError(f"Invalid layer: {k}.")

        embeddings = jnp.stack(embeddings, axis=0)  # [T, D]
        chex.assert_shape(embeddings, [None, self.embed_dim])
        return embeddings


@dataclasses.dataclass
class NetUnEmbedding(hk.Module):
    """A module that maps embedding vectors back to neural network params.
    Not batched."""
    original_param_shapes: Dict[str, Dict[str, ArrayLike]]  # TODO get this from dummy_params
    linear_chunk_size: int = 1024
    conv_chunk_size: int = 1024

    def get_layers(self, dummy_params):
        chunk = ChunkCNN(self.linear_chunk_size, self.conv_chunk_size)
        dummy_chunked = chunk(dummy_params)
        layers = list(dummy_chunked.keys())
        return layers

#        # for every layer, get the size of weights + biases (vectorized):
#        flat_param_sizes = {k: sum([np.prod(v) for v in layer.values()])
#                                  for k, layer in self.original_param_shapes.items()}
#
#        # Now we need to figure out how many chunks each layer has
#        n_chunks = {
#            k: int(np.ceil(size / self.conv_chunk_size)) if 'conv' in k else \
#                            int(np.ceil(size / self.linear_chunk_size))
#                            for k, size in flat_param_sizes.items()}
#
#        # list with layer names repeated n_chunks times:
#        # (this is so we know which embedding vector corresponds to which layer)
#        layers = [f"{k}_chunk_{i}" for k, nc in n_chunks.items()
#                       for i in range(nc)]
#        return layers


    def __call__(
            self,
            embeddings: ArrayLike,  # [T, D]
            dummy_params: Dict[str, Dict[str, ArrayLike]],
    ) -> dict:
        conv_unembed = hk.Linear(self.conv_chunk_size)
        linear_unembed = hk.Linear(self.linear_chunk_size)
        layers = self.get_layers(dummy_params=dummy_params)
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
            else:
                raise ValueError(f"Invalid layer: {layer_name}.")
        
        # Now we need to unchunk the layers
        unchunk = UnChunkCNN(self.original_param_shapes)
        unchunked_params = unchunk(unembeddings)  # dict
        return unchunked_params
