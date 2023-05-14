import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from typing import Optional, Mapping, Any, Dict, Tuple
from jax.typing import ArrayLike


# Parameter chunking
def pad_to_chunk_size(arr: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    pad_size = -len(arr) % chunk_size
    padded = jnp.pad(arr, (0, pad_size))
    return padded


def chunk_layer(weights: jnp.ndarray, biases: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    flat_weights = weights.flatten()
    flat_weights = pad_to_chunk_size(flat_weights, chunk_size)
    weight_chunks = jnp.split(flat_weights, len(flat_weights) // chunk_size)
    biases = pad_to_chunk_size(biases, chunk_size)
    bias_chunks = jnp.split(biases, len(biases) // chunk_size)
    return weight_chunks, bias_chunks


def chunk_params(params: dict, chunk_size: int) -> dict:
    """
    Chunk the parameters of an MLP into chunks of size chunk_size.
    Chunks don't cross layer boundaries and are padded with zeros if necessary.
    """
    return {
        k: chunk_layer(layer["w"], layer["b"], chunk_size) for k, layer in params.items()
    }


def count_params(params: dict) -> int:
    """Count the number of parameters in a pytree of parameters."""
    return sum([x.size for x in jax.tree_util.tree_leaves(params)])


def tree_list(trees):
    """Maps a list of trees to a tree of lists."""
    return jax.tree_map(lambda *x: list(x), *trees)


def tree_stack(trees):
    """Stacks a list of trees into a single tree with an extra dimension."""
    return jax.tree_map(lambda *x: jnp.stack(x), *trees)