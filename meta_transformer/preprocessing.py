import jax
import jax.flatten_util
import jax.numpy as jnp
from jax.typing import ArrayLike
import nnaugment


def pad_and_chunk(arr: ArrayLike, chunk_size: int) -> jax.Array:
    pad_size = -len(arr) % chunk_size
    padded = jnp.pad(arr, (0, pad_size))
    chunks = padded.reshape(-1, chunk_size)
    return chunks


def chunk(params: dict[str, dict[str, ArrayLike]], 
          chunk_size: int) -> (jax.Array, callable):
    """Preprocess a pytree of parameters into a flat array of chunks."""
    flat_params, unflatten = jax.flatten_util.ravel_pytree(params)
    chunks = pad_and_chunk(flat_params, chunk_size)
    num_params = len(flat_params)
    
    def unchunk(chunks: ArrayLike) -> dict[str, dict[str, jax.Array]]:
        """Inverse of flatten_and_chunk."""
        flat_params_new = chunks.flatten()[:num_params]
        return unflatten(flat_params_new)

    return chunks, unchunk


def augment_and_chunk(rng, params, chunk_size, layers_to_permute) -> jax.Array:
    augmented = nnaugment.random_permutation(
        rng, params, layers_to_permute, sort=True)
    return chunk(augmented, chunk_size)[0]