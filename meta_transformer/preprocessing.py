import jax
import jax.flatten_util
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit
from typing import Dict, Sequence, Tuple, Callable
from functools import partial


def pad_and_chunk(arr: jax.Array, chunk_size: int):
    pad_size = -len(arr) % chunk_size
    padded = jnp.pad(arr, (0, pad_size))
    chunks = padded.reshape(-1, chunk_size)
    return chunks


def flatten_and_chunk(
        params: Dict[str, Dict[str, ArrayLike]], 
        chunk_size: int
        ) -> Tuple[jax.Array, Callable]:
    """Preprocess a pytree of parameters into a flat array of chunks."""
    flat_params, unflatten = jax.flatten_util.ravel_pytree(params)
    chunks = pad_and_chunk(flat_params, chunk_size)
    num_params = len(flat_params)
    
    def unchunk_and_unflatten(chunks: ArrayLike) -> Dict[str, Dict[str, ArrayLike]]:
        """Inverse of flatten_and_chunk."""
        flat_params_new = chunks.flatten()[:num_params]
        return unflatten(flat_params_new)

    return chunks, unchunk_and_unflatten


def flatten_and_chunk_list(
        nets_list: Sequence[Dict[str, Dict[str, ArrayLike]]],
        chunk_size: int,
        data_std: float,
        ) -> Tuple[jax.Array, Callable]:
    assert len(nets_list) > 0, "Empty list of nets."
    flat, inverse_fns = zip(
        *[flatten_and_chunk(net, chunk_size) for net in nets_list])
    unchunk_and_unflatten_single = inverse_fns[0]
    return jnp.stack(flat) / data_std, unchunk_and_unflatten_single


@partial(jit, static_argnames="chunk_size")
def flatten_and_chunk_batch(
        batch: dict,
        chunk_size: int,
        data_std: float,
        ) -> Tuple[jax.Array, Callable]:
    return {k: flatten_and_chunk_list(
        v, chunk_size, data_std)[0] for k, v in batch.items()}