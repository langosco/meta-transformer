import chex
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import random
from typing import Tuple, Iterator, Dict


# TODO: this creates a lot of copies of the data. Can we avoid this?
def shuffle_arrays(
        rng: ArrayLike, 
        *arrays: ArrayLike
        ) -> Tuple[ArrayLike, ArrayLike]:
    """Shuffle the data."""
    chex.assert_equal_shape_prefix(arrays, 1)
    idx = jnp.arange(len(arrays[0]))
    idx = random.permutation(rng, idx)
    return (arr[idx] for arr in arrays)


def data_iterator(
        inputs: ArrayLike, 
        targets: ArrayLike, 
        batchsize: int = 1024, 
        skip_last: bool = False,
        ) -> Iterator[Tuple[ArrayLike, ArrayLike]]:
    """Iterate over the data in batches."""
    for i in range(0, len(inputs), batchsize):
        if skip_last and i + batchsize > len(inputs):
            break
        yield dict(input=inputs[i:i + batchsize], 
                   target=targets[i:i + batchsize])


# TODO replace all this with huggingface datasets
def split_data(data: list, targets: list, val_data_ratio: float = 0.1):
    split_index = int(len(data)*(1-val_data_ratio))
    return (data[:split_index], targets[:split_index], 
            data[split_index:], targets[split_index:])