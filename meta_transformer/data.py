from jax.typing import ArrayLike
from typing import Tuple, Iterator
import numpy as np
from itertools import cycle


def data_iterator(
        inputs: ArrayLike, 
        targets: ArrayLike, 
        batchsize: int = 1024, 
        skip_last: bool = False,
        ) -> Iterator[Tuple[ArrayLike, ArrayLike]]:
    """Iterate over the data in batches."""
    n = len(inputs)
    assert n == len(targets), "Inputs and targets must be the same length."

    for i in range(0, n, batchsize):
        if skip_last and i + batchsize > n:
            break
        yield dict(input=inputs[i:i + batchsize], 
                   target=targets[i:i + batchsize])


def reached_last_batch(n, i, batchsize, skip_last):
    """Return true if index i has reached the last batch, such
    that seq[i:i+batchsize] is the last batch of size batchsize
    in a sequence of n elements."""
    if skip_last:
        return i + 2 * batchsize > n
    else:
        return i + batchsize >= n


def data_cycler(
        inputs: ArrayLike, 
        targets: ArrayLike, 
        batchsize: int = 1024, 
        skip_last: bool = False,
        ) -> Iterator[Tuple[ArrayLike, ArrayLike]]:
    """Cycle over the data in batches."""
    n = len(inputs)
    assert n == len(targets), "Inputs and targets must be the same length."

    batch_indices = range(0, n, batchsize)
    epoch_done = False
    for i in cycle(batch_indices):
        if skip_last and i + batchsize > n:
            continue
        epoch_done = reached_last_batch(n, i, batchsize, skip_last)
        yield dict(input=inputs[i:i + batchsize], 
                   target=targets[i:i + batchsize]), epoch_done


def split_data(data: ArrayLike, targets: ArrayLike, val_data_ratio: float = 0.1):
    split_index = int(len(data)*(1-val_data_ratio))
    return (data[:split_index], targets[:split_index], 
            data[split_index:], targets[split_index:])