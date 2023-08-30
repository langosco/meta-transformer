from jax.typing import ArrayLike
from typing import Tuple, Iterator, List
import numpy as np
from itertools import cycle
import nnaugment
from nnaugment.conventions import import_params, export_params
from meta_transformer import utils, preprocessing
import concurrent.futures


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



def augment_list_of_nets(nets: List[dict],
                         numpy_rng: np.random.Generator,
                         layers_to_permute: list):
    """Augment a list of nets with random augmentations"""
    augmented = [nnaugment.random_permutation(
        import_params(net), layers_to_permute=layers_to_permute, rng=numpy_rng) for net in nets]
    return [export_params(net) for net in augmented]


def augment_batch(
        batch: dict, 
        rng: np.random.Generator,
        layers_to_permute: list,
    ) -> dict:
    """Augment a batch of nets with permutation augmentations."""
    return {k: augment_list_of_nets(
            v, utils.clone_numpy_rng(rng), layers_to_permute
        ) for k, v in batch.items()}


class DataLoader:
    def __init__(self, inputs, targets, batch_size, data_std,
                 rng: np.random.Generator = None, 
                 max_workers: int = None,
                 augment: bool = False,
                 skip_last_batch: bool = True,
                 layers_to_permute: list = None,
                 chunk_size: int = 256,
                 ):
        self.batches = data_iterator(inputs, targets, 
                                    batchsize=batch_size, 
                                    skip_last=skip_last_batch)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.rng = rng
        self.augment = augment
        self.len = len(inputs) // batch_size
        self.layers_to_permute = layers_to_permute
        self.chunk_size = chunk_size
        self.data_std = data_std
        # self.data_std = np.std(flatten(inputs))

    def __iter__(self):
        for batch in self.batches:
            if self.augment:
                batch = augment_batch(
                    batch, self.rng, self.layers_to_permute)
            yield preprocessing.flatten_and_chunk_batch(
                batch, self.chunk_size, self.data_std)
#        else:
#            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:  # >2x faster than ThreadPool
#                rngs = self.rng.spawn(len(self)) if self.augment else [None] * len(self)
#                
#                print('yay concurrency')
#                futures = [
#                    executor.submit(process_batch,
#                                    batch,
#                                    numpy_rng=np_rng,
#                                    augment=self.augment,
#                                    layers_to_permute=self.layers_to_permute,
#                                    chunk_size=self.chunk_size)
#                    for np_rng, batch in zip(rngs, self.batches)
#                ]
#                print("ok done great! yielding futures as they complete...")
#                
#                for future in concurrent.futures.as_completed(futures):
#                    yield future.result()
    
    def __len__(self):
        return self.len  # number of batches


def validate_shapes(batch):
    """Check that the shapes are correct."""
    if not batch["input"].shape == batch["target"].shape:
        raise ValueError("Input and target shapes do not match. "
                        f"Received input shaped: {batch['input'].shape} "
                        f"and target shaped: {batch['target'].shape}.")