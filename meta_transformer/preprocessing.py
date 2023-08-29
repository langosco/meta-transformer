import jax
import jax.flatten_util
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit
from typing import Dict, Sequence, Tuple, Callable, List
import chex
import numpy as np
import nnaugment
from nnaugment.conventions import import_params, export_params
from meta_transformer.data import data_iterator
from meta_transformer import utils
import concurrent.futures
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
            yield flatten_and_chunk_batch(
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
