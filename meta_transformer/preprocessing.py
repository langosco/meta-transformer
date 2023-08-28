import dataclasses
#import haiku as hk
import jax
import jax.flatten_util
import jax.numpy as jnp
from meta_transformer import utils
from jax.typing import ArrayLike
from typing import Dict, Sequence, Tuple, Callable, List
import chex
import numpy as np
import nnaugment
from nnaugment.conventions import import_params, export_params
from meta_transformer.data import data_iterator
import concurrent.futures


def pad_to_chunk_size(arr: ArrayLike, chunk_size: int) -> np.ndarray:
    pad_size = -len(arr) % chunk_size
    padded = np.pad(arr, (0, pad_size))
    return padded


def preprocess(
        params: Dict[str, Dict[str, ArrayLike]], 
        chunk_size: int
        ) -> Tuple[jax.Array, Callable]:
    """Preprocess a pytree of parameters into a flat array of chunks."""
    with jax.default_device(jax.devices("cpu")[0]):
        flat_params, unflatten = jax.flatten_util.ravel_pytree(params)
    padded = pad_to_chunk_size(flat_params, chunk_size)
    chunks = padded.reshape(-1, chunk_size)
    num_params = len(flat_params)
    
    def unpreprocess(chunks: ArrayLike) -> Dict[str, Dict[str, ArrayLike]]:
        """Inverse of preprocess."""
        flat_params_new = chunks.flatten()[:num_params]
        return unflatten(flat_params_new)

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
    chunks, unpreprocess = preprocess(params, chunk_size)
    if verbose:
        raveled_params = flatten(params)
        print()
        print(f"Number of (relevant) layers per net: {len(params)}")
        print(f"Number of parameters per net: "
            f"{raveled_params.shape[0]}")
        print(f"Chunk size: {chunk_size}")
        print(f"Number of chunks per net: {chunks.shape[0]}")
        print()
    return unpreprocess


# Check for high variance or mean of params
def flatten(x, on_cpu=True):
    if on_cpu:
        with jax.default_device(jax.devices("cpu")[0]):
            flat = jax.flatten_util.ravel_pytree(x)[0]
    flat = jax.flatten_util.ravel_pytree(x)[0]
    return flat


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


def augment_list_of_nets(nets: List[dict],
                         numpy_rng: np.random.Generator,
                         layers_to_permute: list):
    """Augment a list of nets with random augmentations"""
    augmented = [nnaugment.random_permutation(
        import_params(net), layers_to_permute=layers_to_permute, rng=numpy_rng) for net in nets]
    return [export_params(net) for net in augmented]


def process_nets(
        nets: List[dict], 
        augment: bool = True,
        data_std: float = 0.05,
        numpy_rng: np.random.Generator = None,
        layers_to_permute: list = None,
        chunk_size: int = 256,
    ) -> ArrayLike:
    """Permutation augment, then flatten to arrays."""
    assert len(nets) > 0, "Empty list of nets."
    if augment:
        assert layers_to_permute is not None
        nets = augment_list_of_nets(
            nets, numpy_rng=numpy_rng, layers_to_permute=layers_to_permute)
    nets = np.stack([preprocess(net, chunk_size)[0]
                        for net in nets])
    return nets / data_std


def process_batch(
        batch: dict, 
        numpy_rng: np.random.Generator = None,
        augment: bool = True, 
        data_std: float = 0.05,
        layers_to_permute: list = None,
        chunk_size: int = 256,
    ) -> dict:
    """process a batch of nets."""
    if numpy_rng is None:
        numpy_rng = np.random.default_rng()
    rng_state = numpy_rng.bit_generator.state
    rng_0 = np.random.default_rng()
    rng_1 = np.random.default_rng()
    rng_0.bit_generator.state = rng_state
    rng_1.bit_generator.state = rng_state
    inputs = process_nets(
        nets=batch["input"], augment=augment, data_std=data_std, 
        numpy_rng=rng_0, layers_to_permute=layers_to_permute, chunk_size=chunk_size)
    targets = process_nets(
        nets=batch["target"], augment=augment, data_std=data_std, 
        numpy_rng=rng_1, layers_to_permute=layers_to_permute, chunk_size=chunk_size)
    processed_batch = dict(input=inputs, target=targets)
    validate_shapes(processed_batch)
    return processed_batch


class DataLoader:
    def __init__(self, inputs, targets, batch_size, 
                 rng: np.random.Generator = None, 
                 num_workers: int = 32,
                 augment: bool = False,
                 skip_last_batch: bool = True,
                 layers_to_permute: list = None,
                 chunk_size: int = 256):
        self.batches = data_iterator(inputs, targets, batchsize=batch_size, skip_last=skip_last_batch)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rng = rng
        self.augment = augment
        self.len = len(inputs) // batch_size
        self.layers_to_permute = layers_to_permute
        self.chunk_size = chunk_size

    def __iter__(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            rngs = self.rng.spawn(len(self)) if self.augment else [None] * len(self)
            
            futures = [
                executor.submit(process_batch,
                                batch,
                                numpy_rng=np_rng,
                                augment=self.augment,
                                layers_to_permute=self.layers_to_permute,
                                chunk_size=self.chunk_size)
                for np_rng, batch in zip(rngs, self.batches)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                yield future.result()
    
    def __len__(self):
        return self.len  # number of batches


def validate_shapes(batch):
    """Check that the shapes are correct."""
    if not batch["input"].shape == batch["target"].shape:
        raise ValueError("Input and target shapes do not match. "
                        f"Received input shaped: {batch['input'].shape} "
                        f"and target shaped: {batch['target'].shape}.")
