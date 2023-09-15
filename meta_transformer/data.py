import os
import pickle
from functools import partial
import jax.random
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Tuple, Iterator, List
import numpy as np
from itertools import cycle
import nnaugment
from meta_transformer import utils, preprocessing
import flax.struct
from typing import Sequence
import concurrent.futures
import orbax.checkpoint
from etils.epath import Path
import json
checkpointer = orbax.checkpoint.PyTreeCheckpointer()


@flax.struct.dataclass
class ParamsArrSingle:
    params: ArrayLike
    label: {0, 1}

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, i):
        return ParamsArrSingle(self.params[i], self.label[i])


@flax.struct.dataclass
class ParamsTreeSingle:
    params: Sequence[dict] | dict
    label: Sequence[int] | int

    def __len__(self):
        assert len(self.params) == len(self.label), \
                     "Inputs and targets must be the same length."
        return len(self.label)
    
    def __getitem__(self, i):
        return ParamsTreeSingle(self.params[i], self.label[i])


@flax.struct.dataclass  # TODO: replace with dataclass-array?
class ParamsArrPair:  # ParamsArrPairs
    """Individual datapoints or batches, in flat representation."""
    backdoored: ArrayLike
    clean: ArrayLike
    target_label: ArrayLike

    def __len__(self):
        assert len(self.clean) == len(self.backdoored), \
                   "Inputs and targets must be the same length."
        return len(self.clean)
    
    def __getitem__(self, i):
        return ParamsArrPair(self.backdoored[i], self.clean[i], self.target_label[i])


@flax.struct.dataclass
class ParamsTreePair:
    """Individual datapoints or batches, in pytree representation."""
    backdoored: Sequence[dict] | dict
    clean: Sequence[dict] | dict
    info: Sequence[dict] | dict

    def __len__(self):
        return len(self.clean)
    
    def __getitem__(self, i):
        return ParamsTreePair(self.backdoored[i], self.clean[i], self.info[i])


def load_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f)
    return batch


def load_batches(datadir, max_datapoints=None):
    """Load all batches from a directory"""
    big_batch = []
    for entry in os.scandir(datadir):
        if entry.name.startswith('checkpoints'):
            big_batch.extend(load_batch(entry.path))
        if max_datapoints is not None and len(big_batch) >= max_datapoints:
            big_batch = big_batch[:max_datapoints]
            break
    return big_batch


def load_model(idx: int, dir: str) -> (dict, dict[str]):
    """Load a model from a directory."""
    path = Path(dir) / str(idx) / 'params'
    info = json.loads((Path(dir) / str(idx) / 'info.json').read_text())
    return checkpointer.restore(path), info


def load_models(idxs: Sequence[int], dir: str, max_workers: int = 1):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        out = executor.map(partial(load_model, dir=dir), idxs)
    models, info = [list(x) for x in zip(*out)]
    return models, info


def load_pair_of_models(idx: int, backdoored_dir: str, clean_dir: str
                        ) -> (dict, dict, dict[str]):
    """Load a pair of models from two different directories."""
    infopath = Path(backdoored_dir) / str(idx) / 'info.json'
    bpath = Path(backdoored_dir) / str(idx) / 'params'
    cpath = Path(clean_dir) / str(idx) / 'params'
    poison_info = json.loads(infopath.read_text())
    return (checkpointer.restore(bpath), 
            checkpointer.restore(cpath), 
            poison_info)


def load_clean_and_backdoored(
        num_pairs: int,
        backdoored_dir: str,
        clean_dir: str,
        max_workers: int = 1,
        ) -> ParamsTreePair:
    """Load a batch of pairs of models from two different directories."""
    def load_pair(idx):
        return load_pair_of_models(idx, backdoored_dir, clean_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        loaded_models = executor.map(load_pair, range(num_pairs))
    
    backdoored, clean, info = [list(x) for x in zip(*loaded_models)]
    return ParamsTreePair(backdoored, clean, info=info)


def data_iterator(
        data: ParamsTreePair,
        batchsize: int = 1024, 
        skip_last: bool = False,
        ) -> Iterator[ParamsTreePair]:
    """Iterate over the data in batches."""
    n = len(data)
    for i in range(0, n, batchsize):
        if skip_last and i + batchsize > n:
            break
        yield data[i:i + batchsize]


def reached_last_batch(n, i, batchsize, skip_last):
    """Return true if index i has reached the last batch, such that 
    seq[i:i+batchsize] is the last batch of size batchsize in a 
    sequence of n elements."""
    if skip_last:
        return i + 2 * batchsize > n
    else:
        return i + batchsize >= n


def data_cycler(
        data: ParamsTreePair,
        batchsize: int = 1024, 
        skip_last: bool = False,
        ) -> Iterator[Tuple[ArrayLike, ArrayLike]]:
    """Cycle over the data in batches."""
    n = len(data)

    batch_indices = range(0, n, batchsize)
    epoch_done = False
    for i in cycle(batch_indices):
        if skip_last and i + batchsize > n:
            continue
        epoch_done = reached_last_batch(n, i, batchsize, skip_last)
        yield data[i:i + batchsize], epoch_done


def split_data(data: ParamsTreePair, val_data_ratio: float = 0.1):
    split_index = int(len(data)*(1-val_data_ratio))
    return data[:split_index], data[split_index:]


def flatten_and_chunk_batch(
        batch: ParamsTreePair,
        chunk_size: int,
        data_std: float,
        ):
    assert len(batch.backdoored) == len(batch.clean), \
                "Inputs and targets must be the same length."
    proc = preprocessing.flatten_and_chunk_list
    b, inverse = proc(batch.backdoored, chunk_size, data_std)
    c, inverse = proc(batch.clean, chunk_size, data_std)
    return ParamsArrPair(
        backdoored=b,
        clean=c,
        target_label=np.array([x['target_label'] for x in batch.info]),
    ), inverse


def augment_list_of_params(rng: jax.random.PRNGKey,
                           params: List[dict],
                           layers_to_permute: list):
    """Augment a list of nets with random augmentations"""
    rngs = jax.random.split(rng, len(params))
    return [nnaugment.random_permutation(
        r, p, layers_to_permute=layers_to_permute, 
        convention="flax", sort=True) for r, p in zip(rngs, params)]


def augment_batch(
        rng: jax.random.PRNGKey,
        batch: ParamsTreePair, 
        layers_to_permute: list,
    ) -> dict:
    """Augment a batch of nets with permutation augmentations."""
    return ParamsTreePair(
        augment_list_of_params(rng, batch.backdoored, layers_to_permute),
        augment_list_of_params(rng, batch.clean, layers_to_permute),
        info=batch.info,
    )


class BaseDataLoader:
    def __init__(
            self,
            rng: jax.random.PRNGKey,
            data,
            batch_size: int,
            data_std: float,
            max_workers: int = None,
            augment: bool = False,
            skip_last_batch: bool = True,
            layers_to_permute: list = None,
            chunk_size: int = 256,
        ):
        self.rng = rng
        self.data = data
        self.batch_size = batch_size
        self.data_std = data_std
        self.max_workers = max_workers
        self.augment = augment
        self.skip_last_batch = skip_last_batch
        self.layers_to_permute = layers_to_permute
        self.chunk_size = chunk_size

        self.batches = self.init_data_iterator()
        self.len = len(self.data) // self.batch_size + (not self.skip_last_batch)

    def process_batch(self, batch):
        raise NotImplementedError
    
    def init_data_iterator(self):
        return data_iterator(
            self.data,
            batchsize=self.batch_size,
            skip_last=self.skip_last_batch,
        )

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch, self.rng = self.process_batch(self.rng, next(self.batches))
            return batch
        except StopIteration:
            self.batches = self.init_data_iterator()
            raise StopIteration

    def __len__(self):
        return len(self.data) // self.batch_size + (not self.skip_last_batch)


class DataLoaderSingle(BaseDataLoader):
    @partial(jax.jit, static_argnames="self")
    def process_batch(self, rng, batch):
        subrng, rng = jax.random.split(rng)
        if self.augment:
            params = augment_list_of_params(subrng,
                batch.params, self.layers_to_permute)
        else:
            params = batch.params

        flat_params, inverse = preprocessing.flatten_and_chunk_list(
            params, self.chunk_size, self.data_std)
        
        return ParamsArrSingle(
            params=flat_params,
            label=jnp.array(batch.label)
        ), rng

    def shuffle(self):
        subrng, self.rng = jax.random.split(self.rng)
        perm = jax.random.permutation(subrng, len(self.data))
        self.data = ParamsTreeSingle(
            params=[self.data.params[i] for i in perm],
            label=[self.data.label[i] for i in perm],
        )


class DataLoaderPair(BaseDataLoader):
    def process_batch(self, batch):
        if self.augment:
            batch = augment_batch(
                batch, self.rng, self.layers_to_permute)
        return flatten_and_chunk_batch(
            batch, self.chunk_size, self.data_std)[0]

    def shuffle(self):
        perm = self.rng.permutation(len(self.data))
        return ParamsTreePair(
            backdoored=[self.data.backdoored[i] for i in perm],
            clean=[self.data.clean[i] for i in perm],
            info=[self.data.info[i] for i in perm],
        )
    

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
    if not batch.backdoors.shape == batch.clean.shape:
        raise ValueError("Input and target shapes do not match. "
                            f"Got {batch.backdoors.shape} and {batch.clean.shape}.")
