import os
import pickle
import random
from functools import partial
import jax.random
from jax.typing import ArrayLike
from typing import Iterator
from meta_transformer import utils, preprocessing
import flax.struct


@flax.struct.dataclass  # TODO: replace with dataclass-array?
class Data:
    input: ArrayLike
    target: ArrayLike
    info: dict = None

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, i):
        return Data(self.input[i], self.target[i], self.info[i] if self.info else None)


def load_batch(filename: str) -> list[dict]:
    """Load a batch of checkpoints from a pickle file.
    Returns a list of dictionaries. Each dictionary has keys
    "params", "info", and "index".
    """
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


def load_batches_from_dirs(dirs, max_datapoints_per_dir=None):
    """Load all batches from a list of directories"""
    big_batch = []
    for datadir in dirs:
        big_batch.extend(load_batches(datadir, max_datapoints_per_dir))
    random.shuffle(big_batch)  # shuffle so in val split all types are present
    return big_batch


def data_iterator(
        data,
        batchsize=1024, 
        skip_last=False,
        stacked_tree=False,
        ) -> Iterator:
    """Iterate over the data in batches."""
    n = utils.tree_leaves_len(data) if stacked_tree else len(data)
    for i in range(0, n, batchsize):
        if i + batchsize > n:
            if skip_last:
                break
            else:
                batchsize = n - i

        if stacked_tree:
            yield jax.tree_map(lambda x: x[i:i + batchsize], data)
        else:
            yield data[i:i + batchsize]


class BaseDataLoader:
    def __init__(
            self,
            rng: jax.random.PRNGKey,
            data,
            batch_size: int,
            augment: bool = False,
            skip_last_batch: bool = True,
            layers_to_permute: list = None,
            chunk_size: int = 256,
            normalize_fn: callable = lambda x: x,
        ):
        self.rng = rng
        self.data = data
        self.batch_size = batch_size
        self.augment = augment
        self.skip_last_batch = skip_last_batch
        self.layers_to_permute = layers_to_permute
        self.chunk_size = chunk_size
        self.normalize_fn = normalize_fn

        self.batches = self.init_data_iterator()
        self.len = utils.tree_leaves_len(self.data) // \
            self.batch_size + (not self.skip_last_batch)

    def process_batch(self, batch):
        raise NotImplementedError
    
    def init_data_iterator(self):
        return data_iterator(
            self.data,
            batchsize=self.batch_size,
            skip_last=self.skip_last_batch,
            stacked_tree=True,
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

    def shuffle(self):
        subrng, self.rng = jax.random.split(self.rng)
        l = utils.tree_leaves_len(self.data)
        perm = jax.random.permutation(subrng, l)
        self.data = jax.tree_map(lambda x: x[perm], self.data)

    def _process(self, rng, params: dict):
        """transform a single dict of params"""
        if self.augment:
            out = preprocessing.augment_and_chunk(
                rng, params, self.chunk_size, self.layers_to_permute)
        else:
            out = preprocessing.chunk(params, self.chunk_size)[0]
        return self.normalize_fn(out)


class DataLoaderDetection(BaseDataLoader):
    """Dataloader for the backdoor detection task"""
    @partial(jax.jit, static_argnames="self")
    def process_batch(self, rng, batch):
        subrng, rng = jax.random.split(rng)
        rngs = jax.random.split(subrng, utils.tree_leaves_len(batch))
        params = jax.vmap(self._process)(rngs, batch["params"])
        return Data(
            input=params,
            target=batch["labels"],
        ), rng


class DataLoaderDepoison(BaseDataLoader):
    """Dataloader for the backdoor removal task (depoisoning)"""
    @partial(jax.jit, static_argnames="self")
    def process_batch(self, rng, batch):
        subrng, rng = jax.random.split(rng)
        rngs = jax.random.split(subrng, utils.tree_leaves_len(batch))
        return Data(
            input=jax.vmap(self._process)(rngs, batch["backd"]),
            target=jax.vmap(self._process)(rngs, batch["clean"]),
            info=batch["info"] if "info" in batch else None,
        ), rng


class DataLoaderTrainId(BaseDataLoader):
    """Dataloader for the backdoor removal task (depoisoning)"""
    @partial(jax.jit, static_argnames="self")
    def process_batch(self, rng, batch):
        subrng, rng = jax.random.split(rng)
        rngs = jax.random.split(subrng, utils.tree_leaves_len(batch))
        return Data(
            input=jax.vmap(self._process)(rngs, batch["backd"]),
            target=None,
        ), rng