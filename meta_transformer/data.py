from functools import partial
from jax import jit
from jax.typing import ArrayLike
from typing import Tuple, Iterator, List
import numpy as np
from itertools import cycle
import nnaugment
from nnaugment.conventions import import_params, export_params
from meta_transformer import utils, preprocessing
import flax.struct
from typing import Sequence
import concurrent.futures
import orbax.checkpoint
from etils.epath import Path
import json
checkpointer = orbax.checkpoint.PyTreeCheckpointer()


@flax.struct.dataclass  # TODO: replace with dataclass-array?
class ParamsData:
    """Individual datapoints or batches, in flat representation."""
    backdoored: ArrayLike
    clean: ArrayLike
    target_label: ArrayLike

    def __len__(self):
        assert len(self.clean) == len(self.backdoored), \
                   "Inputs and targets must be the same length."
        return len(self.clean)
    
    def __getitem__(self, i):
        return ParamsData(self.backdoored[i], self.clean[i], self.target_label[i])


@flax.struct.dataclass
class ParamsDataTree:
    """Individual datapoints or batches, in pytree representation."""
    backdoored: Sequence[dict] | dict
    clean: Sequence[dict] | dict
    info: Sequence[dict] | dict

    def __len__(self):
        return len(self.clean)
    
    def __getitem__(self, i):
        return ParamsDataTree(self.backdoored[i], self.clean[i], self.info[i])


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
        ) -> ParamsDataTree:
    """Load a batch of pairs of models from two different directories."""
    def load_pair(idx):
        return load_pair_of_models(idx, backdoored_dir, clean_dir)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        loaded_models = executor.map(load_pair, range(num_pairs))
    
    backdoored, clean, info = [list(x) for x in zip(*loaded_models)]
    return ParamsDataTree(backdoored, clean, info=info)


def data_iterator(
        data: ParamsDataTree,
        batchsize: int = 1024, 
        skip_last: bool = False,
        ) -> Iterator[ParamsDataTree]:
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
        data: ParamsDataTree,
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


def split_data(data: ParamsDataTree, val_data_ratio: float = 0.1):
    split_index = int(len(data)*(1-val_data_ratio))
    return data[:split_index], data[split_index:]


def flatten_and_chunk_batch(
        batch: ParamsDataTree,
        chunk_size: int,
        data_std: float,
        ):
    assert len(batch.backdoored) == len(batch.clean), \
                "Inputs and targets must be the same length."
    proc = preprocessing.flatten_and_chunk_list
    b, inverse = proc(batch.backdoored, chunk_size, data_std)
    c, inverse = proc(batch.clean, chunk_size, data_std)
    return ParamsData(
        backdoored=b,
        clean=c,
        target_label=np.array([x['target_label'] for x in batch.info]),
    ), inverse


def augment_list_of_params(params: List[dict],
                           numpy_rng: np.random.Generator,
                           layers_to_permute: list):
    """Augment a list of nets with random augmentations"""
    return [nnaugment.random_permutation(
        p, layers_to_permute=layers_to_permute, rng=numpy_rng) for p in params]


def augment_batch(
        batch: ParamsDataTree, 
        rng: np.random.Generator,
        layers_to_permute: list,
    ) -> dict:
    """Augment a batch of nets with permutation augmentations."""
    c = utils.clone_numpy_rng
    return ParamsDataTree(
        augment_list_of_params(batch.backdoored, c(rng), layers_to_permute),
        augment_list_of_params(batch.clean, c(rng), layers_to_permute),
        info=batch.info,
    )



class DataLoader:
    def __init__(
            self,
            data: ParamsDataTree,
            batch_size: int,
            data_std: float,
            rng: np.random.Generator = None, 
            max_workers: int = None,
            augment: bool = False,
            skip_last_batch: bool = True,
            layers_to_permute: list = None,
            chunk_size: int = 256,
        ):
        self.data = data
        self.batch_size = batch_size
        self.data_std = data_std
        self.rng = rng
        self.max_workers = max_workers
        self.augment = augment
        self.skip_last_batch = skip_last_batch
        self.layers_to_permute = layers_to_permute
        self.chunk_size = chunk_size

        self.batches = self.init_data_iterator()
        self.len = len(self.data) // self.batch_size + (not self.skip_last_batch)

    def process_batch(self, batch):
        if self.augment:
            batch = augment_batch(
                batch, self.rng, self.layers_to_permute)
        return flatten_and_chunk_batch(
            batch, self.chunk_size, self.data_std)[0]
    
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
            return self.process_batch(next(self.batches))
        except StopIteration:
            self.batches = self.init_data_iterator()
            raise StopIteration

    def __len__(self):
        return len(self.data) // self.batch_size + (not self.skip_last_batch)

    def shuffle(self):
        """Shuffle without creating a copy."""
        clone = utils.clone_numpy_rng(self.rng)
        self.rng.shuffle(self.data.backdoored)
        clone.shuffle(self.data.clean)
        del clone


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