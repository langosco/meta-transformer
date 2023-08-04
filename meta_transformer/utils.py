import jax
import jax.numpy as jnp
from jax import jit, vmap, random
import numpy as np
from typing import Optional, Mapping, Any, Dict, Tuple
from jax.typing import ArrayLike


# Parameter chunking
def pad_to_chunk_size(arr: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    pad_size = -len(arr) % chunk_size
    padded = jnp.pad(arr, (0, pad_size))
    return padded


def chunk_layer(weights: jnp.ndarray, biases: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    flat_weights = weights.flatten()
    flat_weights = pad_to_chunk_size(flat_weights, chunk_size)
    weight_chunks = jnp.split(flat_weights, len(flat_weights) // chunk_size)
    biases = pad_to_chunk_size(biases, chunk_size)
    bias_chunks = jnp.split(biases, len(biases) // chunk_size)
    return weight_chunks, bias_chunks


def chunk_params(params: dict, chunk_size: int) -> dict:
    """
    Chunk the parameters of an MLP into chunks of size chunk_size.
    Chunks don't cross layer boundaries and are padded with zeros if necessary.
    """
    return {
        k: chunk_layer(layer["w"], layer["b"], chunk_size) for k, layer in params.items()
    }


def count_params(params: dict) -> int:
    """Count the number of parameters in a pytree of parameters."""
    return sum([x.size for x in jax.tree_util.tree_leaves(params)])


def tree_list(trees):
    """Maps a list of trees to a tree of lists."""
    return jax.tree_map(lambda *x: list(x), *trees)


def tree_stack(trees):
    """Stacks a list of trees into a single tree with an extra dimension."""
    return jax.tree_map(lambda *x: jnp.stack(x), *trees)


def tree_unstack(tree):
    """Unstacks a tree with an extra dimension into a list of trees."""
    trees = []
    i = 0
    tree = tree_to_numpy(tree)  # to make sure we get an IndexError
    while True:
        try:
            trees.append(jax.tree_map(lambda x: x[i], tree))
            i += 1
        except IndexError:
            break
    return trees


def tree_to_numpy(tree):
    """Converts a tree of arrays to a tree of numpy arrays."""
    return jax.tree_map(lambda x: np.array(x), tree)


def flatten_dict(d: dict, parent_key: str = '', sep: str = '__') -> dict:
    flat = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_dict(v, new_key, sep=sep))
        else:
            flat[new_key] = v
    return flat


# Checkpointing
import orbax.checkpoint
from etils import epath
import os
MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHECKPOINTS_DIR = epath.Path(MODULE_PATH) / "experiments/checkpoints"
checkpointer = orbax.checkpoint.PyTreeCheckpointer()


def save_checkpoint(params, name="test", path=CHECKPOINTS_DIR):
    savedir = path / name
    params = {k.replace("/", "::"): v for k, v in params.items()}
    checkpointer.save(savedir, params)
    return


def load_checkpoint(name="test", path=CHECKPOINTS_DIR):
    savedir = path / name
    params = checkpointer.restore(savedir)
    params = {k.replace("::", "/"): v for k, v in params.items()}
    return params