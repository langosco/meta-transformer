import jax
import jax.numpy as jnp
import numpy as np


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
    """Maps a nested dict to a flat dict by concatenating the keys."""
    flat = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_dict(v, new_key, sep=sep))
        else:
            flat[new_key] = v
    return flat


def get_activation_stats(x):
    return {"std": x.std(), "l1": jnp.abs(x).mean()}


def clone_numpy_rng(rng: np.random.Generator) -> np.random.Generator:
    """Clone a numpy random number generator."""
    rng_state = rng.bit_generator.state
    new_rng = np.random.default_rng()
    new_rng.bit_generator.state = rng_state
    return new_rng