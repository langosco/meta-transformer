import pytest
import jax
import jax.numpy as jnp
import haiku as hk
import chex
from meta_transformer.preprocessing import pad_to_chunk_size, filter_layers, preprocess, skip_layer
import numpy.testing as npt


# Initialize a simple CNN with Haiku
def net_fn(x):
    model = hk.Sequential([
        hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        #hk.MaxPool(window_shape=2, strides=2, padding='SAME'),
        hk.MaxPool(window_shape=(2, 2, 1), strides=(2, 2, 1), padding='SAME'),
        hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        #hk.MaxPool(window_shape=2, strides=2, padding='SAME'),
        hk.MaxPool(window_shape=(2, 2, 1), strides=(2, 2, 1), padding='SAME'),
        hk.Flatten(),
        hk.Linear(10),
    ])
    return model(x)


net = hk.without_apply_rng(hk.transform(net_fn))
params = net.init(jax.random.PRNGKey(42), jnp.ones([1, 28, 28, 1]))


def test_skip_layer():
    assert skip_layer('dropout') == True
    assert skip_layer('layer_norm') == True
    assert skip_layer('batch_norm') == True
    assert skip_layer('conv_layer') == False
    assert skip_layer('linear') == False


def test_pad_to_chunk_size():
    arr = jnp.array([1, 2, 3, 4, 5])
    chunk_size = 3

    padded = pad_to_chunk_size(arr, chunk_size)

    assert len(padded) % chunk_size == 0
    assert jnp.all(padded[len(arr):] == 0)


def test_filter_layers_round_trip():
    filtered_params, unfilter = filter_layers(params)
    round_trip_params = unfilter(filtered_params)
    chex.assert_tree_all_close(round_trip_params, params, rtol=1e-7)


def test_preprocess_round_trip():
    chunk_size = 3
    chunks, unprocess = preprocess(params, chunk_size)
    round_trip_params = unprocess(chunks)
    chex.assert_tree_all_close(round_trip_params, params, rtol=1e-7)


def test_filter_layers_empty():
    with pytest.raises(ValueError):
        filter_layers({})


def test_preprocess_various_chunk_sizes():
    for chunk_size in [1, 2, 10, 50, 100]:
        chunks, unprocess = preprocess(params, chunk_size)
        assert chunks.shape[1] == chunk_size


def test_preprocess_operation_reconstruct_variant():
    chunk_size = 3
    chunks, unprocess = preprocess(params, chunk_size)

    # Save a copy of the original chunks for later comparison
    original_chunks = chunks.copy()

    # Add a constant to all chunks
    modified_chunks = chunks + 5

    # Unprocess the modified chunks
    modified_params = unprocess(modified_chunks)

    # Check that the values in the modified_params have increased by 5
    # We'll use tree_map to apply the check to every leaf in the tree
    def check_increase(original, modified):
        npt.assert_allclose(original + 5, modified, rtol=1e-7)

    jax.tree_map(check_increase, params, modified_params)

    # Check that unprocess did not mutate its input
    chex.assert_trees_all_close(original_chunks, chunks, rtol=1e-7)

    # Check that skipped layers are the same in the original and modified params
    for layer_name in ['dropout', 'layer_norm', 'batch_norm']:
        if layer_name in params:
            chex.assert_trees_all_close(params[layer_name], modified_params[layer_name], rtol=1e-7)

    # Check dtype consistency
    chex.assert_trees_all_equal_dtypes(params, modified_params)

    # Check that all values are finite
    chex.assert_tree_all_finite(params)
    chex.assert_tree_all_finite(modified_params)

    # Check shape consistency
    chex.assert_trees_all_equal_shapes(params, modified_params)