from typing import Dict, Sequence, Tuple
from jax.typing import ArrayLike
import pytest
import jax
import haiku as hk
import jax.numpy as jnp
import chex
import numpy as np

from meta_transformer import utils
from meta_transformer.meta_model import ChunkCNN, UnChunkCNN, \
    NetEmbedding, NetUnEmbedding, MetaModelClassifierConfig, \
    create_meta_model_classifier


# Define a simple model
def net_fn(x: ArrayLike) -> ArrayLike:
    net = hk.Sequential([
        hk.Conv2D(32, kernel_shape=3, stride=1, padding='SAME'),
        jax.nn.relu,
        hk.MaxPool(window_shape=(1,2,2,1), strides=(1,2,2,1), padding='SAME'),
        hk.Flatten(),
        hk.Linear(10)
    ])
    return net(x)

# Transform the model
net = hk.transform(net_fn)


def get_param_shapes(
        params: Dict[str, Dict[str, ArrayLike]]) -> Dict[str, Dict[str, Tuple[int, ...]]]:
    return {
        k: {subk: v.shape for subk, v in layer.items()}
        for k, layer in params.items()
    }


@pytest.fixture
def model_params() -> Dict[str, Dict[str, ArrayLike]]:
    # Generate some data
    x = jnp.ones((1, 28, 28, 1))

    # Initialize the model
    params = net.init(jax.random.PRNGKey(42), x)
    return params


def test_chunk_unchunk(model_params: Dict[str, Dict[str, ArrayLike]]):
    # Define the ChunkCNN and UnChunkCNN instances
    chunk = ChunkCNN(2000, 500)
    unchunk = UnChunkCNN(original_shapes=get_param_shapes(model_params))

    # Chunk the parameters
    chunked_params = chunk(model_params)

    # Unchunk the parameters
    unchunked_params = unchunk(chunked_params)

    # Test if the original parameters and the unchunked parameters are the same
    chex.assert_tree_all_close(model_params, unchunked_params, rtol=1e-7, atol=1e-7)


def test_embed_unembed(model_params: dict):
    # Embed params
    @hk.without_apply_rng
    @hk.transform
    def embed_params(params_batch):
        embed = NetEmbedding(embed_dim=8)
        return embed(params_batch)

    embedding_weights = embed_params.init(
        jax.random.PRNGKey(42), model_params)
    embedded_params = embed_params.apply(embedding_weights, model_params)


    # Unbembedding
    @hk.without_apply_rng
    @hk.transform
    def unembed_params(embedded_params):
        unembed = NetUnEmbedding(
            original_param_shapes=get_param_shapes(model_params))
        return unembed(embedded_params)


    unbembedding_weights = unembed_params.init(jax.random.PRNGKey(42), embedded_params)
    unembedded_params = unembed_params.apply(unbembedding_weights, embedded_params)

    chex.assert_trees_all_equal_shapes(model_params, unembedded_params)


def test_meta_model_classifier_init(model_params: dict):
    model = create_meta_model_classifier(MetaModelClassifierConfig())
    batch = utils.tree_stack([model_params] * 2)
    model.init(jax.random.PRNGKey(42), batch)
