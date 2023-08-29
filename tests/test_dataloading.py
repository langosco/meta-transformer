import pytest
import os
import jax
from meta_transformer import torch_utils, preprocessing
import chex
import numpy as np

rng = np.random.default_rng(42)

@pytest.fixture
def test_architecture():
    return torch_utils.CNNMedium()


def test_data_loading(test_checkpoint_data):
    inputs, targets = test_checkpoint_data
    assert len(inputs) == len(targets), "Length of inputs and targets is not equal."
    for inp, tar in zip(inputs, targets):
        chex.assert_tree_all_close(inputs[0], targets[0])


def test_data_loading_multiproc(test_checkpoint_data_multiproc):
    inputs, targets = test_checkpoint_data_multiproc
    assert len(inputs) == len(targets), "Length of inputs and targets is not equal."
    for inp, tar in zip(inputs, targets):
        chex.assert_tree_all_close(inputs[0], targets[0])


def test_data_processing(test_architecture, 
                      test_checkpoint_data,
                      layers_to_permute):
    inputs, targets = test_checkpoint_data
    loader = preprocessing.DataLoader(inputs, targets,
                                  batch_size=2,
                                  rng=rng,
                                  max_workers=None,
                                  augment=True,
                                  layers_to_permute=layers_to_permute,
                                  skip_last_batch=True)

    for batch in loader:
        for inp, tar in zip(batch["input"], batch["target"]):
            chex.assert_trees_all_close(inp, tar)
