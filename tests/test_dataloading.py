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


def test_data_loading(test_architecture, 
                      test_checkpoint_data,
                      layers_to_permute):
    inputs, targets = test_checkpoint_data
    chex.assert_tree_all_close(inputs[0], targets[0])

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
