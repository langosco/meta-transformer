import pytest
import jax
import jax.numpy as jnp
import chex
from meta_transformer.preprocessing import augment_list_of_nets
import numpy.testing as npt
from meta_transformer import on_cluster, module_path, torch_utils
import os
import numpy as np


@pytest.fixture(scope="module")
def data_dir():
    if not on_cluster:
        dpath = os.path.join(module_path, "data/david_backdoors")  # local
    else:
        dpath = "/rds/user/lsl38/rds-dsk-lab-eWkDxBhxBrQ/model-zoo/"  
    return dpath


@pytest.fixture(scope="module")
def layers_to_permute():
    return [f'Conv2d_{i}' for i in range(6)] + ['Dense_6']


@pytest.fixture(scope="module")
def test_checkpoint_data(data_dir):
    data_dir = os.path.join(data_dir, "test")
    inputs_dir = os.path.join(data_dir, "inputs")
    targets_dir = os.path.join(data_dir, "targets")
    architecture = torch_utils.CNNMedium()
    inputs, targets, get_pytorch_model = torch_utils.load_pairs_of_models(
        model=architecture,
        data_dir1=inputs_dir,
        data_dir2=targets_dir,
        num_models=10,
        max_workers=None if on_cluster else 1,
        prefix1="clean",
        prefix2="clean",
    )
    return inputs, targets
