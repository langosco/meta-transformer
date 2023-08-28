import pytest
import jax
import jax.numpy as jnp
import chex
from meta_transformer.preprocessing import augment_list_of_nets
import numpy.testing as npt
from meta_transformer import on_cluster, module_path, torch_utils
import os
import numpy as np


# load checkpoint data
if not on_cluster:
    dpath = os.path.join(module_path, "data/david_backdoors")  # local
else:
    dpath = "/rds/user/lsl38/rds-dsk-lab-eWkDxBhxBrQ/model-zoo/"  
dpath = os.path.join(dpath, "cifar10")
inputs_dir = os.path.join(dpath, "poison_noL1")
targets_dir = os.path.join(dpath, "clean")
architecture = torch_utils.CNNMedium()
inputs, targets, get_pytorch_model = torch_utils.load_pairs_of_models(
    model=architecture,
    data_dir1=inputs_dir,
    data_dir2=targets_dir,
    num_models=100,
    max_workers=None if on_cluster else 1,
)


rng = np.random.default_rng(42)
LAYERS_TO_PERMUTE = [f'Conv2d_{i}' for i in range(6)] + ['Dense_6']


def shapes(net):
    s = {}
    for k, v in net.items():
        try:
            s[k] = v.shape
        except AttributeError:
            s[k] = shapes(v)


@pytest.mark.parametrize("nets", [inputs, targets])
def test_augment(nets):
    augmented = augment_list_of_nets(
        nets,
        rng,
        layers_to_permute=LAYERS_TO_PERMUTE,
    )

    assert len(augmented) == len(nets)
    assert all(aug.keys() == net.keys() for aug, net in zip(augmented, nets)), (
        "Augmented and original networks have different keys (layer names)"
    )
    assert all(shapes(aug) == shapes(net) for aug, net in zip(augmented, nets)), (
        "Layers in augmented networks have different shapes from original networks."
    )
