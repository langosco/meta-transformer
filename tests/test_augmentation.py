import pytest
import numpy as np
from meta_transformer.preprocessing import augment_list_of_params
from meta_transformer import on_cluster, data
from backdoors import checkpoint_dir
from pathlib import Path

# NOT TESTED YET


# load checkpoint data
checkpoint_dir = Path(checkpoint_dir)
dataset = data.load_clean_and_backdoored(
    num_pairs=100,
    backdoored_dir=checkpoint_dir / "simple_pattern",
    clean_dir=checkpoint_dir / "clean",
    max_workers=None if on_cluster else 1,
)

rng = np.random.default_rng(42)
LAYERS_TO_PERMUTE = [f'Conv_{i}' for i in range(6)] + ['Dense_6']


def shapes(net):
    s = {}
    for k, v in net.items():
        try:
            s[k] = v.shape
        except AttributeError:
            s[k] = shapes(v)


@pytest.mark.parametrize("param_datapoints", [dataset])
def test_augment(param_datapoints: data.ParamsTreeData):
    augmented = augment_list_of_params(
        param_datapoints,
        rng,
        layers_to_permute=LAYERS_TO_PERMUTE,
    )

    assert len(augmented) == len(param_datapoints)
    assert all(aug.keys() == net.keys() for aug, net in zip(augmented, param_datapoints)), (
        "Augmented and original networks have different keys (layer names)"
    )
    assert all(shapes(aug) == shapes(net) for aug, net in zip(augmented, param_datapoints)), (
        "Layers in augmented networks have different shapes from original networks."
    )
