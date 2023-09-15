import pytest
import jax.random
from backdoors import paths
from meta_transformer import data
from backdoors import models
import backdoors.data
import backdoors.poison


@pytest.fixture(scope="module")
def layers_to_permute():
    return [f'Conv_{i}' for i in range(6)]


@pytest.fixture(scope="module")
def poisoned_params():
    poison_type = "simple_pattern"
    return data.load_batches(paths.PRIMARY_BACKDOOR / poison_type,
                             max_datapoints=50)


@pytest.fixture(scope="module")
def clean_params():
    return data.load_batches(paths.PRIMARY_CLEAN, max_datapoints=50)


@pytest.fixture(scope="module")
def model():
    return models.CNN()


@pytest.fixture(scope="module")
def cifar10_test():
    return backdoors.data.load_cifar10(split="test")


@pytest.fixture(scope="module")
def poisoned_and_filtered_data(cifar10_test):
    return backdoors.poison.filter_and_poison_all(
        cifar10_test,
        target_label=range(10),
        poison_type="simple_pattern",
    )


@pytest.fixture(scope="module")
def rng():
    return jax.random.PRNGKey(42)