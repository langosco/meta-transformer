import pytest
import jax.random
import jax.numpy as jnp
import numpy as np
from meta_transformer import data
import nnaugment
from backdoors import paths, models
import backdoors.utils


poison_type = "simple_pattern"
poisoned_data = data.load_batches(paths.PRIMARY_BACKDOOR / poison_type,
                                  max_datapoints=5)
clean_data = data.load_batches(paths.PRIMARY_CLEAN,
                               max_datapoints=5)
all_params = [x["params"] for x in poisoned_data + clean_data]
all_params = all_params[:3]

batch_size = 5
rng = jax.random.PRNGKey(0)
subrng, rng = jax.random.split(rng)
inputs = jax.random.normal(subrng, (batch_size, 32, 32, 3))
rng, subrng = jax.random.split(rng)


rngs = jax.random.split(subrng, len(all_params))
@pytest.mark.parametrize("rng, params", zip(rngs, all_params))
def test_weight_augmentation(rng, params, model, layers_to_permute):
    augmented_params = nnaugment.random_permutation(
        rng,
        params=params,
        layers_to_permute=layers_to_permute,
        sort=True,
    )

    # Check non-equality of augmented model's parameters against the original
    for name, layer in augmented_params.items():
        if name in layers_to_permute:
            assert not np.allclose(layer["kernel"], params[name]["kernel"], rtol=5e-2), \
                f"Parameters {name} are almost identical after augmentation."
            assert not np.allclose(layer["kernel"], params[name]["kernel"], rtol=5e-2), \
                f"Kernel parameters {name} are almost identical after augmentation."


            assert not np.allclose(layer["kernel"], params[name]["kernel"], rtol=0.2), \
                f"Kernel parameters {name} are within +-20% size of each other after augmentation."
            assert not np.allclose(layer["bias"], params[name]["bias"], rtol=0.2), \
                f"Bias parameters {name} are within +-20% size of each other after augmentation."

    vanilla_out = model.apply({"params": params}, inputs)
    perm_out = model.apply({"params": augmented_params}, inputs)

    # Check that the outputs are the same.
    assert jnp.allclose(vanilla_out, perm_out, rtol=5e-2, atol=1e-4), \
        ("Outputs differ after weight augmentation. "
            f"Differences: {jnp.abs(vanilla_out - perm_out)}")



@pytest.mark.parametrize("params", all_params)
def test_determinism(params, layers_to_permute):
    """Test that the augmentation is deterministic, ie I can
    get the same permutation by setting the seed."""

    rng = jax.random.PRNGKey(0)

    augmented_params_0 = nnaugment.random_permutation(
        rng=rng,
        params=params,
        layers_to_permute=layers_to_permute,
        sort=True,
    )


    augmented_params_1 = nnaugment.random_permutation(
        rng=rng,
        params=params,
        layers_to_permute=layers_to_permute,
        sort=True,
    )

    for name, layer in augmented_params_0.items():
        comparison_layer = augmented_params_1[name]
        for k in layer.keys():
            assert np.allclose(layer[k], comparison_layer[k], rtol=1e-3), \
                f"Parameters {k} in {name} differ despite same seed."


def _augment_and_acc(rng,
                     params: dict, 
                     model, 
                     layers_to_permute, 
                     data):
    """Augment params, then compute acc for both augmented and non-augmented
    on the same data."""
    augmented_params = nnaugment.random_permutation(
        rng=rng,
        params=params,
        layers_to_permute=layers_to_permute,
        sort=True,
    )

    logits = model.apply({"params": params}, data.image)
    aug_logits = model.apply({"params": augmented_params}, data.image)

    acc = backdoors.utils.accuracy(logits, data.label)
    aug_acc = backdoors.utils.accuracy(aug_logits, data.label)
    return acc, aug_acc


def test_augment_poisoned_params(rng,
                                 poisoned_params,
                                 model,
                                 layers_to_permute,
                                 poisoned_and_filtered_data):

    for x in poisoned_params:
        p = x["params"]
        target = x["info"]["target_label"]
        base_data = poisoned_and_filtered_data[target]
        subrng, rng = jax.random.split(rng)
        asr, aug_asr = _augment_and_acc(
            subrng, p, model, layers_to_permute, base_data)

        assert asr > 0.9, ( "Attack success rate should be above 90%, but "
                            f"was {asr} before augmentation.")

        assert aug_asr > 0.9, ( "Attack success after augmentation too low. ASR "
                            "should be above 90%, but was {asr}.")
        
        assert jnp.abs(asr - aug_asr) < 0.01, ("ASR is not the same "
                                    "before and after augmentation. "
                                    f"Before: {asr}, after: {aug_asr}.")


def test_augment_clean_params(rng,
                              clean_params,
                              model,
                              layers_to_permute,
                              cifar10_test):
    for x in clean_params:
        p = x["params"]
        rng, subrng = jax.random.split(rng)
        acc, aug_acc = _augment_and_acc(
            subrng, p, model, layers_to_permute, cifar10_test)
        
        assert acc > 0.8, ( "Accuracy should be above 80%, but "
                            f"was {acc} before augmentation.")
        assert aug_acc > 0.8, ( "Accuracy after augmentation too low. "
                               "ASR should be above 80%, but was {acc}.")

        assert jnp.abs(acc - aug_acc) < 0.01, ("Accuracy is not the same "
                                    "before and after augmentation. "
                                    f"Before: {acc}, after: {aug_acc}.")