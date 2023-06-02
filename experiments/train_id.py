import jax
from jax import random, jit, value_and_grad, nn
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import chex
import functools
from typing import Mapping, Any, Tuple, List, Iterator, Optional, Dict
from jax.typing import ArrayLike
from meta_transformer import utils, preprocessing
from meta_transformer.meta_model import create_meta_model
from meta_transformer.meta_model import MetaModelConfig as ModelConfig
import wandb
from nninn.repl.utils import load_nets, classes_per_task
import nninn
import os
import argparse
from dataclasses import asdict


VAL_DATA_RATIO = 0.1


def flatten(x):
    return jax.flatten_util.ravel_pytree(x)[0]


# TODO replace all this with huggingface datasets
def split_data(data: list, labels: list):
    split_index = int(len(data)*(1-VAL_DATA_RATIO))
    return (data[:split_index], labels[:split_index], 
            data[split_index:], labels[split_index:])


def is_fine(params: dict):
    """Return false if std or mean is too high."""
    flat = flatten(params)
    if flat.std() > 5.0 or jnp.abs(flat.mean()) > 5.0:
        return False
    else:
        return True


def filter_data(data: List[dict], labels: List[ArrayLike]):
    """Given a list of net params, filter out those
    with very large means or stds."""
    assert len(data) == len(labels)
    f_data, f_labels = zip(*[(x, y) for x, y in zip(data, labels) if is_fine(x)])
    print(f"Filtered out {len(data) - len(f_data)} nets.\
          That's {100*(len(data) - len(f_data))/len(data):.2f}%.")
    return np.array(f_data), np.array(f_labels)


def acc_from_outputs(outputs, targets):
    return None


def loss_from_outputs(outputs, targets):
    """MSE between flattened trees"""
    return jnp.mean((outputs - targets)**2)


def create_loss_fn(model_forward: callable):
    """model_forward = model.apply if model is a hk.transform"""
    def loss_fn(params, rng, data: Dict[str, ArrayLike], is_training: bool = True):
        """data is a dict with keys 'input' and 'label'."""
        inputs, _ = data["input"], data["label"]
        param_out = model_forward(params, rng, inputs, is_training)  # dict of param batches
        loss = loss_from_outputs(param_out, inputs)
        return loss#, {}
    return loss_fn


# Optimizer and update function
@chex.dataclass
class TrainState:
    step: int
    rng: random.PRNGKey
    opt_state: optax.OptState
    params: dict


@chex.dataclass(frozen=True)  # needs to be immutable to be hashable (for static_argnums)
class Updater: # Could also make this a function of loss_fn, model.apply, etc if we want to be flexible
    """Holds training methods. All methods are jittable."""
    opt: optax.GradientTransformation
    model: hk.TransformedWithState
    loss_fn: callable

    def get_loss(self, rng, params, data):
        """Compute acc and loss on test set."""
        loss = self.loss_fn(params, rng, data, is_training=False)
        return {"val/loss": loss}

    @functools.partial(jit, static_argnums=0)
    def init_params(self, rng: ArrayLike, data: dict) -> dict:
        """Initializes state of the updater."""
        out_rng, subkey = jax.random.split(rng)
        params = self.model.init(subkey, data["input"])
        opt_state = self.opt.init(params)
        return TrainState(
            step=0,
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
    
    @functools.partial(jit, static_argnums=0)
    def update(self, state: TrainState, data: dict) -> Tuple[TrainState, dict]:
        state.rng, subkey = jax.random.split(state.rng)
        loss, grads = value_and_grad(self.loss_fn)(
                state.params, subkey, data)
        updates, state.opt_state = self.opt.update(
            grads, state.opt_state, state.params)
        state.params = optax.apply_updates(state.params, updates)
        metrics = {
                "train/loss": loss,
                "step": state.step,
        }
        state.step += 1
        return state, metrics

    @functools.partial(jit, static_argnums=0)
    def compute_val_metrics(self, 
                            state: TrainState, 
                            data: dict) -> Tuple[TrainState, dict]:
        state.rng, subkey = random.split(state.rng)
        return state, self.get_loss(subkey, state.params, data)


@chex.dataclass
class Logger:
    log_interval: int = 50
    disable_wandb: bool = False

    def log(self,
            state: TrainState,
            train_metrics: dict = None,
            val_metrics: dict = None):
        metrics = train_metrics or {}
        if val_metrics is not None:
            metrics.update(val_metrics)
        metrics = {k: v.item() if isinstance(v, jnp.ndarray) else v
                   for k, v in metrics.items()}
        if state.step % self.log_interval == 0 or val_metrics is not None:
            print(", ".join([f"{k}: {round(v, 3)}" for k, v in metrics.items()]))
            if not self.disable_wandb:
                wandb.log(metrics, step=state.step)


def shuffle_data(
        rng: ArrayLike, 
        inputs: ArrayLike, 
        labels: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """Shuffle the data."""
    idx = jnp.arange(len(inputs))
    idx = random.permutation(rng, idx)
    return inputs[idx], labels[idx]


def data_iterator(
        inputs: ArrayLike, 
        labels: ArrayLike, 
        batchsize: int = 1048, 
        skip_last: bool = False
        ) -> Iterator[Tuple[ArrayLike, ArrayLike]]:
    """Iterate over the data in batches."""
    for i in range(0, len(inputs), batchsize):
        if skip_last and i + batchsize > len(inputs):
            break
        yield dict(input=inputs[i:i + batchsize], 
                   label=labels[i:i + batchsize])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--lr', type=float, help='Learning rate', default=5e-5)
    parser.add_argument('--wd', type=float, help='Weight decay', default=5e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', 
                        default=25)
    parser.add_argument('--task', type=str, help='Task to train on. One of \
                        "batch_size", "augmentation", "optimizer", \
                        "activation", "initialization"', default="batch_size")
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    args = parser.parse_args()


    rng = random.PRNGKey(42)
    LEARNING_RATE = args.lr  # TODO don't use extra variables
    WEIGHT_DECAY = args.wd
    BATCH_SIZE = args.bs
    NUM_EPOCHS = args.epochs
    USE_WANDB = args.use_wandb
    DATASET = "ctc_fixed_10k"
    FILTER = False

    CHUNK_SIZE = 1024
    D_MODEL = 512
    LOG_INTERVAL = 5

    NOTES = "Now using encoder / decoder instead of linear."

    # Load MNIST model checkpoints
    print("Loading data...")
    all_inputs, all_labels = load_nets(
        n=10000, data_dir=os.path.join(nninn.data_dir, DATASET), verbose=False);
    unpreprocess = preprocessing.get_unpreprocess(all_inputs[0], CHUNK_SIZE)
    print("Preprocessing data...")
    all_inputs = [preprocessing.preprocess(inp, CHUNK_SIZE)[0]
                  for inp in all_inputs]
    all_inputs = np.stack(all_inputs, axis=0)
    labels = all_labels["batch_size"]  # task doesn't matter
    if FILTER:
        all_inputs, labels = filter_data(all_inputs, labels)
    else:
        all_inputs, labels = np.array(all_inputs), np.array(labels)  # TODO do this in dataloader instead
    train_inputs, train_labels, val_inputs, val_labels = split_data(
        all_inputs, labels)
    print("Done.")


    model_config = ModelConfig(
        model_size=D_MODEL,
        num_heads=8,
        num_layers=12,
        dropout_rate=0.0,
        use_embedding=True,
    )


    wandb.init(
        mode="online" if USE_WANDB else "disabled",
        project="meta-models-backdoors",
        tags=[],
        config={
            "dataset": "MNIST-meta",
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batchsize": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "dataset": DATASET,
            "model_config": asdict(model_config),
        },
        notes=NOTES,
        )  # TODO properly set and log config

    steps_per_epoch = len(train_inputs) // BATCH_SIZE

    print()
    print(f"Number of training examples: {len(train_inputs)}.")
    print("Steps per epoch:", steps_per_epoch)
    print("Total number of steps:", steps_per_epoch * NUM_EPOCHS)
    print()


    # Initialization
    opt = optax.adamw(LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model = create_meta_model(model_config)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)
    logger = Logger(log_interval=LOG_INTERVAL)
    rng, subkey = random.split(rng)
    state = updater.init_params(subkey, {
        "input": train_inputs[:2],
        "label": train_labels[:2],
        })

    print("Number of parameters:",
           utils.count_params(state.params) / 1e6, "Million")


    # Training loop
    for epoch in range(NUM_EPOCHS):
        rng, subkey = random.split(rng)

        # Prepare data
        inputs, labels = shuffle_data(subkey, train_inputs, train_labels)
        train_batches = data_iterator(
            inputs, labels, batchsize=BATCH_SIZE, skip_last=True)
        val_batches = data_iterator(
            val_inputs, val_labels, batchsize=BATCH_SIZE, skip_last=False)

        # Validate every epoch
        valdata = []
        for batch in val_batches:
            state, val_metrics_dict = updater.compute_val_metrics(
                state, batch)
            val_metrics_dict.update({"epoch": epoch})
            valdata.append(val_metrics_dict)

        means = jax.tree_map(lambda *x: np.mean(x), *valdata)
        logger.log(state, means)

        # Train
        for batch in train_batches:
            state, train_metrics = updater.update(state, batch)
            train_metrics.update({"epoch": epoch})
            logger.log(state, train_metrics)