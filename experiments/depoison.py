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
from meta_transformer import utils, preprocessing, torch_utils, module_path
from meta_transformer.meta_model import create_meta_model
from meta_transformer.meta_model import MetaModelConfig as ModelConfig
import wandb
from nninn.repl.utils import load_nets, classes_per_task
import nninn
import os
import argparse
from dataclasses import asdict
from meta_transformer.train import Updater, Logger
from meta_transformer.data import data_iterator, shuffle_arrays, split_data


VAL_DATA_RATIO = 0.1
DATA_STD = 0.0582


def acc_from_outputs(outputs, targets):
    return None


def loss_from_outputs(outputs, targets):
    """MSE between flattened trees"""
    return jnp.mean((outputs - targets)**2)


def create_loss_fn(model_forward: callable):
    """model_forward = model.apply if model is a hk.transform"""
    def loss_fn(params, rng, data: Dict[str, ArrayLike], is_training: bool = True):
        """data is a dict with keys 'input' and 'target'."""
        outputs = model_forward(params, rng, data["input"], is_training)
        loss = loss_from_outputs(outputs, data["target"])
        return loss, {}
    return loss_fn


def load_and_process_nets(name: str, n: int):
    path_to_processed = os.path.join(
            module_path, "data/cache/depoisoning", name)
    os.makedirs(os.path.dirname(path_to_processed), exist_ok=True)

    if os.path.exists(path_to_processed) and n == 10000:
        inputs = np.load(path_to_processed)
    else:
        inputs = torch_utils.load_pytorch_nets(
            n=n, data_dir=os.path.join(DATA_DIR, name)
        )
        unpreprocess = preprocessing.get_unpreprocess(inputs[0], CHUNK_SIZE)
        inputs = np.stack([preprocessing.preprocess(inp, CHUNK_SIZE)[0]
                      for inp in inputs])

    if n == 10000:
        np.save(path_to_processed, inputs)

    return inputs / DATA_STD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--lr', type=float, help='Learning rate', default=2e-5)
    parser.add_argument('--wd', type=float, help='Weight decay', default=5e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', 
                        default=25)
    parser.add_argument('--task', type=str, help='Task to train on. One of \
                        "batch_size", "augmentation", "optimizer", \
                        "activation", "initialization"', default="batch_size")
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    parser.add_argument('--ndata', type=int, help='Number of data points',
                        default=10000)
    args = parser.parse_args()

    rng = random.PRNGKey(42)

    FILTER = False
    CHUNK_SIZE = 1024
    D_MODEL = 512
    LOG_INTERVAL = 5
    DATA_DIR = os.path.join(module_path, 'data/david_backdoors/cifar10')

    NOTES = "Fixed the terrible bug, I think"
    TAGS = ["test"]

    # Load model checkpoints
    print("Loading data...")
    inputs = load_and_process_nets(name="poison_easy", n=args.ndata)
    targets = load_and_process_nets(name="clean", n=args.ndata)

    if FILTER:
        inputs, targets = preprocessing.filter_data(inputs, targets)
    (train_inputs, train_targets, 
        val_inputs, val_targets) = split_data(inputs, targets)
    print("Done.")


    model_config = ModelConfig(
        model_size=D_MODEL,
        num_heads=8,
        num_layers=12,
        dropout_rate=0.0,
        use_embedding=True,
    )


    wandb.init(
        mode="online" if args.use_wandb else "disabled",
        project="meta-models-depoison",
        tags=[],
        config={
            "dataset": "MNIST-meta",
            "lr": args.lr,
            "weight_decay": args.wd,
            "batchsize": args.bs,
            "num_epochs": args.epochs,
            "dataset": DATA_DIR,
            "model_config": asdict(model_config),
        },
        notes=NOTES,
        )  

    steps_per_epoch = len(train_inputs) // args.bs

    print()
    print(f"Number of training examples: {len(train_inputs)}.")
    print("Steps per epoch:", steps_per_epoch)
    print("Total number of steps:", steps_per_epoch * args.epochs)
    print()


    # Initialization
    opt = optax.adamw(args.lr, weight_decay=args.wd)
    model = create_meta_model(model_config)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)
    logger = Logger(log_interval=LOG_INTERVAL)
    rng, subkey = random.split(rng)
    state = updater.init_params(subkey, {
        "input": train_inputs[:2],
        "target": train_targets[:2],
        })

    print("Number of parameters:",
           utils.count_params(state.params) / 1e6, "Million")


    np_rng = np.random.default_rng()

    # Training loop
    for epoch in range(args.epochs):
        rng, subkey = random.split(rng)

        # Prepare data
        # too expensive to shuffle in memory
        # shuff_inputs, shuff_targets = shuffle_data(subkey, train_inputs, train_targets)

        # shuffle separately, should not work!!!
#        np_rng.shuffle(train_inputs)
#        np_rng.shuffle(train_targets)

        train_batches = data_iterator(
            train_inputs, train_targets, batchsize=args.bs, skip_last=True)
        val_batches = data_iterator(
            val_inputs, val_targets, batchsize=args.bs, skip_last=False)

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
