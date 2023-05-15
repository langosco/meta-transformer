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
from meta_transformer.meta_model import create_meta_model_classifier
from meta_transformer.meta_model import MetaModelClassifierConfig as ModelConfig
import wandb
from nninn.repl.utils import load_nets, classes_per_task
import nninn
import os
import argparse
from dataclasses import asdict
from meta_transformer.train import Updater, Logger
from meta_transformer.data import data_iterator, shuffle_arrays, split_data
import jax


VAL_DATA_RATIO = 0.1


def acc_from_logits(logits, targets):
    """expects index targets, not one-hot"""
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == targets)


def loss_from_logits(logits, targets):
    """targets are index labels"""
    targets = nn.one_hot(targets, logits.shape[-1])
    chex.assert_equal_shape([logits, targets])
    return -jnp.sum(targets * nn.log_softmax(logits, axis=-1), axis=-1).mean()


def create_loss_fn(model_forward: callable):
    """model_forward = model.apply if model is a hk.transform"""
    def loss_fn(params, rng, data: Dict[str, ArrayLike], is_training: bool = True):
        """data is a dict with keys 'input' and 'target'."""
        inputs, targets = data["input"], data["target"]
        logits = model_forward(params, rng, inputs, is_training)  # [B, C]
        loss = loss_from_logits(logits, targets)
        acc = acc_from_logits(logits, targets)
        return loss, {"acc": acc}
    return loss_fn



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
    parser.add_argument('--ndata', type=int, help='Number of data points',
                        default=10000)
    args = parser.parse_args()


    rng = random.PRNGKey(42)
    LEARNING_RATE = args.lr  # TODO don't use extra variables
    WEIGHT_DECAY = args.wd
    BATCH_SIZE = args.bs
    NUM_EPOCHS = args.epochs
    TASK = args.task
    USE_WANDB = args.use_wandb
    DATASET = "ctc_fixed_10k"
    FILTER = False

    CHUNK_SIZE = 512
    D_MODEL = 256

    NOTES = "Training on bigger dataset (10k)."

    # Load MNIST model checkpoints
    print(f"Training task: {TASK}.")
    print("Loading data...")
    all_inputs, all_labels = load_nets(
        n=args.ndata, data_dir=os.path.join(nninn.data_dir, DATASET), verbose=False);
    labels = all_labels[TASK]

    unpreprocess = preprocessing.get_unpreprocess(all_inputs[0], CHUNK_SIZE)
    print("Preprocessing data...")
    all_inputs = [preprocessing.preprocess(inp, CHUNK_SIZE)[0]
                  for inp in all_inputs]
    all_inputs = np.stack(all_inputs, axis=0)

    if FILTER:
        all_inputs, labels = preprocessing.filter_data(all_inputs, labels)
    else:
        all_inputs, labels = np.array(all_inputs), np.array(labels)  # TODO do this in dataloader instead
    train_inputs, train_labels, val_inputs, val_labels = split_data(
        all_inputs, labels)
    print("Done.")


#    gpt2_small_model_config = ModelConfig(  # for comparison
#        num_classes=classes_per_task[TASK],
#        model_size=768,  # for medium: 1024
#        num_heads=12,  # for medium: 16
#        num_layers=12,  # for medium: 24
#        dropout_rate=0.1,
#    )


#    model_config = ModelConfig(
#        num_classes=classes_per_task[TASK],
#        model_size=512,
#        num_heads=8,
#        num_layers=8,
#        dropout_rate=0.1,
#    )


    model_config = ModelConfig(
        num_classes=classes_per_task[TASK],
        model_size=D_MODEL,
        num_heads=4,
        num_layers=8,
        dropout_rate=0.1,
        use_embedding=True,
    )


    wandb.init(
        mode="online" if USE_WANDB else "disabled",
        project="meta-models",
        tags=[],
        config={
            "dataset": "MNIST-meta",
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batchsize": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "target_task": TASK,
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
    model = create_meta_model_classifier(model_config)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)
    logger = Logger(log_interval=5)
    rng, subkey = random.split(rng)
    state = updater.init_params(subkey, {
        "input": train_inputs[:2],
        "target": train_labels[:2]
        })

    print("Number of parameters:",
           utils.count_params(state.params) / 1e6, "Million")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        rng, subkey = random.split(rng)

        # Prepare data
        inputs, labels = shuffle_arrays(subkey, train_inputs, train_labels)
        batches = data_iterator(
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
        for batch in batches:
            state, train_metrics = updater.update(state, batch)
            train_metrics.update({"epoch": epoch})
            logger.log(state, train_metrics)
