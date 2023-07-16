import jax
from jax import random, jit, value_and_grad, nn
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
from typing import Mapping, Any, Tuple, List, Iterator, Optional, Dict
from jax.typing import ArrayLike
from meta_transformer import utils, preprocessing, torch_utils, module_path
from meta_transformer.meta_model import create_meta_model
from meta_transformer.meta_model import MetaModelConfig as ModelConfig
import wandb
import os
import argparse
from dataclasses import asdict
from meta_transformer.train import Updater, Logger
from meta_transformer.data import data_iterator, split_data
from augmentations import permute_checkpoint
#permute_checkpoint = lambda *args, **kwargs: [None]

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



LAYERS_TO_PERMUTE = ['Conv2d_0', 'Conv2d_1', 'Conv2d_2', 'Conv2d_3', 
          'Conv2d_4', 'Conv2d_5', 'Linear_6', 'Linear_7'] 


def augment_list_of_nets(nets: List[dict], seed):
    """Augment a list of nets with random augmentations"""
    rng = np.random.default_rng(seed)
    augmented_nets = []
    for net in nets:
        augmented = permute_checkpoint(
            rng,
            net,
            permute_layers=LAYERS_TO_PERMUTE,
            num_permutations=2,  # 2x batch size
        )
        del augmented[0]  # don't need the original
        augmented_nets += augmented
    return augmented_nets


def process_nets(nets: List[dict], 
                seed: int,
                augment: bool = True,
                 ) -> ArrayLike:
    """Permutation augment, then flatten to arrays."""
    nets = augment_list_of_nets(nets, seed) if augment else nets
    nets = np.stack([preprocessing.preprocess(net, CHUNK_SIZE)[0]
                        for net in nets])
    return nets / DATA_STD


def process_batch(batch: dict, seed: int, augment: bool = True) -> dict:
    """Process a batch of nets."""
    inputs = process_nets(batch["input"], seed, augment=augment)
    targets = process_nets(batch["target"], seed, augment=augment)
    return dict(input=inputs, target=targets)


import concurrent.futures
import jax.random as random


class DataLoader:
    def __init__(self, batches, seed, num_workers=4):
        self.batches = batches
        self.seed = seed
        self.num_workers = num_workers

    def __iter__(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for batch in self.batches:
                self.seed += 1
                future = executor.submit(process_batch, batch, self.seed, augment=False)
                yield future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--lr', type=float, help='Learning rate', default=2e-5)
    parser.add_argument('--wd', type=float, help='Weight decay', default=5e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', 
                        default=25)
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    parser.add_argument('--ndata', type=int, help='Number of data points',
                        default=10000)
    args = parser.parse_args()

    rng = random.PRNGKey(42)

    FILTER = False
    CHUNK_SIZE = 1024
    D_MODEL = 512
    LOG_INTERVAL = 5
    #DATA_DIR = os.path.join(module_path, 'data/david_backdoors/cifar10')
    DATA_DIR = os.path.join(module_path, 'data/david_backdoors/mnist/models')
    #INPUTS_DIRNAME = "poison_easy"  # for CIFAR-10
    INPUTS_DIRNAME = "poison"  # for MNIST
    TARGETS_DIRNAME = "clean"

#    architecture = torch_utils.CNNMedium()  # for CIFAR-10
    architecture = torch_utils.CNNSmall()  # for MNIST

#    NOTES = "Fixed the terrible bug, I think"
    NOTES = "testing"
    TAGS = ["test", "MNIST"]

    # Load model checkpoints
    print("Loading data...")
    inputs, targets = torch_utils.load_input_and_target_weights(
        model = architecture,
        num_models=args.ndata, 
        data_dir=DATA_DIR,
        inputs_dirname=INPUTS_DIRNAME,
    )

    if FILTER:
        inputs, targets = preprocessing.filter_data(inputs, targets)

    (train_inputs, train_targets, 
        val_inputs, val_targets) = split_data(inputs, targets, 0.9)
    print("Done.")


    model_config = ModelConfig(
        model_size=D_MODEL,
        num_heads=8,
        num_layers=12,
        dropout_rate=0.1,
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
    print(f"Number of validation examples: {len(val_inputs)}.")
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
    init_batch = {
        "input": train_inputs[:2],
        "target": train_targets[:2],
    }
    init_batch = process_batch(init_batch, 0, augment=False)
    state = updater.init_params(subkey, init_batch)

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
#        np_rng.shuffle(train_targets)

        train_batches = data_iterator(
            train_inputs, train_targets, batchsize=args.bs, skip_last=True)
        val_batches = data_iterator(
            val_inputs, val_targets, batchsize=args.bs, skip_last=False)

        train_loader = DataLoader(train_batches, 43, num_workers=4)


        # Validate every epoch
        if epoch % 1 == 0:
            valdata = []
            for batch in val_batches:
                rng, subkey = random.split(rng)
                batch = process_batch(batch, 0, augment=False)
                state, val_metrics_dict = updater.compute_val_metrics(
                    state, batch)
                val_metrics_dict.update({"epoch": epoch})
                valdata.append(val_metrics_dict)

            means = jax.tree_map(lambda *x: np.mean(x), *valdata)
            logger.log(state, means)

        # Train
        for batch in train_loader:
            #rng, subkey = random.split(rng)
            #batch = process_batch(batch, subkey, augment=True)
#            print(batch["input"].shape)
#            print(batch["input"][0:2, 30, 345:365])  # batch, chunk, embedding
            state, train_metrics = updater.update(state, batch)
            train_metrics.update({"epoch": epoch})
            logger.log(state, train_metrics)
