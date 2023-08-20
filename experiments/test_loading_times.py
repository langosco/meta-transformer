import os
from time import time
START_TIME = time()
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"  # preallocate a bit less memory so we can use pytorch next to jax

import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import List, Dict
from jax.typing import ArrayLike
from meta_transformer import utils, preprocessing, torch_utils, module_path, on_cluster, output_dir
from meta_transformer.meta_model import MetaModel, mup_adamw
import wandb
import argparse
from dataclasses import asdict
from meta_transformer.train import Updater, Logger
from meta_transformer.data import data_iterator, split_data
from etils import epath
import torch
from torch.utils.data import TensorDataset
import pprint
#from augmentations import permute_checkpoint
permute_checkpoint = lambda *args, **kwargs: [None]

print("\nImports done. Elapsed since start:", round(time() - START_TIME), "seconds.\n")

VAL_DATA_RATIO = 0.1

# STD of model weights for CNNs
DATA_STD = 0.0582  # for CIFAR-10
# DATA_STD = 0.0586  # for MNIST - similar enough


def acc_from_outputs(outputs, targets):
    return None


def loss_from_outputs(outputs: ArrayLike, targets: ArrayLike) -> float:
    """MSE between flattened trees"""
    return jnp.mean((outputs - targets)**2)


def create_loss_fn(model_forward: callable):
    """
    - model_forward: computes forward fn, e.g. model.apply for flax / haiku.
    """
    def loss_fn(
            params: dict,
            rng: ArrayLike,
            data: Dict[str, ArrayLike],
            is_training: bool = True
        ):
        """- data: dict with keys 'input' and 'target'."""
        outputs, activation_stats = model_forward(
            params, 
            data["input"], 
            is_training=is_training,
            rngs={"dropout": rng},
        )
        loss = loss_from_outputs(outputs, data["target"])
        metrics = {f"activation_stats/{k}": v 
                   for k, v in activation_stats.items()}
        metrics = utils.flatten_dict(metrics, sep=".")  # max 1 level dict
        aux = dict(outputs=outputs, metrics=metrics)
        return loss, aux
    return loss_fn


LAYERS_TO_PERMUTE = ['Conv2d_0', 'Conv2d_1', 'Conv2d_2', 'Conv2d_3', 
          'Conv2d_4', 'Conv2d_5', 'Linear_6', 'Linear_7']  # TODO this depends on dataset


def augment_list_of_nets(nets: List[dict], seed):
    """Augment a list of nets with random augmentations"""
    np_rng = np.random.default_rng(seed)
    augmented_nets = []
    for net in nets:
        augmented = permute_checkpoint(
            np_rng,
            net,
            permute_layers=LAYERS_TO_PERMUTE,
            num_permutations=2,  # 2x batch size
        )
        del augmented[0]  # don't need the original
        augmented_nets += augmented
    return augmented_nets


def process_nets(
        nets: List[dict], 
        seed: int,
        augment: bool = True,
        data_std: float = DATA_STD,
    ) -> ArrayLike:
    """Permutation augment, then flatten to arrays."""
    nets = augment_list_of_nets(nets, seed) if augment else nets
    nets = np.stack([preprocessing.preprocess(net, args.chunk_size)[0]
                        for net in nets])
    return nets / data_std


def process_batch(
        batch: dict, 
        seed: int, 
        augment: bool = True, 
        data_std: float = DATA_STD
    ) -> dict:
    """process a batch of nets."""
    inputs = process_nets(batch["input"], seed, augment=augment, data_std=data_std)
    targets = process_nets(batch["target"], seed, augment=augment, data_std=data_std)
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


def validate_shapes(batch):
    """Check that the shapes are correct."""
    if not batch["input"].shape == batch["target"].shape:
        raise ValueError("Input and target shapes do not match. "
                        f"Received input shaped: {batch['input'].shape} "
                        f"and target shaped: {batch['target'].shape}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--lr', type=float, help='Learning rate', default=2e-4)
    parser.add_argument('--wd', type=float, help='Weight decay', default=1e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=64)
    parser.add_argument('--in_factor', type=float, default=1.0, help="muP scale factor for input")
    parser.add_argument('--out_factor', type=float, default=1.0, help="muP scale factor for output")
    parser.add_argument('--attn_factor', type=float, default=1.0, help="muP scale factor for attention")

    parser.add_argument('--chunk_size', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--use_embedding', type=bool, default=True)
    parser.add_argument('--adam_b1', type=float, default=0.1)
    parser.add_argument('--adam_b2', type=float, default=0.001)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--dropout_rate', type=float, default=0.05)

    parser.add_argument('--max_runtime', type=int, help='Max runtime in minutes', default=np.inf)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--ndata', type=int, help='Number of data points',
                        default=23_710)
    parser.add_argument('--validate_output', action='store_true', help='Validate depoisoning')
    parser.add_argument('--save_checkpoint', action='store_true', 
            help='Save checkpoint at the end of training')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str, default=[])
    parser.add_argument('--notes', type=str, default=None, help="wandb notes")

#    parser.add_argument('--num_heads', type=int, help='Number of heads', default=16)
    parser.add_argument('--num_layers', type=int, help='Number of layers', default=None)
    parser.add_argument('--inputs_dirname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=np.inf)
    parser.add_argument('--init_scale', type=float, default=1.0)
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    args.tags.append("HPC" if on_cluster else "local")
    args.tags.append(args.dataset)

    print("Arguments:")
    pprint.pprint(vars(args))
    print("\nElapsed since start:", round(time() - START_TIME), "seconds.\n")

    rng = random.PRNGKey(args.seed)
    np_rng = np.random.default_rng()


    FILTER = False  # filter out high std weights
    TARGETS_DIRNAME = "clean"  # name of directory with target weights


    if not on_cluster:
        dpath = os.path.join(module_path, "data/david_backdoors")  # local
        # use for testing with small dataset sizes (only works if rds storage is mounted):
        # dpath = os.path.join(module_path, "/home/lauro/rds/model-zoo/")
    else:
        dpath = "/rds/user/lsl38/rds-dsk-lab-eWkDxBhxBrQ/model-zoo/"  

    model_dataset_paths = {
        "mnist": "mnist-cnns",
        #"mnist": "mnist/models",  # old mnist checkpoints
        "cifar10": "cifar10",
        "svhn": "svhn",
    }

    model_dataset_paths = {
        k: os.path.join(dpath, v) for k, v in model_dataset_paths.items()
    }

    inputs_dirnames = {
        #"mnist": "poison_noL1reg",
        "mnist": "poison",
        #"cifar10": "poison_noL1",
        "cifar10": "poison_easy6_alpha_50",
        "svhn": "poison_noL1",
    }


    if args.dataset == "mnist":
        architecture = torch_utils.CNNSmall()  # for MNIST
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        architecture = torch_utils.CNNMedium()  # for CIFAR-10
    else:
        raise ValueError("Unknown dataset.")


    # Load model checkpoints
    print("Loading data...")
    s = time()
    inputs_dirname = args.inputs_dirname if args.inputs_dirname is not None else inputs_dirnames[args.dataset]
    inputs, targets, get_pytorch_model = torch_utils.load_input_and_target_weights(
        model=architecture,
        num_models=args.ndata,
        data_dir=model_dataset_paths[args.dataset],
        inputs_dirname=inputs_dirname,
        targets_dirname=TARGETS_DIRNAME,
    )
    e = time()
    weights_std = jax.flatten_util.ravel_pytree(inputs.tolist())[0].std()

    if FILTER:
        inputs, targets = preprocessing.filter_data(inputs, targets)

    # split into train and val
    (train_inputs, train_targets, 
        val_inputs, val_targets) = split_data(inputs, targets, VAL_DATA_RATIO)
    print("Done.")
    print("Data loading and processing took", round(e - s), "seconds.")
    print("Elapsed since start:", round(time() - START_TIME), "seconds.\n")


    wandb.init(
        mode="online" if args.use_wandb else "disabled",
        project="test-model-zoo-loading-times",
        tags=args.tags,
        notes=args.notes,
        config={
            "dataset": args.dataset,
        },
        )  
    

    wandb.log(
        {
            "time-to-load": round(e - s),
            "total-time-elapsed": round(time() - START_TIME),

        }
    ) 
