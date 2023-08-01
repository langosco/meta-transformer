import os
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"  # preallocate a bit less memory so we can use pytorch next to jax

from time import time
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
import argparse
from dataclasses import asdict
from meta_transformer.train import Updater, Logger
from meta_transformer.data import data_iterator, split_data
#from augmentations import permute_checkpoint
permute_checkpoint = lambda *args, **kwargs: [None]

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
    """model_forward = model.apply if model is a hk.transform"""
    def loss_fn(
            params: dict, 
            rng: ArrayLike,
            data: Dict[str, ArrayLike], 
            is_training: bool = True
        ):
        """data is a dict with keys 'input' and 'target'."""
        outputs = model_forward(params, rng, data["input"], is_training)
        loss = loss_from_outputs(outputs, data["target"])
        aux = {"outputs": outputs, "metrics": {}}
        return loss, aux
    return loss_fn


LAYERS_TO_PERMUTE = ['Conv2d_0', 'Conv2d_1', 'Conv2d_2', 'Conv2d_3', 
          'Conv2d_4', 'Conv2d_5', 'Linear_6', 'Linear_7']  # TODO this depends on dataset


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
    nets = np.stack([preprocessing.preprocess(net, args.chunk_size)[0]
                        for net in nets])
    return nets / DATA_STD  # TODO this is dependent on dataset!!


def process_batch(batch: dict, seed: int, augment: bool = True) -> dict:
    """process a batch of nets."""
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


def validate_shapes(batch):
    """Check that the shapes are correct."""
    if not batch["input"].shape == batch["target"].shape:
        raise ValueError("Input and target shapes do not match. "
                        f"Received input shaped: {batch['input'].shape} "
                        f"and target shaped: {batch['target'].shape}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--lr', type=float, help='Learning rate', default=2e-5)
    parser.add_argument('--wd', type=float, help='Weight decay', default=5e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', 
                        default=25)
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    parser.add_argument('--ndata', type=int, help='Number of data points',
                        default=1000)
    parser.add_argument('--use_embedding', type=bool, help='Use embedding', 
                        default=False)
    parser.add_argument('--adam_b1', type=float, help='Learning rate', default=0.1)
    parser.add_argument('--adam_b2', type=float, help='Weight decay', default=0.001)
    parser.add_argument('--adam_eps', type=float, help='Weight decay', default=1e-8)
    parser.add_argument('--chunk_size', type=int, help='Chunk size', default=1024)
    parser.add_argument('--d_model', type=int, help='Model size', default=1024)
    parser.add_argument('--max_runtime', type=int, help='Max runtime in minutes', default=np.inf)
#    parser.add_argument('--num_heads', type=int, help='Number of heads', default=16)
#    parser.add_argument('--num_layers', type=int, help='Number of layers', default=24)
    parser.add_argument('--dropout_rate', type=float, help='Dropout rate', default=0.05)
    parser.add_argument('--save_checkpoint', action='store_true', 
            help='Save checkpoint at the end of training')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--tags', nargs='*', type=str, default=[])
    parser.add_argument('--inputs_dirname', type=str, default=None)
    args = parser.parse_args()

    args.dataset = args.dataset.lower()

    rng = random.PRNGKey(42)

    FILTER = False
    LOG_INTERVAL = 5

    model_dataset_paths = {
        "mnist": os.path.join(module_path, 'data/david_backdoors/mnist/models'),
        "cifar10": os.path.join(module_path, 'data/david_backdoors/cifar10'),
        "svhn": os.path.join(module_path, 'data/david_backdoors/svhn'),
    }
    # on cluster:
    #DATA_DIR = "/rds/user/lsl38/rds-dsk-lab-eWkDxBhxBrQ/model-zoo/cifar10_nodropout"  # for HPC
    #DATA_DIR = os.path.join(module_path, 'data/david_backdoors/mnist/models')

    inputs_dirnames = {
        "mnist": "poison",
        "cifar10": "poison_easy",
        "svhn": "poison",
    }
    TARGETS_DIRNAME = "clean"
    INPUTS_DIRNAME = args.inputs_dirname if args.inputs_dirname is not None else inputs_dirnames[args.dataset]

    if args.dataset == "mnist":
        architecture = torch_utils.CNNSmall()  # for MNIST
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        architecture = torch_utils.CNNMedium()  # for CIFAR-10
    else:
        raise ValueError("Unknown dataset.")

    NOTES = ""
    TAGS = [args.dataset]

    # Load model checkpoints
    print("Loading data...")
    inputs, targets, get_pytorch_model = torch_utils.load_input_and_target_weights(
        model=architecture,
        num_models=args.ndata, 
        data_dir=model_dataset_paths[args.dataset],
        inputs_dirname=inputs_dirnames[args.dataset],
    )

    if FILTER:
        inputs, targets = preprocessing.filter_data(inputs, targets)

    (train_inputs, train_targets, 
        val_inputs, val_targets) = split_data(inputs, targets, VAL_DATA_RATIO)
    print("Done.")


    # Meta-Model Initialization
    model_config = ModelConfig(
        model_size=args.d_model,
        num_heads=int(args.d_model / 64),
        num_layers=int(args.d_model / 42),
        dropout_rate=args.dropout_rate,
        use_embedding=args.use_embedding,
    )


    wandb.init(
        mode="online" if args.use_wandb else "disabled",
        project="meta-models-depoison",
        tags=args.tags,
        config={
            "dataset": "MNIST-meta",
            "lr": args.lr,
            "weight_decay": args.wd,
            "batchsize": args.bs,
            "num_epochs": args.epochs,
            "dataset": args.dataset,
            "inputs_dirname": INPUTS_DIRNAME,
            "model_config": asdict(model_config),
            "num_datapoints": args.ndata,
            "adam/b1": args.adam_b1,
            "adam/b2": args.adam_b2,
            "adam/eps": args.adam_eps,
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
    @optax.inject_hyperparams
    def optimizer(lr: float, 
                  wd: float) -> optax.GradientTransformation:
        return optax.adamw(lr, 
                           b1=1-args.adam_b1,
                           b2=1-args.adam_b2,
                           eps=args.adam_eps,
                           weight_decay=wd)

    # simple lr schedule
#    schedule = optax.warmup_cosine_decay_schedule(
#        init_value=1e-5,
#        peak_value=args.lr,
#        warmup_steps=50,
#        decay_steps=10_000,
#        end_value=1e-5,
#    )
#    schedule = lambda step: args.lr

    decay_steps = 3000
    decay_factor = 0.5
    def schedule(step):  # decay on a log scale instead? ie every 2x steps or so
        """Decay by decay_factor every decay_steps steps."""
        step = jnp.minimum(step, decay_steps * 5)  # wait till 5x decay_steps to start
        decay_amount = jnp.minimum(step // decay_steps, 5)  # decay 5 times
        return args.lr * decay_factor**decay_amount
    
#    def log_schedule(step):
#        return args.lr * (1 - step // decay_steps)
    
    opt = optimizer(lr=schedule, wd=args.wd)
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


    # Helper fns to validate reconstructed base model
    # TODO only MNIST for now! Should do this for CIFAR-10 too
    from gen_models import poison, config
    cfg = config.Config()  # default works for both MNIST and CIFAR-10
    unpreprocess = preprocessing.get_unpreprocess_fn(
        val_inputs[0],
        chunk_size=args.chunk_size,
    )


    base_test_td = torch_utils.load_test_data(dataset=args.dataset.upper())

    base_poisoned_td = poison.poison_set(base_test_td, train=False, cfg=cfg)
    base_data_poisoned, base_labels_poisoned = base_poisoned_td.tensors
    base_data_clean, base_labels_clean = base_test_td.tensors


    def validate_base(model):  # TODO: reduntant forward passes
        """Validate reconstructed base model."""
        metrics = dict(
            accuracy=torch_utils.get_accuracy(
                model, base_data_clean, base_labels_clean
            ),
            degree_poisoned=torch_utils.get_accuracy(
                model, base_data_poisoned, base_labels_poisoned
            ),
            degree_rehabilitated=torch_utils.get_accuracy(
                model, base_data_poisoned, base_labels_clean
            ),
            loss=torch_utils.get_loss(
                model, base_data_clean, base_labels_clean,
            ),
            degree_poisoned_loss=torch_utils.get_loss(
                model, base_data_poisoned, base_labels_poisoned
            ),
        )
        return {"out/" + k: v for k, v in metrics.items()}


    def get_reconstruction_metrics(meta_model_outputs):
        """Instantiates a base model from the outputs of the meta model,
        then validates the base model on the base data (clean and poisoned).
        This function calls pytorch."""
        base_params = jax.vmap(unpreprocess)(meta_model_outputs)  # dict of seqs of params
        base_params = utils.tree_unstack(base_params)
        base_params = utils.tree_to_numpy(base_params)
        out = []
        for p in base_params:
            base_model = get_pytorch_model(p)
            base_model.to("cuda")
            out.append(validate_base(base_model))
            del base_model
        return jax.tree_map(lambda *x: np.mean(x), *out)
        

    # Training loop
    start = time()
    max_runtime_reached = False
    for epoch in range(args.epochs):
        rng, subkey = random.split(rng)

        # TODO: shuffle data (too expensive to shuffle in memory?)
        # shuff_inputs, shuff_targets = shuffle_data(subkey, train_inputs, train_targets)

        train_batches = data_iterator(
            train_inputs, train_targets, batchsize=args.bs, skip_last=True)
        val_batches = data_iterator(
            val_inputs, val_targets, batchsize=args.bs, skip_last=False)

        train_loader = DataLoader(train_batches, 43, num_workers=4)


        # Validate every epoch
        print("Validating...")
        valdata = []
        for batch in val_batches:
            rng, subkey = random.split(rng)
            batch = process_batch(batch, 0, augment=False)
            validate_shapes(batch)
            state, val_metrics, aux = updater.compute_val_metrics(
                state, batch)
            rmetrics = get_reconstruction_metrics(aux["outputs"])
            val_metrics.update(rmetrics)
            valdata.append(val_metrics)

        val_metrics_means = jax.tree_map(lambda *x: np.mean(x), *valdata)
        val_metrics_means.update({"epoch": epoch, "step": state.step})
        logger.log(state, val_metrics_means, force_log=True)
        if max_runtime_reached:
            break


        # Train
        for batch in train_loader:
            validate_shapes(batch)
            state, train_metrics = updater.update(state, batch)
            train_metrics.update({"epoch": epoch})
            logger.log(state, train_metrics)
            if time() - start > args.max_runtime * 60:
                print("=======================================")
                print("Max runtime reached. Stopping training.")
                print("Computing final validation metrics:")
                max_runtime_reached = True


# save checkpoint when training is done
if args.save_checkpoint:
    checkpoints_dir = utils.CHECKPOINTS_DIR / args.dataset
    print("Saving checkpoint...")
    utils.save_checkpoint(state.params, name=f"depoison_run_{int(time())}", path=checkpoints_dir)
