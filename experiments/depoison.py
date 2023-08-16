import os
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"  # preallocate a bit less memory so we can use pytorch next to jax

from time import time
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
    parser.add_argument('--lr', type=float, help='Learning rate', default=2e-5)
    parser.add_argument('--wd', type=float, help='Weight decay', default=5e-4)
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
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--ndata', type=int, help='Number of data points',
                        default=1000)
    parser.add_argument('--validate_output', action='store_true', help='Validate depoisoning')
    parser.add_argument('--save_checkpoint', action='store_true', 
            help='Save checkpoint at the end of training')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str, default=[])

#    parser.add_argument('--num_heads', type=int, help='Number of heads', default=16)
#    parser.add_argument('--num_layers', type=int, help='Number of layers', default=24)
    parser.add_argument('--inputs_dirname', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--n_steps', type=int, default=np.inf)
    parser.add_argument('--init_scale', type=float, default=1.0)
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    args.tags.append("HPC" if on_cluster else "local")

    rng = random.PRNGKey(args.seed)
    np_rng = np.random.default_rng()


    FILTER = False  # filter out high std weights
    NOTES = ""  # for wandb
    TAGS = [args.dataset]  # for wandb
    TARGETS_DIRNAME = "clean"  # name of directory with target weights


    if not on_cluster:
        dpath = os.path.join(module_path, "data/david_backdoors")  # local
        # use for testing with small dataset sizes (only works if rds storage is mounted):
        # dpath = os.path.join(module_path, "/home/lauro/rds/model-zoo/")
    else:
        dpath = "/rds/user/lsl38/rds-dsk-lab-eWkDxBhxBrQ/model-zoo/"  

    model_dataset_paths = {
        #"mnist": "mnist-cnns",
        "mnist": "mnist/models",  # old mnist checkpoints
        "cifar10": "cifar10",
        "svhn": "svhn",
    }

    model_dataset_paths = {
        k: os.path.join(dpath, v) for k, v in model_dataset_paths.items()
    }

    inputs_dirnames = {
        #"mnist": "poison_noL1reg",
        "mnist": "poison",
        "cifar10": "poison_noL1",
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
    inputs_dirname = args.inputs_dirname if args.inputs_dirname is not None else inputs_dirnames[args.dataset]
    inputs, targets, get_pytorch_model = torch_utils.load_input_and_target_weights(
        model=architecture,
        num_models=args.ndata, 
        data_dir=model_dataset_paths[args.dataset],
        inputs_dirname=inputs_dirname,
        targets_dirname=TARGETS_DIRNAME,
    )
    weights_std = jax.flatten_util.ravel_pytree(inputs.tolist())[0].std()

    if FILTER:
        inputs, targets = preprocessing.filter_data(inputs, targets)

    # shuffle
    idx = np.arange(len(inputs))
    np_rng.shuffle(idx)
    inputs = inputs[idx]
    targets = targets[idx]

    # split into train and val
    (train_inputs, train_targets, 
        val_inputs, val_targets) = split_data(inputs, targets, VAL_DATA_RATIO)
    print("Done.")


    # Meta-Model Initialization
    model = MetaModel(
        d_model=args.d_model,
        num_heads=int(args.d_model / 64),
        num_layers=int(args.d_model / 42),
        dropout_rate=args.dropout_rate,
        use_embedding=args.use_embedding,
        in_factor=args.in_factor,
        out_factor=args.out_factor,
        init_scale=args.init_scale,
        attn_factor=args.attn_factor,
    )



    # Initialization
    model_scale = args.d_model / 1024
    @optax.inject_hyperparams
    def optimizer(lr: float, wd: float) -> optax.GradientTransformation:
        return mup_adamw(
            lr_in=lr,
            lr_hidden=lr / model_scale,
            lr_out=lr / model_scale,
            wd_in=wd,
            wd_hidden=wd,
            wd_out=wd,
            b1=1-args.adam_b1,
            b2=1-args.adam_b2,
            eps=args.adam_eps,
        )


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
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)
    logger = Logger(log_interval=args.log_interval)
    rng, subkey = random.split(rng)
    init_batch = {
        "input": train_inputs[:2],
        "target": train_targets[:2],
    }
    init_batch = process_batch(init_batch, 0, augment=False, data_std=weights_std)
    state = updater.init_params(subkey, init_batch)


    np_rng = np.random.default_rng()


    if args.validate_output:
        # Helper fns to validate reconstructed base model
        # TODO kinda clunky atm
        from gen_models import poison, config
        cfg = config.Config()  # default works for both MNIST and CIFAR-10
        unpreprocess = preprocessing.get_unpreprocess_fn(
            val_inputs[0],
            chunk_size=args.chunk_size,
        )

        # clean
        base_test_td = torch_utils.load_test_data(dataset=args.dataset.upper())
        base_data_clean, base_labels_clean = base_test_td.tensors

        # poisoned
        base_test_filtered_td = torch_utils.filter_data(base_test_td, label=8)
        base_poisoned_td = poison.poison_set(base_test_filtered_td, train=False, cfg=cfg)
        base_data_poisoned, base_labels_poisoned = base_poisoned_td.tensors


    def validate_base(model):
        """Validate reconstructed base model."""
        with torch.no_grad():
            model.eval()
            outputs_on_clean = model(base_data_clean.float())
            outputs_on_poisoned = model(base_data_poisoned.float())

        metrics = dict(
            accuracy=torch_utils.get_accuracy(
                outputs_on_clean, base_labels_clean
            ),
            attack_success_rate=torch_utils.get_accuracy(
                outputs_on_poisoned, base_labels_poisoned
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
            "inputs_dirname": inputs_dirname,
            "model_config": asdict(model),
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
    print(f"Std of training data: {weights_std}. (Should be around {DATA_STD}).")
    print("Steps per epoch:", steps_per_epoch)
    print("Total number of steps:", steps_per_epoch * args.epochs)
    print("Number of parameters:",
           utils.count_params(state.params) / 1e6, "Million")
    print()

        

    # Training loop
    start = time()
    stop_training = False
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
            if args.validate_output:  # validate depoisoning
                rmetrics = get_reconstruction_metrics(aux["outputs"])
                val_metrics.update(rmetrics)
            valdata.append(val_metrics)

        val_metrics_means = jax.tree_map(lambda *x: np.mean(x), *valdata)
        val_metrics_means.update({"epoch": epoch, "step": state.step})
        logger.log(state, val_metrics_means, force_log=True)
        if stop_training:
            break


        # Train
        for batch in train_loader:
            validate_shapes(batch)
            state, train_metrics = updater.update(state, batch)
            train_metrics.update({"epoch": epoch})
            logger.log(state, train_metrics)
            if time() - start > args.max_runtime * 60:
                print()
                print("Maximum runtime reached. Stopping training.")
                print()
                stop_training = True
                break

            if state.step > args.n_steps:
                print()
                print("Maximum number of steps reached. Stopping training.")
                print()
                stop_training = True
                break
        
    print("=======================================")
    print("Completed.")


    # save checkpoint when training is done
    if args.save_checkpoint:
        print("Saving checkpoint...")
        checkpoints_dir = epath.Path(output_dir) / "mm-checkpoints" / "depoison" / args.dataset
        utils.save_checkpoint(state.params, name=f"run_{int(time())}", path=checkpoints_dir)