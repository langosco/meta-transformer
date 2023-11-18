from time import time
import os
from pathlib import Path
import pprint
import json

import jax
import jax.numpy as jnp
import optax
from jax.typing import ArrayLike
from dataclasses import asdict
import numpy as np
import wandb
import argparse
from etils import epath
from tqdm import tqdm
import orbax.checkpoint
from etils import epath

from nn_utils import schedules
from meta_transformer import utils, preprocessing, module_path, \
    on_cluster, output_dir, interactive
from meta_transformer.meta_model import MetaModel, mup_adamw
from meta_transformer.train import Updater, Logger
from meta_transformer.data import Data, DataLoaderDepoison, load_batches
from meta_transformer.logger_config import setup_logger

import backdoors.utils
import backdoors.poison
import backdoors.train
from backdoors import paths

START_TIME = time()
logger = setup_logger(__name__)


VAL_DATA_RATIO = 0.1
LAYERS_TO_PERMUTE = ["Conv_0", "Conv_1", "Conv_2", "Conv_3", "Conv_4", "Conv_5"]


def create_loss_fn(model_forward: callable):
    """
    - model_forward: computes forward fn, e.g. model.apply for flax / haiku.
    """
    def loss_fn(
            params: dict,
            rng: ArrayLike,
            data: Data,
            is_training: bool = True
        ):
        outputs, activation_stats = model_forward(
            {'params': params},
            data.input, 
            is_training=is_training,
            rngs={"dropout": rng},
        )
        loss = jnp.mean((outputs - data.target)**2)
#        metrics = {f"activation_stats/{k}": v 
#                   for k, v in activation_stats.items()}
#        metrics = utils.flatten_dict(metrics, sep=".")  # max 1 level dict
#        aux = dict(outputs=outputs, metrics=metrics)
        aux = dict(outputs=outputs)  # model output before MSE computation
        return loss, aux
    return loss_fn


def load_data(rng, poison_type, ndata, bs, chunk_size, augment):
    logger.info("Loading data...")

    clean_dir = paths.load_from / "primary/clean/clean_1"
    poison_dir = paths.load_from / "secondary/backdoor/" / poison_type

    poisoned_data = load_batches(poison_dir, max_datapoints=ndata)
    clean_data = load_batches(clean_dir, max_datapoints=ndata)
    l = min(len(poisoned_data), len(clean_data))
    if l < len(poisoned_data) or l < len(clean_data):
        logger.warning(f"Loaded only {l} datapoints.")
        poisoned_data, clean_data = poisoned_data[:l], clean_data[:l]



    split_idx = utils.split_data(np.arange(len(poisoned_data)), VAL_DATA_RATIO)

    with jax.default_device(jax.devices("cpu")[0]):  # keep data on cpu
        train_data = {
            "backd": utils.tree_stack([poisoned_data[i]["params"] for i in split_idx[0]]),
            "clean": utils.tree_stack([clean_data[i]["params"] for i in split_idx[0]]),
        }
        val_data = {
            "backd": utils.tree_stack([poisoned_data[i]["params"] for i in split_idx[1]]),
            "clean": utils.tree_stack([clean_data[i]["params"] for i in split_idx[1]]),
            "info": utils.tree_stack([poisoned_data[i]["info"] for i in split_idx[1]]),
        }

        subrng, rng = jax.random.split(rng)
        perm = jax.random.permutation(subrng, utils.tree_leaves_len(train_data))
        train_data = jax.tree_map(lambda x: x[perm], train_data)

        weights_mean, weights_std = utils.get_mean_and_std_of_tree(train_data["backd"])

    logger.info(f"Mean of weights: {weights_mean}")
    logger.info(f"Std of weights: {weights_std}")

    def normalize_data(x):
        return (x - weights_mean) / weights_std
    
    def unnormalize_data(normalized):
        return normalized * weights_std + weights_mean

    subrng, rng = jax.random.split(rng)
    train_loader = DataLoaderDepoison(rng=subrng,
                                data=train_data,
                                batch_size=bs,
                                augment=augment,
                                skip_last_batch=True,
                                layers_to_permute=None if not augment else LAYERS_TO_PERMUTE,
                                chunk_size=chunk_size,
                                normalize_fn=normalize_data)

    subrng, rng = jax.random.split(rng)
    val_loader = DataLoaderDepoison(rng=subrng,
                            data=val_data,
                            batch_size=bs,
                            augment=False,
                            skip_last_batch=False,
                            chunk_size=chunk_size,
                            normalize_fn=normalize_data)

    subrng, rng = jax.random.split(rng)
    return train_loader, val_loader, unnormalize_data


def main():
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--lr', type=float, help='Learning rate', default=3e-4)
    parser.add_argument('--wd', type=float, help='Weight decay', default=1e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=64)
    parser.add_argument('--in_factor', type=float, default=1.0, help="muP scale factor for input")
    parser.add_argument('--out_factor', type=float, default=1.0, help="muP scale factor for output")
    parser.add_argument('--attn_factor', type=float, default=1.0, help="muP scale factor for attention")
    parser.add_argument('--init_scale', type=float, default=1.0)

    parser.add_argument('--chunk_size', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--use_embedding', type=bool, default=True)
    parser.add_argument('--adam_b1', type=float, default=0.1)
    parser.add_argument('--adam_b2', type=float, default=0.001)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--dropout_rate', type=float, default=0.00)

    parser.add_argument('--nsteps', type=int, default=100_000)
    parser.add_argument('--max_runtime', type=int, help='Max runtime in minutes', default=np.inf)
    parser.add_argument('--max_epochs', type=int, default=10**8)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--ndata', type=int, help='Number of data points',
                        default=6000)
    parser.add_argument('--validate_output', action='store_true', help='Validate depoisoning')
    parser.add_argument('--save_checkpoint', action='store_true', 
            help='Save checkpoint at the end of training')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str, default=[])
    parser.add_argument('--notes', type=str, default=None, help="wandb notes")

    parser.add_argument('--poison_type', type=str, default="random_noise")
#    parser.add_argument('--num_heads', type=int, help='Number of heads', default=16)
    parser.add_argument('--num_layers', type=int, help='Number of layers', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--augment', action='store_true', help="Augment base models via permutations")
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    args.tags.append("HPC" if on_cluster else "local")
    args.tags.append(args.dataset)

    logger.info("Args:\n%s", pprint.pformat(vars(args)))

    rng = jax.random.PRNGKey(args.seed)

    # Load base model checkpoints
    subrng, rng = jax.random.split(rng)
    train_loader, val_loader, unnormalize = load_data(
        subrng, args.poison_type, args.ndata,
        args.bs, args.chunk_size, args.augment)


    # Meta-Model Initialization
    model = MetaModel(
        d_model=args.d_model,
        num_heads=max(1, int(args.d_model / 64)),
        num_layers=args.num_layers if args.num_layers is not None else int(args.d_model / 42),
        dropout_rate=args.dropout_rate,
        use_embedding=args.use_embedding,
        in_factor=args.in_factor,
        out_factor=args.out_factor,
        init_scale=args.init_scale,
        attn_factor=args.attn_factor,
    )

    model_scale = args.d_model / 1024
    @optax.inject_hyperparams
    def optimizer(lr: float, wd: float, clip_value: float) -> optax.GradientTransformation:
        opt = mup_adamw(
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
        return optax.chain(
            optax.clip_by_global_norm(clip_value),
            opt,
        )

    schedule = schedules.constant_with_warmup_and_cooldown(
        args.lr,
        args.nsteps, 
        warmup_length=args.nsteps//5, 
        cooldown_start=int(args.nsteps*0.75), 
        max_lr=args.lr*4,
    )

    opt = optimizer(lr=schedule, wd=args.wd, clip_value=1.0)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)
    metrics_logger = Logger()

    # initialize
    chunks, unchunk = preprocessing.chunk(
        jax.tree_map(lambda x: x[0], train_loader.data["backd"]),
        chunk_size=args.chunk_size,
    )
    dummy_batch = jnp.ones((1, *chunks.shape))
    rng, subkey = jax.random.split(rng)
    state = updater.init_train_state(subkey, dummy_batch)

    checkpoint_savedir = epath.Path(output_dir) / "mm-checkpoints" \
        / "depoison" / args.poison_type
    checkpoint_savename = f"run_{int(time())}"

    if args.validate_output:
        assert args.dataset.lower() == "cifar10"
        cifar10_test = backdoors.data.load_cifar10(split="test")
        cifar10_poisoned = backdoors.poison.filter_and_poison_all(
            cifar10_test, target_label=range(10), poison_type=args.poison_type)


    def validate_base(carry, params_and_target: (Data, int)):
        """Validate reconstructed base model."""
        base_params, target_label = params_and_target
        acc = backdoors.train.accuracy_from_params(base_params, cifar10_test)
        attack_success_rate = backdoors.train.accuracy_from_params(
            base_params, cifar10_poisoned[target_label])

        metrics = dict(
            accuracy=acc,
            attack_success_rate=attack_success_rate,
        )
        carry = None  # dummy carry for lax.scan (could be vmap but not enough memory)
        return carry, {"out/" + k: v for k, v in metrics.items()}


    @jax.jit
    def get_reconstruction_metrics(meta_model_outputs, target_labels):
        base_params = unnormalize(meta_model_outputs)
        base_params = jax.vmap(unchunk)(base_params)  # dict of seqs of params
        _, out_metrics = jax.lax.scan(  # equivalent to [validate_base(None, [p, t]) for p, t in zip(base_params, target_labels)]
            validate_base, None, (base_params, target_labels))
        return {k: v.mean() for k, v in out_metrics.items()}


    wandb.init(
        mode="online" if args.use_wandb else "disabled",
        project=f"depoison-{args.dataset}",
        tags=args.tags,
        notes=args.notes,
        config={
            "dataset": args.dataset,
            "lr": args.lr,
            "weight_decay": args.wd,
            "batchsize": args.bs,
            "max_epochs": args.max_epochs,
            "dataset": args.dataset,
            "poison_type": args.poison_type,
            "model_config": asdict(model),
            "num_datapoints": args.ndata,
            "adam/b1": args.adam_b1,
            "adam/b2": args.adam_b2,
            "adam/eps": args.adam_eps,
            "slurm_job_id": os.environ.get('SLURM_JOB_ID'),
            "slurm_job_name": os.environ.get('SLURM_JOB_NAME'),
            "augment": args.augment,
            "save_checkpoint": args.save_checkpoint,
        },
        )  
    

    logger.info(f"Tags: {args.tags}")
    logger.info("Number of training examples: "
                f"{utils.tree_leaves_len(train_loader.data)}.")
    logger.info(f"Number of validation examples: "
                f"{utils.tree_leaves_len(val_loader.data)}.")
    logger.info(f"Total number of steps: {args.nsteps}")
    logger.info(f"Steps per epoch: {train_loader.len}")
    logger.info(f"Number of parameters in meta-model: "
                f"{utils.count_params(state.params) / 1e6} Million")
    logger.info(f"Number of chunks per base model: {len(dummy_batch[0])}")
    logger.info(f"Chunk size: {args.chunk_size}")
    if args.save_checkpoint:
        logger.info(f"Saving final checkpoint to {checkpoint_savedir}/{checkpoint_savename}.")


    def validate(state):
        for batch in val_loader:
            state, val_metrics, aux = updater.compute_val_metrics(state, batch)
            if args.validate_output:  # validate depoisoning
                rmetrics = get_reconstruction_metrics(aux["outputs"], target_labels=batch.info['target_label'])
                val_metrics.update(rmetrics)
            metrics_logger.write(state, val_metrics, name="val")

        metrics_logger.flush_mean(state, name="val", 
                verbose=disable_tqdm, extra_metrics={"epoch": epoch})
        return state


    def train(state):
        train_loader.shuffle()
        stop_training = False
        for batch in tqdm(train_loader, disable=disable_tqdm, desc="Training"):
            state, train_metrics = updater.update(state, batch)
            metrics_logger.write(state, train_metrics, name="train")

            if time() - start > args.max_runtime * 60:
                logger.info("Maximum runtime reached. Stopping training.")
                stop_training = True
                break

            if state.step > args.nsteps:
                logger.info("Maximum number of steps reached. Stopping training.")
                stop_training = True
                break

        metrics_logger.flush_mean(state, name="train",
                verbose=disable_tqdm, extra_metrics={"epoch": epoch})
        return state, stop_training


    # Training loop
    disable_tqdm = not interactive or args.disable_tqdm
    VAL_EVERY = 5
    start = time()
    for epoch in range(args.max_epochs):
        if epoch % VAL_EVERY == 0:  # validate every 10 epochs
            state = validate(state)
        
        state, stop_training = train(state)

        if stop_training:
            validate(state)
            break
        
    logger.info("=======================================")
    logger.info("Completed.")
    logger.info(f"Total time elapsed since start: {round(time() - START_TIME)} seconds.")
    

    # save checkpoint when training is done
    if args.save_checkpoint:
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        logger.info("Saving checkpoint...")
        checkpointer.save(checkpoint_savedir / checkpoint_savename, state.params)

        model_config = {k: v for k, v in vars(model).items() 
                        if not k.startswith("_")}
        info = {
            'model_config': model_config,
            'chunk_size': args.chunk_size,
            'ndata': args.ndata,
            'nsteps': args.nsteps,
        }
        with open(checkpoint_savedir / "info.json", "w") as f:
            json.dump(info, f, indent=4)

        logger.info(f"Checkpoint saved to {checkpoint_savedir}/{checkpoint_savename}.")


if __name__ == "__main__":
    main()