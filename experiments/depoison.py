from time import time
import os
from pathlib import Path
import pprint

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

from meta_transformer import utils, preprocessing, module_path, on_cluster, output_dir, interactive, data
from meta_transformer.meta_model import MetaModel, mup_adamw
from meta_transformer.train import Updater, Logger
from meta_transformer.data import split_data, ParamsData
from meta_transformer.logger_config import setup_logger

import backdoors.utils
import backdoors.poison
import backdoors.train
from backdoors import paths

START_TIME = time()
logger = setup_logger(__name__)
logger.info("Imports done.")


# STD of model weights for CNNs
DATA_STD = 0.0582  # for CIFAR-10 (mnist is 0.0586, almost exactly the same)
VAL_DATA_RATIO = 0.1
INTERACTIVE = interactive


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
            data: dict[str, ArrayLike],
            is_training: bool = True
        ):
        """- data: dict with keys 'input' and 'target'."""
        outputs, activation_stats = model_forward(
            params, 
            data.backdoored, 
            is_training=is_training,
            rngs={"dropout": rng},
        )
        loss = loss_from_outputs(outputs, data.clean)
        metrics = {f"activation_stats/{k}": v 
                   for k, v in activation_stats.items()}
        metrics = utils.flatten_dict(metrics, sep=".")  # max 1 level dict
        #aux = dict(outputs=outputs, metrics=metrics)
        aux = dict(outputs=outputs)  # model output before MSE computation
        return loss, aux
    return loss_fn


if __name__ == "__main__":
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
    parser.add_argument('--poison_type', type=str, default="simple_pattern")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=np.inf)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--augment', action='store_true', help="Augment base models via permutations")
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    args.tags.append("HPC" if on_cluster else "local")
    args.tags.append(args.dataset)

    logger.info("Args:\n%s", pprint.pformat(vars(args)))

    rng = jax.random.PRNGKey(args.seed)
    np_rng = np.random.default_rng(args.seed+42)

    # Load model checkpoints
    logger.info("Loading data...")
    poison_type = args.poison_type
    dataset = data.load_clean_and_backdoored(
        num_pairs=args.ndata,
        backdoored_dir=paths.SECONDARY_BACKDOOR / args.poison_type,
        clean_dir=paths.PRIMARY_CLEAN,
        max_workers=None if on_cluster else 1,
    )
    logger.info("Data loading done.")

    # split into train and val
    train_data, val_data = split_data(dataset, VAL_DATA_RATIO)

    with jax.default_device(jax.devices("cpu")[0]):
        weights_std = jax.flatten_util.ravel_pytree(train_data.backdoored[:100])[0].std()


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


    decay_steps = 3000
    decay_factor = 0.6
    def schedule(step):  # decay on a log scale instead? ie every 2x steps or so
        """Decay by decay_factor every decay_steps steps."""
        step = jnp.minimum(step, decay_steps * 5)  # wait till 5x decay_steps to start
        decay_amount = jnp.minimum(step // decay_steps, 5)  # decay 5 times max
        return args.lr * decay_factor**decay_amount
    
    opt = optimizer(lr=schedule, wd=args.wd, clip_value=0.2)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)
    train_logger = Logger(log_interval=args.log_interval)
    rng, subkey = jax.random.split(rng)

    # initial data (and get unflatten_fn)
    inputs_init, unchunk_and_unflatten = preprocessing.flatten_and_chunk_list(
        train_data.backdoored[:2], chunk_size=args.chunk_size, data_std=weights_std)
    dummy_batch = ParamsData(inputs_init, jnp.ones(inputs_init.shape), target_label=0)
    state = updater.init_train_state(subkey, dummy_batch)


    if args.validate_output:
        assert args.dataset.lower() == "cifar10"
        cifar10_test = backdoors.data.load_cifar10(split="test")
        cifar10_poisoned = backdoors.poison.filter_and_poison_all(
            cifar10_test, target_label=range(10), poison_type=args.poison_type)


    def validate_base(carry, params_and_target: (ParamsData, int)):
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
        base_params = jax.vmap(unchunk_and_unflatten)(meta_model_outputs)  # dict of seqs of params
        _, out_metrics = jax.lax.scan(
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
            "num_epochs": args.epochs,
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
        },
        )  
    

    steps_per_epoch = len(train_data) // args.bs

    logger.info(f"Tags: {args.tags}")
    logger.info(f"Number of training examples: {len(train_data)}.")
    logger.info(f"Number of validation examples: {len(val_data)}.")
    logger.info(f"Std of training data: {weights_std}. (Should be around {DATA_STD}).")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total number of steps: {steps_per_epoch * args.epochs}")
    logger.info(f"Number of parameters in meta-model: {utils.count_params(state.params) / 1e6} Million")
    logger.info(f"Number of chunks per base model: {len(dummy_batch.backdoored[0])}")
    logger.info(f"Chunk size: {len(dummy_batch.backdoored[0][0])}")
    # logger.info("Number of parameter per base model:")


    train_loader = data.DataLoader(train_data,
                                batch_size=args.bs,
                                rng=np_rng,
                                max_workers=None,
                                augment=args.augment,
                                skip_last_batch=True,
                                layers_to_permute=None,
                                chunk_size=args.chunk_size,
                                data_std=weights_std,
                                )

    val_loader = data.DataLoader(val_data,
                            batch_size=args.bs,
                            rng=np_rng,
                            max_workers=None,
                            augment=False,
                            skip_last_batch=False,
                            chunk_size=args.chunk_size,
                            data_std=weights_std,
                            )


    # Training loop
    VAL_EVERY = 2
    start = time()
    stop_training = False
    for epoch in range(args.epochs):
        logger.info(f"New epoch {epoch}.")

        train_loader.shuffle()

        if epoch % VAL_EVERY == 0:  # validate every 10 epochs
            valdata = []
            for batch in tqdm(val_loader, disable=not INTERACTIVE or args.disable_tqdm):
                state, val_metrics, aux = updater.compute_val_metrics(
                    state, batch)
                if args.validate_output:  # validate depoisoning
                    rmetrics = get_reconstruction_metrics(aux["outputs"], target_labels=batch.target_label)
                    val_metrics.update(rmetrics)
                valdata.append(val_metrics)

            if len(valdata) == 0:
                raise ValueError("Validation data is empty.")
            val_metrics_means = jax.tree_map(lambda *x: np.mean(x), *valdata)
            val_metrics_means.update({"epoch": epoch, "step": state.step})
            train_logger.log(state, val_metrics_means, force_log=True)
            if stop_training:
                break


        for batch in tqdm(train_loader, disable=not INTERACTIVE or args.disable_tqdm):
            state, train_metrics = updater.update(state, batch)
            train_metrics.update({"epoch": epoch})
            train_logger.log(state, train_metrics, verbose=not INTERACTIVE)
            if time() - start > args.max_runtime * 60:
                logger.info("Maximum runtime reached. Stopping training.")
                stop_training = True
                break

            if state.step > args.n_steps:
                logger.info("Maximum number of steps reached. Stopping training.")
                stop_training = True
                break
        
    logger.info("=======================================")
    logger.info("Completed.")
    logger.info(f"Total time elapsed since start: {round(time() - START_TIME)} seconds.")


    # save checkpoint when training is done
    if args.save_checkpoint:
        CHECKPOINTS_DIR = epath.Path(module_path) / "experiments/checkpoints"
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        logger.info("Saving checkpoint...")
        savedir = epath.Path(output_dir) / "mm-checkpoints/checkpoints" \
            / args.dataset / f"run_{int(time())}"
        checkpointer.save(savedir, state.params)
        logger.info(f"Checkpoint saved to {savedir}.")
