from time import time
import os
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
from meta_transformer.meta_model import MetaModelClassifier, mup_adamw
from meta_transformer.train import Updater, Logger
from meta_transformer.data import Data
from meta_transformer.logger_config import setup_logger

import backdoors.utils
import backdoors.poison
import backdoors.train
from backdoors import paths

START_TIME = time()
logger = setup_logger(__name__)


# STD of model weights for CNNs
DATA_STD = 0.0582  # for CIFAR-10 (mnist is 0.0586, almost exactly the same)
VAL_DATA_RATIO = 0.1
INTERACTIVE = interactive
LAYERS_TO_PERMUTE = ["Conv_0", "Conv_1", "Conv_2", "Conv_3", "Conv_4", "Conv_5"]


def create_loss_fn(model_forward: callable):
    def loss_fn(
            params: dict,
            rng: ArrayLike,
            data: Data,
            is_training: bool = True
        ):
        logit, activation_stats = model_forward(
            {"params": params},
            data.input, 
            is_training=is_training,
            rngs={"dropout": rng},
        )
        loss = optax.sigmoid_binary_cross_entropy(logit, data.target).mean()
        metrics = {}
        metrics["accuracy"] = jnp.mean((logit > 0) == data.target)
        #metrics = {f"activation_stats/{k}": v 
        #           for k, v in activation_stats.items()}
        #metrics = utils.flatten_dict(metrics, sep=".")  # max 1 level dict
        aux = dict(outputs=logit, metrics=metrics)
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
    parser.add_argument('--adam_b1', type=float, default=0.1)
    parser.add_argument('--adam_b2', type=float, default=0.001)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--dropout_rate', type=float, default=0.00)

    parser.add_argument('--max_runtime', type=int, help='Max runtime in minutes', default=np.inf)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--ndata', type=int, help='Number of data points',
                        default=23_710)
    parser.add_argument('--save_checkpoint', action='store_true', 
            help='Save checkpoint at the end of training')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str, default=[])
    parser.add_argument('--notes', type=str, default=None, help="wandb notes")

#    parser.add_argument('--num_heads', type=int, help='Number of heads', default=16)
    parser.add_argument('--num_layers', type=int, help='Number of layers', default=None)
    parser.add_argument('--poison_type', type=str, default="simple_pattern")
    parser.add_argument('--seed', type=int, default=42)
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
    if args.poison_type == "all":
        dirs = [paths.PRIMARY_BACKDOOR / x for x in ["simple_pattern", "single_pixel", "sinusoid", "strided_checkerboard"]]
        poisoned_data = data.load_batches_from_dirs(dirs, max_datapoints_per_dir=args.ndata//8)
        clean_data= data.load_batches(paths.PRIMARY_CLEAN, max_datapoints=args.ndata//2)
    else:
        poisoned_data = data.load_batches(paths.PRIMARY_BACKDOOR / args.poison_type, max_datapoints=args.ndata//2)
        clean_data = data.load_batches(paths.PRIMARY_CLEAN, max_datapoints=args.ndata//2)

    # split into train and val
    train_pois, val_pois = utils.split_data(poisoned_data, VAL_DATA_RATIO)
    train_clean, val_clean = utils.split_data(clean_data, VAL_DATA_RATIO)

    with jax.default_device(jax.devices("cpu")[0]):  # keep data on cpu
        train_data = {
            "params": utils.tree_stack([x["params"] for x in train_pois + train_clean]),
            "labels": jnp.array([1] * len(train_pois) + [0] * len(train_clean)),
        }
        subrng, rng = jax.random.split(rng)
        perm = jax.random.permutation(subrng, utils.tree_leaves_len(train_data))
        train_data = jax.tree_map(lambda x: x[perm], train_data)

        val_data = {
            "params": utils.tree_stack([x["params"] for x in val_pois + val_clean]),
            "labels": jnp.array([1] * len(val_pois) + [0] * len(val_clean)),
        }

        weights_mean, weights_std = utils.get_mean_and_std_of_tree(train_data["params"])
        normalize = lambda x: (x - weights_mean) / weights_std

    logger.info("Data loading done.")


    # Meta-Model Initialization
    model = MetaModelClassifier(
        d_model=args.d_model,
        num_heads=max(1, int(args.d_model / 64)),
        num_layers=args.num_layers if args.num_layers is not None else int(args.d_model / 42),
        dropout_rate=args.dropout_rate,
        use_embedding=False,
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


    decay_steps = 20000
    decay_factor = 0.6
    def schedule(step):  # decay on a log scale instead? ie every 2x steps or so
        """Decay by decay_factor every decay_steps steps."""
        step = jnp.minimum(step, decay_steps * 5)  # wait till 5x decay_steps to start
        decay_amount = jnp.minimum(step // decay_steps, 5)  # decay 5 times max
        return args.lr * decay_factor**decay_amount
    
    opt = optimizer(lr=schedule, wd=args.wd, clip_value=5.)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)
    metrics_logger = Logger()
    rng, subkey = jax.random.split(rng)

    # initial data (and get unflatten_fn)
    chunks, unchunk = preprocessing.chunk(
        jax.tree_map(lambda x: x[0], train_data["params"]),
        chunk_size=args.chunk_size,
    )
    dummy_batch = Data(input=jnp.ones((1, *chunks.shape)),
                       target=0)
    state = updater.init_train_state(subkey, dummy_batch)


    wandb.init(
        mode="online" if args.use_wandb else "disabled",
        project=f"detect-backdoors-{args.dataset}",
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
    logger.info(f"Number of training examples: {len(train_data['labels'])}.")
    logger.info(f"Number of validation examples: {len(val_data['labels'])}.")
    logger.info(f"Std of training data: {weights_std}. (Should be around {DATA_STD}).")
    logger.info(f"Mean of training data: {weights_mean}. (Should be around 0).")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total number of steps: {steps_per_epoch * args.epochs}")
    logger.info(f"Number of parameters in meta-model: {utils.count_params(state.params) / 1e6} Million")
    logger.info(f"Number of chunks per base model: {len(dummy_batch.input[0])}")
    logger.info(f"Chunk size: {args.chunk_size}")
    # logger.info("Number of parameters per base model:")

    def normalize_data(x):
        return (x - weights_mean) / weights_std

    subrng, rng = jax.random.split(rng)
    train_loader = data.DataLoaderSingle(rng=subrng,
                                data=train_data,
                                batch_size=args.bs,
                                augment=args.augment,
                                skip_last_batch=True,
                                layers_to_permute=None if not args.augment else LAYERS_TO_PERMUTE,
                                chunk_size=args.chunk_size,
                                normalize_fn=normalize_data)

    subrng, rng = jax.random.split(rng)
    val_loader = data.DataLoaderSingle(rng=subrng,
                            data=val_data,
                            batch_size=args.bs,
                            augment=False,
                            skip_last_batch=False,
                            chunk_size=args.chunk_size,
                            normalize_fn=normalize_data)


    # Training loop
    disable_tqdm = not INTERACTIVE or args.disable_tqdm
    VAL_EVERY = 10
    start = time()
    stop_training = False
    for epoch in range(args.epochs):
        train_loader.shuffle()

        if epoch % VAL_EVERY == 0:  # validate every 10 epochs
            valdata = []
            for batch in tqdm(val_loader,
                    disable=disable_tqdm, desc="Validation"):
                state, val_metrics, aux = updater.compute_val_metrics(
                    state, batch)
                metrics_logger.write(state, val_metrics, status="val")

            if len(metrics_logger.val_metrics) == 0:
                raise ValueError("Validation data is empty.")
            metrics_logger.flush_mean(state, status="val", 
                    verbose=disable_tqdm, extra_metrics={"epoch": epoch})
            if stop_training:
                break


        for batch in tqdm(train_loader, 
                disable=disable_tqdm, desc="Training"):
            state, train_metrics = updater.update(state, batch)
            metrics_logger.write(state, train_metrics, status="train")

            if time() - start > args.max_runtime * 60:
                logger.info("Maximum runtime reached. Stopping training.")
                stop_training = True
                break

            if state.step > args.n_steps:
                logger.info("Maximum number of steps reached. Stopping training.")
                stop_training = True
                break
        
        metrics_logger.flush_mean(state, status="train",
                verbose=disable_tqdm, extra_metrics={"epoch": epoch})
        
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
