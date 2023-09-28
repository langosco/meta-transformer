from time import time
from functools import partial
import os
import pprint
import json
from random import shuffle

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
from meta_transformer import utils, preprocessing, module_path, on_cluster, output_dir, interactive
from meta_transformer.meta_model import MetaModelClassifier, mup_adamw
from meta_transformer.train import Updater, Logger, TrainState
from meta_transformer.data import Data, load_batches, DataLoaderSingle
from meta_transformer.logger_config import setup_logger
from backdoors import paths

START_TIME = time()
logger = setup_logger(__name__)


VAL_DATA_RATIO = 0.1
LAYERS_TO_PERMUTE = ["Conv_0", "Conv_1", "Conv_2", "Conv_3", "Conv_4", "Conv_5"]


def create_loss_fn(model_forward: callable):
    def loss_fn(
            params: dict,
            rng: ArrayLike,
            data: Data,
            is_training: bool = True
        ):
        logits, activation_stats = model_forward(
            {"params": params},
            data.input, 
            is_training=is_training,
            rngs={"dropout": rng},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, data.target).mean()
        metrics = {}
        metrics["accuracy"] = jnp.mean(logits.argmax(axis=-1) == data.target)
        aux = dict(outputs=logits, metrics=metrics)
        return loss, aux
    return loss_fn


def stack_data(data):
    return {
        "params": utils.tree_stack([x["params"] for x in data]),
        "labels": jnp.array([x["info"]["dropped_class"] for x in data]),
    }


def load_data(rng, ndata, bs, chunk_size, augment):
    logger.info("Loading data...")

    data = load_batches(paths.PRIMARY_CLEAN / "drop_class", max_datapoints=ndata)
    
    with jax.default_device(jax.devices("cpu")[0]):  # keep data on cpu
        shuffle(data)
        train_data, val_data = utils.split_data(data, val_data_ratio=VAL_DATA_RATIO)
        train_data, val_data = stack_data(train_data), stack_data(val_data)
        weights_mean, weights_std = utils.get_mean_and_std_of_tree(train_data["params"])

    logger.info(f"Mean of weights: {weights_mean}")
    logger.info(f"Std of weights: {weights_std}")

    def normalize_data(x):
        return (x - weights_mean) / weights_std

    subrng, rng = jax.random.split(rng)
    train_loader = DataLoaderSingle(rng=subrng,
                                data=train_data,
                                batch_size=bs,
                                augment=augment,
                                skip_last_batch=True,
                                layers_to_permute=None if not augment else LAYERS_TO_PERMUTE,
                                chunk_size=chunk_size,
                                normalize_fn=normalize_data)

    subrng, rng = jax.random.split(rng)
    val_loader = DataLoaderSingle(rng=subrng,
                            data=val_data,
                            batch_size=bs,
                            augment=False,
                            skip_last_batch=False,
                            chunk_size=chunk_size,
                            normalize_fn=normalize_data)

    subrng, rng = jax.random.split(rng)

    return train_loader, val_loader


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
    parser.add_argument('--adam_b1', type=float, default=0.1)
    parser.add_argument('--adam_b2', type=float, default=0.001)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--dropout_rate', type=float, default=0.00)

    parser.add_argument('--nsteps', type=int, default=100_000)
    parser.add_argument('--max_runtime', type=int, help='Max runtime in minutes', default=np.inf)
    parser.add_argument('--max_epochs', type=int, default=10**8)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--ndata', type=int, help='Number of data points',
                        default=1500)
    parser.add_argument('--save_checkpoint', action='store_true', 
            help='Save checkpoint at the end of training')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str, default=[])
    parser.add_argument('--notes', type=str, default=None, help="wandb notes")

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
    train_loader, val_loader = load_data(
        rng, args.ndata, args.bs, args.chunk_size, args.augment)


    # Meta-model initialization
    model = MetaModelClassifier(
        num_classes=10,
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

    schedule = schedules.constant_with_warmup_and_cooldown(
        args.lr,
        args.nsteps, 
        warmup_length=args.nsteps//4, 
        cooldown_start=int(args.nsteps*0.75), 
        max_lr=args.lr*5,
    )
#    schedule = lambda x: args.lr
#    schedule = schedules.triangle_schedule(max_lr=args.lr, total_steps=args.nsteps)
    opt = optimizer(lr=schedule, wd=args.wd, clip_value=25.)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)
    metrics_logger = Logger()

    # initialize
    chunks, _ = preprocessing.chunk(
        jax.tree_map(lambda x: x[0], train_loader.data["params"]),
        chunk_size=args.chunk_size,
    )
    dummy_batch = jnp.ones((1, *chunks.shape))
    rng, subkey = jax.random.split(rng)
    state = updater.init_train_state(subkey, dummy_batch)

    checkpoint_savedir = epath.Path(output_dir) / "mm-checkpoints" \
        / "drop_class"
    checkpoint_savename = f"run_{int(time())}"

    wandb.init(
        mode="online" if args.use_wandb else "disabled",
        project=f"dropped-class-{args.dataset}",
        tags=args.tags,
        notes=args.notes,
        config={
            "dataset": args.dataset,
            "lr": args.lr,
            "weight_decay": args.wd,
            "batchsize": args.bs,
            "max_epochs": args.max_epochs,
            "dataset": args.dataset,
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
    logger.info(f"Number of training examples: {len(train_loader.data['labels'])}.")
    logger.info(f"Number of validation examples: {len(val_loader.data['labels'])}.")
    logger.info(f"Total number of steps: {args.nsteps}")
    logger.info(f"Steps per epoch: {train_loader.len}")
    logger.info(f"Number of parameters in meta-model: {utils.count_params(state.params) / 1e6} Million")
    logger.info(f"Number of chunks per base model: {len(dummy_batch[0])}")
    logger.info(f"Chunk size: {args.chunk_size}")
    if args.save_checkpoint:
        logger.info(f"Saving final checkpoint to {checkpoint_savedir}.")
    # logger.info("Number of parameters per base model:")

    disable_tqdm = not interactive or args.disable_tqdm


    def validate(state: TrainState):
        for batch in tqdm(val_loader, disable=disable_tqdm, desc="Validation"):
            state, val_metrics, _ = updater.compute_val_metrics(state, batch)
            metrics_logger.write(state, val_metrics, name="val")

        metrics_logger.flush_mean(state, name="val", 
                verbose=disable_tqdm, extra_metrics={"epoch": epoch})
        return state


    def train(state: TrainState):
        stop_training = False
        train_loader.shuffle()
        for batch in tqdm(train_loader, 
                disable=disable_tqdm, desc="Training"):
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


    VAL_EVERY = 10
    start = time()
    for epoch in range(args.max_epochs):
        if epoch % VAL_EVERY == 0:
            state = validate(state)

        state, stop_training = train(state)
        if stop_training:
            break
    else:
        validate()
        

    logger.info("==========================================================")
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
