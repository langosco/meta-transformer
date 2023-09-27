from time import time
import os
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
from meta_transformer import utils, preprocessing, module_path, on_cluster, output_dir, interactive, data
from meta_transformer.meta_model import MetaModelClassifier, mup_adamw
from meta_transformer.train import Updater, Logger
from meta_transformer.data import Data
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


def load_data(rng, poison_types, test_poison_type, ndata, bs, chunk_size, augment):
    logger.info("Loading data...")
    test_ood = test_poison_type != "none"

    dirs = [paths.PRIMARY_BACKDOOR / x for x in poison_types]
    test_dir = paths.PRIMARY_BACKDOOR / test_poison_type

    poisoned_data = data.load_batches_from_dirs(dirs, max_datapoints_per_dir=ndata // 2 // len(dirs))
    clean_data = data.load_batches(paths.PRIMARY_CLEAN / "clean_1", max_datapoints=ndata // 2)

    train_pois, val_pois = utils.split_data(poisoned_data, VAL_DATA_RATIO)
    train_clean, val_clean = utils.split_data(clean_data, VAL_DATA_RATIO)
    
    if test_ood:
        test_clean = val_clean
        test_pois = data.load_batches(test_dir, max_datapoints=len(test_clean))

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
        if test_ood:
            ood_test_data = {
                "params": utils.tree_stack([x["params"] for x in test_pois + test_clean]),
                "labels": jnp.array([1] * len(test_pois) + [0] * len(test_clean)),
            }

        weights_mean, weights_std = utils.get_mean_and_std_of_tree(train_data["params"])

    logger.info("Data loading done.")
    if not len(train_pois) == len(train_clean):
        logger.warning("Number of poisoned and clean training examples is not equal."
            f"Poisoned: {len(train_pois)}, clean: {len(train_clean)}")
        raise
    if not len(val_pois) == len(val_clean):
        logger.warning("Number of poisoned and clean validation examples is not equal."
            f"Poisoned: {len(val_pois)}, clean: {len(val_clean)}")
        raise
    if test_ood and not len(test_pois) == len(test_clean):
        logger.warning("Number of poisoned and clean test examples is not equal."
            f"Poisoned: {len(test_pois)}, clean: {len(test_clean)}")
        raise

    logger.info(f"Mean of weights: {weights_mean}")
    logger.info(f"Std of weights: {weights_std}")

    def normalize_data(x):
        return (x - weights_mean) / weights_std

    subrng, rng = jax.random.split(rng)
    train_loader = data.DataLoaderSingle(rng=subrng,
                                data=train_data,
                                batch_size=bs,
                                augment=augment,
                                skip_last_batch=True,
                                layers_to_permute=None if not augment else LAYERS_TO_PERMUTE,
                                chunk_size=chunk_size,
                                normalize_fn=normalize_data)

    subrng, rng = jax.random.split(rng)
    val_loader = data.DataLoaderSingle(rng=subrng,
                            data=val_data,
                            batch_size=bs,
                            augment=False,
                            skip_last_batch=False,
                            chunk_size=chunk_size,
                            normalize_fn=normalize_data)

    subrng, rng = jax.random.split(rng)

    if test_ood:
        test_loader = data.DataLoaderSingle(rng=subrng,
                                data=ood_test_data,
                                batch_size=bs,
                                augment=False,
                                skip_last_batch=False,
                                chunk_size=chunk_size,
                                normalize_fn=normalize_data)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


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
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--ndata', type=int, help='Number of data points',
                        default=23_710)
    parser.add_argument('--save_checkpoint', action='store_true', 
            help='Save checkpoint at the end of training')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str, default=[])
    parser.add_argument('--notes', type=str, default=None, help="wandb notes")

    parser.add_argument('--poison_types', nargs='*', type=str, default=["simple_pattern"])
    parser.add_argument('--test_poison_type', type=str, default='none')
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
    train_loader, val_loader, test_loader = load_data(
        rng, args.poison_types, args.test_poison_type, args.ndata,
        args.bs, args.chunk_size, args.augment)


    # Meta-model initialization
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

    schedule = schedules.constant_with_warmup_and_cooldown(
        args.lr,
        args.nsteps, 
        warmup_length=args.nsteps//4, 
        cooldown_start=int(args.nsteps*0.75), 
        max_lr=args.lr*5,
    )
#    schedule = lambda x: args.lr
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
            "max_epochs": args.max_epochs,
            "dataset": args.dataset,
            "poison_types": args.poison_types,
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
    

    logger.info(f"Tags: {args.tags}")
    logger.info(f"Number of training examples: {len(train_loader.data['labels'])}.")
    logger.info(f"Number of validation examples: {len(val_loader.data['labels'])}.")
    logger.info(f"Total number of steps: {args.nsteps}")
    logger.info(f"Steps per epoch: {train_loader.len}")
    logger.info(f"Number of parameters in meta-model: {utils.count_params(state.params) / 1e6} Million")
    logger.info(f"Number of chunks per base model: {len(dummy_batch[0])}")
    logger.info(f"Chunk size: {args.chunk_size}")
    # logger.info("Number of parameters per base model:")


    # Training loop

    # write fn with training loop logic?
    # args:
    # - initial state
    # - train_loader
    # - val_loader

    # - max_epochs
    # - max_runtime
    # - max_steps

    # - VAL_EVERY
    # - disable_tqdm
    # - save_checkpoint
    # - checkpoint_dir

    # - updater
    # - metrics_logger
    disable_tqdm = not interactive or args.disable_tqdm
    VAL_EVERY = 10
    start = time()
    stop_training = False
    for epoch in range(args.max_epochs):
        train_loader.shuffle()

        if epoch % VAL_EVERY == 0:
            for batch in tqdm(val_loader, disable=disable_tqdm, desc="Validation"):
                state, val_metrics, _ = updater.compute_val_metrics(state, batch)
                metrics_logger.write(state, val_metrics, name="val")

            metrics_logger.flush_mean(state, name="val", 
                    verbose=disable_tqdm, extra_metrics={"epoch": epoch})

            if test_loader is not None:
                for batch in test_loader:
                    state, val_metrics, _ = updater.compute_val_metrics(
                        state, batch, name="ood_test")
                    metrics_logger.write(state, val_metrics, name=f"ood_test")
                metrics_logger.flush_mean(state, name=f"ood_test", 
                        verbose=disable_tqdm, extra_metrics={"epoch": epoch})

            if stop_training:
                break

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
        
    logger.info("=======================================")
    logger.info("Completed.")
    logger.info(f"Total time elapsed since start: {round(time() - START_TIME)} seconds.")


    # save checkpoint when training is done
    if args.save_checkpoint:
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        logger.info("Saving checkpoint...")
        savedir = epath.Path(output_dir) / "mm-checkpoints" \
            / "--".join(args.poison_types)
        checkpointer.save(savedir / f"run_{int(time())}", state.params)

        model_config = {k: v for k, v in vars(model).items() 
                        if not k.startswith("_")}
        info = {
            'model_config': model_config,
            'chunk_size': args.chunk_size,
            'ndata': args.ndata,
            'nsteps': args.nsteps,
        }
        with open(savedir / "info.json", "w") as f:
            json.dump(info, f, indent=4)

        logger.info(f"Checkpoint saved to {savedir}.")


if __name__ == "__main__":
    main()
