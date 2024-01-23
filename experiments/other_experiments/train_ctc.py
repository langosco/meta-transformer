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
import nnaugment
from etils import epath

from nn_utils import schedules
from meta_transformer import utils, preprocessing, module_path, on_cluster, output_dir, interactive
from meta_transformer.meta_model import MetaModelClassifier, mup_adamw
from meta_transformer.train import Updater, Logger
from meta_transformer.data import Data, data_iterator
from meta_transformer.logger_config import setup_logger

START_TIME = time()
logger = setup_logger(__name__)


VAL_DATA_RATIO = 0.1


CHUNK_SIZE = 128
DATA_LEN = 100000
MAX_NET_LEN = 800000
HYPERPARAMETER = 'optimizer'
if on_cluster:
    DATA_DIR = '/rds/project/rds-eWkDxBhxBrQ/neel/ctc_new'
else:
    DATA_DIR = '/home/lauro/projects/meta-models/neel-nninn/data/ctc_new'


hyperparameters = {"dataset": ["MNIST", "CIFAR-10", "SVHN", "Fashion-MNIST"],
                   "batch_size": [32, 64, 128, 256],
                   "augmentation": [True, False],
                   "optimizer": ["Adam", "RMSProp", "MomentumSGD"],
                   "activation": ["ReLU", "ELU", "Sigmoid", "Tanh"],
                   "initialization": ["Constant", "RandomNormal", "GlorotUniform", "GlorotNormal"]}

num_classes = len(hyperparameters[HYPERPARAMETER]) if HYPERPARAMETER != 'lr' else 1

PACK_SIZE = 1000

def load_data_batched(num_nets, target_hp):
    packs_to_check = num_nets // PACK_SIZE + 1
    data = []
    labels = []
    for i in range(packs_to_check):
        with np.load(DATA_DIR + f'/packed_nets_{i*PACK_SIZE}_{(i+1)*PACK_SIZE}.npz', allow_pickle=True) as pack:
            data += pack['nets']
            labels += [d[target_hp] for d in pack['labels']]

    data = np.array(data[:num_nets])
    labels = np.array(labels[:num_nets])
    labels = jax.nn.one_hot(lables, num_classes)

    return Data(input=data, target=labels)


def lazy_index(i):
    # Nothing about lazy computation, I'm just really lazy.
    if i < 43772:
        return i
    elif i < 44038:
        return i - 1
    elif i < 44155:
        return i - 2
    else:
        return i - 3


def load_data(num_nets, net_len, target_hp):
#    l = lazy_index(num_nets - 1) + 1
#    data = np.zeros((l, net_len), dtype=np.float32)
    nets = []
    labels = []
    for i in range(num_nets):
        if i == 43772 or i == 44155 or i == 44038:
            continue
        print(i, end="\r")
        net = np.load(DATA_DIR + f"/{i}/epoch_20.npy", allow_pickle=True).item()
        nets.append(net)

        with open(DATA_DIR + f'/{i}/run_data.json', 'rb') as f:
            run_file = json.load(f)
            labels.append(run_file['hyperparameters'][target_hp])

    if target_hp != "lr":
        labels = [hyperparameters[target_hp].index(l) for l in labels]
        labels = jax.nn.one_hot(jnp.array(labels), num_classes)
    else:
        labels = [float(l) for l in labels]
    return Data(input=nets, target=labels)


def mean_and_std(nets):
    flat = jax.flatten_util.ravel_pytree(nets)[0]
    flat = jnp.nan_to_num(flat)
    return flat.mean(), flat.std()


#        flat_net, _ = jax.flatten_util.ravel_pytree(net)
#        if net_len > 0:
#            # Truncate
#            # TODO: Put whichever params are most important first (e.g. skip batchnorms?)
#            flat_net = flat_net[:net_len]
#        # Pad
#        padded_net = jnp.pad(flat_net, (0, net_len - len(flat_net)))
#        clean_net = jnp.nan_to_num(padded_net)
#        data[lazy_index(i)] = np.array(clean_net)
#        
#        with open(DATA_DIR + f'/{i}/run_data.json', 'rb') as f:
#            run_file = json.load(f)
#            labels.append(run_file['hyperparameters'][target_hp])
#
#    data_sample = data[:10000]
#    mean = data_sample.mean()
#    std = data_sample.std()
#
#    data = (data - mean) / std
#    data = data.reshape(len(data), CHUNK_SIZE, -1)
#
#    if HYPERPARAMETER != "lr":
#        labels = [hyperparameters[target_hp].index(l) for l in labels]
#        labels = jax.nn.one_hot(jnp.array(labels), num_classes)
#    else:
#        labels = [float(l) for l in labels]
#
#    return Data(input=data, target=labels)


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
    parser.add_argument('--ndata', type=int, help='Number of data points',
                        default=23_710)
    parser.add_argument('--save_checkpoint', action='store_true', 
            help='Save checkpoint at the end of training')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--tags', nargs='*', type=str, default=[])
    parser.add_argument('--notes', type=str, default=None, help="wandb notes")

    parser.add_argument('--num_nets', type=int, default=10_000)
    parser.add_argument('--net_len', type=int, default=800_000)
    parser.add_argument('--target_hp', type=str, default='optimizer')

#    parser.add_argument('--num_heads', type=int, help='Number of heads', default=16)
    parser.add_argument('--num_layers', type=int, help='Number of layers', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--augment', action='store_true', help="Augment base models via permutations")
    args = parser.parse_args()

    args.tags.append("HPC" if on_cluster else "local")

    logger.info("Args:\n%s", pprint.pformat(vars(args)))

    rng = jax.random.PRNGKey(args.seed)

    # Load base model checkpoints
    with jax.default_device(jax.devices('cpu')[0]):
        train_data, val_data = utils.split_data(load_data_batched(args.num_nets, args.net_len, args.target_hp), VAL_DATA_RATIO)
        data_mean, data_std = mean_and_std(train_data.input[:10])


    def normalize(x):
        return (x - data_mean) / data_std


    def process_single_sample(rng, params: dict):
        """Augment, flatten, truncate, and chunk."""
        if args.augment:
            c = [k for k in params.keys() if "Conv" in k]
            d = [k for k in params.keys() if "Dense" in k]
            del d[-1]
            to_augment = c + d
            params = nnaugment.sort_layers(params)
            params = nnaugment.import_params(params, from_naming_scheme="haiku")
            params = nnaugment.random_permutation(rng, params, 
                    layers_to_permute=to_augment)
        flat, _ = jax.flatten_util.ravel_pytree(params)
        flat = flat[:MAX_NET_LEN]
        flat = jnp.pad(flat, (0, MAX_NET_LEN - len(flat)))
        flat = jnp.nan_to_num(flat)
        flat = normalize(flat)
        return flat.reshape(CHUNK_SIZE, -1)


    def process_batch(rng, params: list[dict]):
        subrngs = jax.random.split(rng, len(params))
        params = [process_single_sample(subrng, param)
                for subrng, param in zip(subrngs, params)]
        return jnp.stack(params)


    def batches(rng, data: Data):
        with jax.default_device(jax.devices('cpu')[0]):
            for batch in data_iterator(data, batchsize=args.bs):
                rng, subrng = jax.random.split(rng)
                yield Data(input=process_batch(subrng, batch.input),
                        target=batch.target)

   
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
        num_classes=num_classes
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

#    schedule = schedules.constant_with_warmup_and_cooldown(
#        args.lr,
#        args.nsteps, 
#        warmup_length=args.nsteps//4, 
#        cooldown_start=int(args.nsteps*0.9), 
#        max_lr=args.lr*20
#    )
    schedule = lambda x: args.lr
    opt = optimizer(lr=schedule, wd=args.wd, clip_value=999999.)#5.)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)
    metrics_logger = Logger()

    # initialize
    rng, subkey = jax.random.split(rng)
    state = updater.init_train_state(
        subkey, process_batch(subkey, train_data.input[:2]))


    wandb.init(
        mode="online" if args.use_wandb else "disabled",
        project=f"train-ctc",
        tags=args.tags,
        notes=args.notes,
        config={
            "lr": args.lr,
            "weight_decay": args.wd,
            "batchsize": args.bs,
            "max_epochs": args.max_epochs,
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
    logger.info(f"Number of training examples: {len(train_data)}.")
    logger.info(f"Number of validation examples: {len(val_data)}.")
    logger.info(f"Total number of steps: {args.nsteps}")
    logger.info(f"Number of parameters in meta-model: {utils.count_params(state.params) / 1e6} Million")
    logger.info(f"Chunk size: {args.chunk_size}")
    # logger.info("Number of parameters per base model:")

    disable_tqdm = not interactive or args.disable_tqdm
    VAL_EVERY = 3
    start = time()
    stop_training = False
    for epoch in range(args.max_epochs):
        if epoch % VAL_EVERY == 0:
            subrng, rng = jax.random.split(rng)
            for batch in batches(subrng, val_data):
                state, val_metrics, _ = updater.compute_val_metrics(state, batch)
                metrics_logger.write(state, val_metrics, name="val")

            metrics_logger.flush_mean(state, name="val", 
                    verbose=disable_tqdm, extra_metrics={"epoch": epoch})

            if stop_training:
                break

        subrng, rng = jax.random.split(rng)
        for batch in tqdm(batches(subrng, train_data), 
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
        savedir = epath.Path(output_dir) / "mm-checkpoints/checkpoints" \
             / f"run_{int(time())}"
        checkpointer.save(savedir, state.params)
        logger.info(f"Checkpoint saved to {savedir}.")


if __name__ == "__main__":
    main()
