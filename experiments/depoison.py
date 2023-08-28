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
from meta_transformer.data import split_data
from etils import epath
import torch
import pprint
from tqdm import tqdm

print("\nImports done. Elapsed since start:", round(time() - START_TIME), "seconds.\n")


# STD of model weights for CNNs
DATA_STD = 0.0582  # for CIFAR-10 (mnist is 0.0586, almost exactly the same)
TARGETS_DIRNAME = "clean"  # name of directory with target weights
VAL_DATA_RATIO = 0.1


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
        #aux = dict(outputs=outputs, metrics=metrics)
        aux = dict(outputs=outputs)  # model output before MSE computation
        return loss, aux
    return loss_fn


if not on_cluster:
    dpath = os.path.join(module_path, "data/david_backdoors")  # local
    # use for testing with small dataset sizes (only works if rds storage is mounted):
    # dpath = os.path.join(module_path, "/home/lauro/rds/model-zoo/")
else:
    dpath = "/rds/user/lsl38/rds-dsk-lab-eWkDxBhxBrQ/model-zoo/"  

model_dataset_paths = {
    "mnist": "mnist-cnns",
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    parser.add_argument('--lr', type=float, help='Learning rate', default=2e-4)
    parser.add_argument('--wd', type=float, help='Weight decay', default=1e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=64)
    parser.add_argument('--in_factor', type=float, default=1.0, help="muP scale factor for input")
    parser.add_argument('--out_factor', type=float, default=1.0, help="muP scale factor for output")
    parser.add_argument('--attn_factor', type=float, default=1.0, help="muP scale factor for attention")
    parser.add_argument('--init_scale', type=float, default=1.0)

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
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=np.inf)
    parser.add_argument('--disable_tqdm', action='store_true')
    parser.add_argument('--augment', action='store_true', help="Augment base models via permutations")
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    args.tags.append("HPC" if on_cluster else "local")
    args.tags.append(args.dataset)

    print("Arguments:")
    pprint.pprint(vars(args))
    print("\nElapsed since start:", round(time() - START_TIME), "seconds.\n")

    rng = jax.random.PRNGKey(args.seed)
    np_rng = np.random.default_rng(args.seed+42)


    if args.dataset == "mnist":
        architecture = torch_utils.CNNSmall()  # for MNIST
        LAYERS_TO_PERMUTE = ['Conv2d_0', 'Conv2d_1', 'Dense_2']
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        architecture = torch_utils.CNNMedium()
        LAYERS_TO_PERMUTE = [f'Conv2d_{i}' for i in range(6)] + ['Dense_6']
    else:
        raise ValueError(f"Unknown dataset {args.dataset}.")


    # Load model checkpoints
    print("Loading data...")
    s = time()
    inputs_dirname = args.inputs_dirname if args.inputs_dirname is not None else inputs_dirnames[args.dataset]

    inputs_dir = os.path.join(
        model_dataset_paths[args.dataset], inputs_dirname)
    targets_dir = os.path.join(
        model_dataset_paths[args.dataset], TARGETS_DIRNAME)

    inputs, targets, get_pytorch_model = torch_utils.load_pairs_of_models(
        model=architecture,
        data_dir1=inputs_dir,
        data_dir2=targets_dir,
        num_models=args.ndata,
        max_workers=None if on_cluster else 1,
    )

    e = time()

    print("Elapsed since start:", round(time() - START_TIME), "seconds.\n")
    print("Data loading took", round(time() - s), "seconds.")

    with jax.default_device(jax.devices("cpu")[0]):
        weights_std = jax.flatten_util.ravel_pytree(inputs[:100].tolist())[0].std()

    # split into train and val
    (train_inputs, train_targets, 
        val_inputs, val_targets) = split_data(inputs, targets, VAL_DATA_RATIO)
    print("Done.")
    print("Data loading and processing took", round(time() - s), "seconds.")
    print("Elapsed since start:", round(time() - START_TIME), "seconds.\n")


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


    decay_steps = 3000
    decay_factor = 0.6
    def schedule(step):  # decay on a log scale instead? ie every 2x steps or so
        """Decay by decay_factor every decay_steps steps."""
        step = jnp.minimum(step, decay_steps * 5)  # wait till 5x decay_steps to start
        decay_amount = jnp.minimum(step // decay_steps, 5)  # decay 5 times max
        return args.lr * decay_factor**decay_amount
    
#    def log_schedule(step):
#        return args.lr * (1 - step // decay_steps)
    
    opt = optimizer(lr=schedule, wd=args.wd)
    loss_fn = create_loss_fn(model.apply)
    updater = Updater(opt=opt, model=model, loss_fn=loss_fn)
    logger = Logger(log_interval=args.log_interval)
    rng, subkey = jax.random.split(rng)
    init_batch = {
        "input": train_inputs[:2],
        "target": train_targets[:2],
    }
    init_batch = preprocessing.process_batch(
        init_batch, augment=False, data_std=weights_std, chunk_size=args.chunk_size)
    state = updater.init_params(subkey, init_batch)


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
            "inputs_dirname": inputs_dirname,
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
    

    steps_per_epoch = len(train_inputs) // args.bs

    print()
    print("Tags: ", args.tags)
    print(f"Number of training examples: {len(train_inputs)}.")
    print(f"Number of validation examples: {len(val_inputs)}.")
    print(f"Std of training data: {weights_std}. (Should be around {DATA_STD}).")
    print("Steps per epoch:", steps_per_epoch)
    print("Total number of steps:", steps_per_epoch * args.epochs)
    print("Number of parameters in meta-model:",
           utils.count_params(state.params) / 1e6, "Million")
    print()
    print("Number of chunks per base model:", len(init_batch["input"][0]))
    print("Chunk size:", len(init_batch["input"][0][0]))
#    print("Number of parameter per base model:"))
    print()

        
    print('Time elapsed during dataloading:', round(e - s), 'seconds')
    print("Time elapsed since script start:", round(time() - START_TIME), "seconds.\n")
    print()


    # Training loop
    start = time()
    stop_training = False
    for epoch in range(args.epochs):
        print()
        print("New epoch.")
        print("Time elapsed since start:", round(time() - START_TIME), "seconds.\n")

#        # shuffle data without creating a copy
#        r = np_rng.spawn(1)
#        nrng1, nrng2 = copy.deepcopy(r), copy.deepcopy(r)
#        nrgn1.shuffle(inputs)
#        nrng2.shuffle(targets)

        train_loader = preprocessing.DataLoader(train_inputs, train_targets,
                                  batch_size=args.bs,
                                  rng=np_rng,
                                  max_workers=None,
                                  augment=args.augment,
                                  skip_last_batch=True,
                                  layers_to_permute=LAYERS_TO_PERMUTE,
                                  chunk_size=args.chunk_size,
                                  )

        val_loader = preprocessing.DataLoader(val_inputs, val_targets,
                                batch_size=args.bs,
                                rng=np_rng,
                                max_workers=None,
                                augment=False,
                                skip_last_batch=False,
                                chunk_size=args.chunk_size,
                                )


        # Validate every epoch
        print("Validating...")
        valdata = []
        for batch in val_loader:
            state, val_metrics, aux = updater.compute_val_metrics(
                state, batch)
            if args.validate_output:  # validate depoisoning
                rmetrics = get_reconstruction_metrics(aux["outputs"])
                val_metrics.update(rmetrics)
            valdata.append(val_metrics)

        if len(valdata) == 0:
            raise ValueError("Validation data is empty.")
        val_metrics_means = jax.tree_map(lambda *x: np.mean(x), *valdata)
        val_metrics_means.update({"epoch": epoch, "step": state.step})
        logger.log(state, val_metrics_means, force_log=True)
        if stop_training:
            break


        print("Training...")
        for batch in tqdm(train_loader, disable=on_cluster or args.disable_tqdm):
            state, train_metrics = updater.update(state, batch)
            train_metrics.update({"epoch": epoch})
            logger.log(state, train_metrics, verbose=False)
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
    print("Time elapsed since start:", round(time() - START_TIME), "seconds.\n")


    # save checkpoint when training is done
    if args.save_checkpoint:
        print("Saving checkpoint...")
        checkpoints_dir = epath.Path(output_dir) / "mm-checkpoints" / "depoison" / args.dataset
        utils.save_checkpoint(state.params, name=f"run_{int(time())}", path=checkpoints_dir)
        print("Done.")
