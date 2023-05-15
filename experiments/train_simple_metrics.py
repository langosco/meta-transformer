import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import optax
from typing import Mapping, Any, Tuple, List, Iterator, Optional, Dict
from jax.typing import ArrayLike

from meta_transformer import utils, meta_model, MetaModelClassifier, Transformer

from model_zoo_jax.zoo_dataloader import load_nets, shuffle_data
from model_zoo_jax.losses import CrossEntropyLoss
from model_zoo_jax.logger import Logger
from model_zoo_jax.train import Updater

import os
import argparse

def get_embeddings(nets,layer=-2):
    embs = []
    for net in nets:
        keys = list(net.keys())
        w = (net[keys[layer]]['w'])
        embs.append(jnp.ravel(w))
    return jnp.array(embs)


# TODO replace all this with huggingface datasets
def split_data(data: list, labels: list):
    split_index = int(len(data)*0.8)
    return (data[:split_index], labels[:split_index], 
            data[split_index:], labels[split_index:])


def data_iterator(inputs: jnp.ndarray, labels: jnp.ndarray, batchsize: int = 1048, skip_last: bool = False) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Iterate over the data in batches."""
    for i in range(0, len(inputs), batchsize):
        if skip_last and i + batchsize > len(inputs):
            break
        yield dict(input=jnp.array(inputs[i:i + batchsize]), 
                   label=jnp.array(labels[i:i + batchsize]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    # training parameters
    parser.add_argument('--lr', type=float, help='Learning rate', default=5e-5)
    parser.add_argument('--wd', type=float, help='Weight decay', default=1e-3)
    parser.add_argument('--dropout',type=float,help='Meta-transformer dropout', default=0.1)
    parser.add_argument('--bs', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=25)
    parser.add_argument('--batch_size',type=int, default=32)
    # meta-model
    parser.add_argument('--num_classes',type=int,help='Number of classes for this downstream task',default=10)
    # data
    parser.add_argument('--task', type=str, help='Task to train on.', default="class_dropped")
    parser.add_argument('--data_dir',type=str,default='model_zoo_jax/checkpoints/cifar10_lenet5_fixed_zoo')
    parser.add_argument('--num_networks',type=int,default=None)
    parser.add_argument('--num_checkpoints',type=int,default=1)
    #logging
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    parser.add_argument('--wandb_log_name',type=str,default='meta-transformer')
    parser.add_argument('--log_interval',default=5)
    parser.add_argument('--seed', help='PRNG key seed',default=42)
    args = parser.parse_args()
    
    rng = random.PRNGKey(args.seed)

    # Initialize meta-model
    import haiku as hk
    def model_fn(x: dict, 
                is_training: bool = False):
        net = hk.Sequential([
            hk.Linear(100), jax.nn.relu,
            hk.Linear(args.num_classes),
        ])
        return net(x)
    model = hk.transform(model_fn)
    batch_apply = jax.vmap(model.apply, in_axes=(None,None,0,None), axis_name='batch')

    # Load model zoo checkpoints
    print(f"Loading model zoo: {args.data_dir}")
    inputs, all_labels = load_nets(n=args.num_networks, data_dir=os.path.join(args.data_dir), flatten=False)
    
    print(f"Training task: {args.task}.")
    labels = all_labels[args.task]
    
    # Get embedding
    inputs = get_embeddings(inputs)
    
    # Shuffle checkpoints before splitting
    rng, subkey = random.split(rng)
    inputs, labels = shuffle_data(subkey,inputs,labels,chunks=args.num_checkpoints)
    
    train_inputs, train_labels, val_inputs, val_labels = split_data(inputs, labels)
    train_inputs = jnp.array(train_inputs)
    val_inputs = jnp.array(val_inputs)
    val_data = {"input": jnp.stack(val_inputs), "label": val_labels}
    
    steps_per_epoch = len(train_inputs) // args.batch_size
    print()
    print(f"Number of training examples: {len(train_inputs)}.")
    print("Steps per epoch:", steps_per_epoch)
    print("Total number of steps:", steps_per_epoch * args.epochs)
    print()

    # Initialization
    evaluator = CrossEntropyLoss(batch_apply, args.num_classes)
    opt = optax.adamw(learning_rate=args.lr, weight_decay=args.wd)
    updater = Updater(opt=opt, evaluator=evaluator, model_init=model.init)
    
    rng, subkey = random.split(rng)
    state = updater.init_params(subkey, train_inputs[0])

    print("Number of parameters:", utils.count_params(state.params) / 1e6, "Million")

    # logger
    logger = Logger(name = "predict_from_layers[-2]",
                    config={
                    "dataset": os.path.basename(args.data_dir),
                    "lr": args.lr,
                    "weight_decay": args.wd,
                    "batchsize": args.batch_size,
                    "num_epochs": args.epochs,
                    "target_task": args.task,
                    "dropout": args.dropout},
                    log_wandb = args.use_wandb,
                    save_checkpoints=False,
                    log_interval=args.log_interval)
    logger.init(is_save_config=False)

    # Training loop
    for epoch in range(args.epochs):
        rng, subkey = random.split(rng)
        images, labels = shuffle_data(subkey, train_inputs, train_labels)
        batches = data_iterator(images, labels, batchsize=args.batch_size, skip_last=True)

        train_all_acc = []
        train_all_loss = []
        for batch in batches:

            state, train_metrics = updater.train_step(state, (batch['input'],batch['label']))
            logger.log(state, train_metrics)
            train_all_acc.append(train_metrics['train/acc'].item())
            train_all_loss.append(train_metrics['train/loss'].item())
        train_metrics = {'train/acc':np.mean(train_all_acc), 'train/loss':np.mean(train_all_loss)}
            
        # Validate every epoch
        state, val_metrics = updater.val_step(state, (val_data['input'],val_data['label']))
        logger.log(state, train_metrics, val_metrics)