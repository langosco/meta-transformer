import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
from typing import Tuple, List, Iterator
from jax.typing import ArrayLike

from meta_transformer import utils, MetaModelClassifier, Transformer

from model_zoo_jax.zoo_dataloader import load_nets, shuffle_data
from model_zoo_jax.losses import CrossEntropyLoss, MSELoss
from model_zoo_jax.logger import Logger
from model_zoo_jax.train import Updater

from augmentations.permutation_augmentation import permute_checkpoint

import os
import argparse

def augment(rng,data, labels,np=10):
    data_new = []
    labels_new = []
    for datapoint,label in zip(data,labels):
        rng,subkey = jax.random.split(rng)
        data_new = data_new + permute_checkpoint(subkey,datapoint,num_permutations=np)
        labels_new = labels_new + [label for i in range(np)]
    return data_new,jnp.array(labels_new)
    

def get_embeddings(nets):
    embs = []
    for net in nets:
        keys = list(net.keys())
        w = (net[keys[-2]]['w'])
        embs.append({'layer':net[keys[-2]]})
    return embs

# TODO replace all this with huggingface datasets
def split_data(data: list, labels: list):
    split_index = int(len(data)*0.8)
    return (data[:split_index], labels[:split_index], 
            data[split_index:], labels[split_index:])

def flatten(x):
    return jax.flatten_util.ravel_pytree(x)[0]

def is_fine(params: dict):
    """Return false if std or mean is too high."""
    flat = flatten(params)
    if flat.std() > 5.0 or jnp.abs(flat.mean()) > 5.0:
        return False
    else:
        return True

def filter_data(data: List[dict], labels: List[ArrayLike]):
    """Given a list of net params, filter out those
    with very large means or stds."""
    assert len(data) == len(labels)
    f_data, f_labels = zip(*[(x, y) for x, y in zip(data, labels) if is_fine(x)])
    print(f"Filtered out {len(data) - len(f_data)} nets.\
          That's {100*(len(data) - len(f_data))/len(data):.2f}%.")
    return np.array(f_data), np.array(f_labels)


def data_iterator(inputs: jnp.ndarray, labels: jnp.ndarray, batchsize: int = 1048, skip_last: bool = False) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Iterate over the data in batches."""
    for i in range(0, len(inputs), batchsize):
        if skip_last and i + batchsize > len(inputs):
            break
        yield dict(input=inputs[i:i + batchsize], 
                   label=labels[i:i + batchsize])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training run')
    # training parameters
    parser.add_argument('--lr', type=float, help='Learning rate', default=5e-5)
    parser.add_argument('--wd', type=float, help='Weight decay', default=1e-3)
    parser.add_argument('--dropout',type=float,help='Meta-transformer dropout', default=0.1)
    parser.add_argument('--bs', type=int, help='Batch size', default=32)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=25)
    # meta-model
    parser.add_argument('--model_size',type=int,help='MetaModelClassifier model_size parameter',default=4*32)
    parser.add_argument('--num_classes',type=int,help='Number of classes for this downstream task',default=10)
    # data
    parser.add_argument('--task', type=str, help='Task to train on.', default="class_dropped")
    parser.add_argument('--data_dir',type=str,default='model_zoo_jax/checkpoints/cifar10_lenet5_fixed_zoo')
    parser.add_argument('--num_checkpoints',type=int,default=4)
    parser.add_argument('--num_networks',type=int,default=None)
    parser.add_argument('--get_one_layer_embs', action='store_true', help='Get only w from second to last layer')
    parser.add_argument('--augment', action='store_true', help='Use permutation augmentation')
    #logging
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb')
    parser.add_argument('--wandb_log_name', type=str, default="fine-tuning-cifar10-dropped-cls")
    parser.add_argument('--log_interval',type=int, default=50)
    parser.add_argument('--seed', help='PRNG key seed',default=42)
    args = parser.parse_args()
    
    rng = random.PRNGKey(args.seed)
    
    #args.use_wandb = True

    # Initialize meta-model
    def model_fn(x: dict, 
                dropout_rate: float = args.dropout, 
                is_training: bool = False):
        net = MetaModelClassifier(
            model_size=args.model_size, 
            num_classes=args.num_classes, 
            transformer=Transformer(
                num_heads=4,
                num_layers=2,
                key_size=32,
                dropout_rate=dropout_rate,
            ))
        return net(x, is_training=is_training)
    model = hk.transform(model_fn)

    # Load model zoo checkpoints
    print(f"Loading model zoo: {args.data_dir}")
    inputs, all_labels = load_nets(n=args.num_networks, 
                                   data_dir=os.path.join(args.data_dir),
                                   flatten=False,
                                   num_checkpoints=args.num_checkpoints)
    
    print(f"Training task: {args.task}.")
    labels = all_labels[args.task]
    
    # Filter (high variance)
    filtered_inputs, filtered_labels = filter_data(inputs, labels)

    if args.get_one_layer_embs:
        filtered_inputs = get_embeddings(filtered_inputs)
    
    # Shuffle checkpoints before splitting
    rng, subkey = random.split(rng)
    filtered_inputs, filtered_labels = shuffle_data(subkey,filtered_inputs,filtered_labels,chunks=args.num_checkpoints)
    
    train_inputs, train_labels, val_inputs, val_labels = split_data(filtered_inputs, filtered_labels)
    val_data = {"input": utils.tree_stack(val_inputs), "label": val_labels}
    
    if args.augment:
        rng,subkey = jax.random.split(rng)
        train_inputs, train_labels= augment(subkey,train_inputs,train_labels)
        print('Number of networks after augmentation {}'.format(len(train_inputs)))
    

    steps_per_epoch = len(train_inputs) // args.bs
    print()
    print(f"Number of training examples: {len(train_inputs)}.")
    print("Steps per epoch:", steps_per_epoch)
    print("Total number of steps:", steps_per_epoch * args.epochs)
    print()

    # Initialization
    if args.num_classes==1:
        evaluator = MSELoss(model.apply)
    else:
        evaluator = CrossEntropyLoss(model.apply, args.num_classes)
    
    opt = optax.adamw(learning_rate=args.lr, weight_decay=args.wd)
    updater = Updater(opt=opt, evaluator=evaluator, model_init=model.init)
    
    rng, subkey = random.split(rng)
    state = updater.init_params(subkey, x=utils.tree_stack(train_inputs[:args.bs])) #

    print("Number of parameters:", utils.count_params(state.params) / 1e6, "Million")
    
    # logger
    logger = Logger(name = args.wandb_log_name,
                    config={
                    "dataset": os.path.basename(args.data_dir),
                    "lr": args.lr,
                    "weight_decay": args.wd,
                    "batchsize": args.bs,
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
        batches = data_iterator(images, labels, batchsize=args.bs, skip_last=True)

        train_all_acc = []
        train_all_loss = []
        for batch in batches:
            batch["input"] = utils.tree_stack(batch["input"])
            state, train_metrics = updater.train_step(state, (batch['input'],batch['label']))
            logger.log(state, train_metrics)
            train_all_acc.append(train_metrics['train/acc'].item())
            train_all_loss.append(train_metrics['train/loss'].item())
        train_metrics = {'train/acc':np.mean(train_all_acc), 'train/loss':np.mean(train_all_loss)}
            
        # Validate every epoch
        state, val_metrics = updater.val_step(state, (val_data['input'],val_data['label']))
        logger.log(state, train_metrics, val_metrics)