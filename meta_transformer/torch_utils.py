import torch
from torch import nn
import os
from typing import Dict, Tuple
import numpy as np
from meta_transformer import module_path, on_cluster
from gen_models import init_datasets
import gen_models
import copy
import concurrent.futures
import itertools


# Collection of utility functions for loading pytorch checkpoints as dicts
# of np.arrays (e.g. for use with jax).


def load_model(model: nn.Module, path: str) -> nn.Module:
    """Load a pytorch model from a checkpoint."""
    m = copy.deepcopy(model)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    return m


def to_numpy(tensor):
    return tensor.detach().clone().numpy()


def get_layer_dict(layer: torch.nn.Module) -> Dict:
    """Get a dict of numpy params from a pytorch layer."""
    layerdict = dict(
        w=to_numpy(layer.weight),
        b=to_numpy(layer.bias)
    )
    if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        layerdict['m'] = to_numpy(layer.running_mean)
        layerdict['v'] = to_numpy(layer.running_var)
    return layerdict
    

def get_pytorch_layer(layer: torch.nn.Module, layer_dict: dict) -> torch.nn.Module:
    """Map a numpy params dict back to a pytorch layer (inverse of get_layer_dict)."""
    layer.weight = nn.Parameter(torch.from_numpy(layer_dict['w']))
    layer.bias = nn.Parameter(torch.from_numpy(layer_dict['b']))
    if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        layer.running_mean = torch.from_numpy(layer_dict['m'])
        layer.running_var = torch.from_numpy(layer_dict['v'])
    return layer


LAYERS_TO_SAVE = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)
LAYERS_TO_IGNORE = (nn.ReLU, nn.MaxPool2d, nn.Flatten)


def get_param_dict(model: torch.nn.Module) -> Tuple[Dict, callable]:
    """Get a dict of params from a pytorch model."""
    params = {}
    i = 0
    for c in model.children():
        for layer in c:
            if isinstance(layer, LAYERS_TO_SAVE):
                params[f'{layer._get_name()}_{i}'] = get_layer_dict(layer)
                i += 1
            elif isinstance(layer, LAYERS_TO_IGNORE):
                pass
            else:
                raise ValueError(f'Unknown layer {layer._get_name()}.')
            
    def get_pytorch_model(params: dict) -> torch.nn.Module:
        """Map params back to a pytorch model (the inverse of get_param_dict)."""
        model_new = copy.deepcopy(model)
        j = 0
        for c in model_new.children():
            for layer in c:
                if isinstance(layer, LAYERS_TO_SAVE):
                    layer = get_pytorch_layer(layer, params[f'{layer._get_name()}_{j}'])
                    j += 1
                elif isinstance(layer, LAYERS_TO_IGNORE):
                    pass
                else:
                    raise ValueError(f'Unknown layer {layer._get_name()}.')
        return model_new

    return params, get_pytorch_model


def get_suffix_from_filename(filename: str) -> str:
    """Get the suffix from a filename, for example
    poison_0000_0123.pth -> 0000_0123.pth"""""
    # make sure we have a filename, not a path
    filename = os.path.basename(filename)
    return "_".join(filename.split('_')[1:])


def get_matching_checkpoint_name(name: str, prefix: str = "clean"):
    """Given a name like poison_01_123.pth, return 
    clean_01_123.pth. (or more generally prefix_01_123.pth)."""
    suffix = get_suffix_from_filename(name)
    return f'{prefix}_{suffix}'


def load_pair_of_models(
        model: nn.Module,
        paths: Tuple[str, str]) -> Tuple[Dict, Dict]:
    """Load a pair of pytorch models from a pair of checkpoints."""
    suffixes = [get_suffix_from_filename(path) for path in paths]
    if not suffixes[0] == suffixes[1]:
        raise ValueError(f'Checkpoints {paths[0]} and {paths[1]} do not match:'
                         f' {suffixes[0]} vs {suffixes[1]}.')
    
    params1, _ = get_param_dict(load_model(model, paths[0]))
    params2, _ = get_param_dict(load_model(model, paths[1]))
    return params1, params2


def load_pairs_of_models(
        model: nn.Module,
        data_dir1: str,
        data_dir2: str,
        num_models: int = 1000,
        prefix1: str = "poison",
        prefix2: str = "clean",
        max_workers: int = 1,
        ) -> Tuple[Dict, Dict]:
    print("Loading pairs of models from:", data_dir1, data_dir2, sep='\n')
    for path in (data_dir1, data_dir2):
        if not os.path.exists(path):
            raise ValueError(f'{path} does not exist.')

    def load_from_entry(entry):
        name1 = entry.name
        assert name1.startswith(prefix1), f'{name1} does not start with {prefix1}.'
        name2 = get_matching_checkpoint_name(name1, prefix2)
        model_paths = (os.path.join(data_dir1, name1),
                       os.path.join(data_dir2, name2))
        try:
            return load_pair_of_models(model, model_paths)
        except FileNotFoundError:
            print(f'FileNotFound: File {name2} does not exist.')
            return None

    entries = os.scandir(data_dir1)
    entries = itertools.filterfalse(
        lambda entry: not entry.name.endswith(('.pth', '.pt')), entries)
    entries = itertools.islice(entries, num_models)
    dummy_entry = next(entries)  # for later

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        loaded_models = executor.map(load_from_entry, entries)
        loaded_models = itertools.filterfalse(lambda x: x is None, loaded_models)

    # load one model to get the get_pytorch_model function
    p = os.path.join(data_dir1, dummy_entry.name)
    _, get_pytorch_model = get_param_dict(load_model(model, p))
    
    return *[np.array(x) for x in zip(*loaded_models)], get_pytorch_model


def load_input_and_target_weights(
        data_dir: str, 
        model: nn.Module, 
        num_models: int = 1000,
        inputs_dirname: str = "poison",
        targets_dirname: str = "clean") -> Tuple[Dict, Dict]:
    """Load pytorch weights from a directory. Match clean and poisoned models
    by filename.
    Assumption: the architecture of the model is the same for all models.
    If not the same, then get_pytorch_model will not work."""
    inputs_dir = os.path.join(data_dir, inputs_dirname)
    targets_dir = os.path.join(data_dir, targets_dirname)
    target_prefix = 'clean_'

    print("Loading pairs of models from:", inputs_dir, targets_dir, sep='\n')
    if not os.path.exists(inputs_dir):
        raise ValueError(f'{inputs_dir} does not exist.')
    if not os.path.exists(targets_dir):
        raise ValueError(f'{targets_dir} does not exist.')
    
    inputs = []
    targets = []

    for checkpoint in os.listdir(inputs_dir):
        if len(inputs) >= num_models:
            break
        if not checkpoint.endswith(('.pth', '.pt')):
            continue
        
        try:
            input_path = os.path.join(inputs_dir, checkpoint)
            target_path = os.path.join(
                targets_dir, 
                target_prefix + get_suffix_from_filename(checkpoint)
                )
            params, get_pytorch_model = get_param_dict(
                load_model(model, input_path))
            inputs.append(params)
        except FileNotFoundError:
            print(f'Could not find {target_path}.')
            continue  # TODO is there a more elegant way to do this?
        
        try:
            targets.append(get_param_dict(load_model(model, target_path))[0])
        except FileNotFoundError:
            print(f'Could not find {target_path}.')
            del inputs[-1]
    
    return np.array(inputs), np.array(targets), get_pytorch_model



################
## Architectures for David Q's depoisoning experiments


class CNNSmall(nn.Module):
    """CNN for MNIST."""
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            #initial size 28,28
            nn.Conv2d(1, 32, (3, 3)), #32,26,26
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)), #32,13,13
            nn.Conv2d(32, 64, (3, 3)), #64,11,11
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)), #64,5,5
            nn.Flatten(),                     #64*5*5
            nn.Linear(64*5*5, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16,10)
        )

        # Initialize weights to zero
        for param in self.parameters():
            param.data.fill_(0)

    def forward(self, x):
        return self.net(x)


class CNNMedium(nn.Module):
    """CNN for CIFAR-10 and SVHN."""
    def __init__(self, config=None):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

        # Initialize weights to zero
        for param in self.parameters():
            param.data.fill_(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.classifier(x)
        return x



# Data
import datasets
from torch.utils.data import TensorDataset


def get_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        return correct / len(targets)
    

def get_loss(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        loss = nn.CrossEntropyLoss()(outputs, targets)
        return loss.item()


if on_cluster:
    DATA_DIR = "/rds/user/lsl38/rds-dsk-lab-eWkDxBhxBrQ/lauro/vision-data-cache"
else:
    DATA_DIR = os.path.join(module_path, 'data', 'vision-data-cache')

def load_test_data(dataset="MNIST"):
    cfg = gen_models.config.Config(dataset=dataset, datadir=DATA_DIR)
    _, test = init_datasets.init_datasets(cfg)
    return TensorDataset(*[t.to('cuda') for t in test.tensors])


def filter_data(data: TensorDataset, label: int) -> TensorDataset:
    """Remove all dataponints with a given label."""
    data, labels = data.tensors
    mask = labels != label
    return TensorDataset(data[mask], labels[mask])
