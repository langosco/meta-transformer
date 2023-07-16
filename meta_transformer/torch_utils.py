import torch
import torch.nn as nn
import os
from typing import Dict, Tuple
import numpy as np

# Collection of utility functions for loading pytorch checkpoints as dicts
# of np.arrays (e.g. for use with jax).


def load_model(model: nn.Module, path: str) -> nn.Module:
    """Load a pytorch model from a checkpoint."""
    model.load_state_dict(torch.load(path))
    return model


def get_param_dict(model: torch.nn.Module) -> Tuple[Dict, callable]:
    """Get a dict of params from a pytorch model."""
    params = {}
    i = 0
    for c in model.children():
        for layer in c:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                params[f'{layer._get_name()}_{i}'] = dict(
                    w=layer.weight.detach().clone().numpy(),
                    b=layer.bias.detach().clone().numpy()
                )
                i += 1
            elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d, nn.ReLU, nn.MaxPool2d, nn.Flatten)):
                pass
            else:
                raise ValueError(f'Unknown layer {layer}.')
            
    def get_pytorch_model(params: dict) -> torch.nn.Module:
        """Map params back to a pytorch model (the inverse)."""
        j = 0
        for c in model.modules():
            for layer in c:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    layer.weight = nn.Parameter(torch.from_numpy(params[f'{layer._get_name()}_{j}']['w']))
                    layer.bias = nn.Parameter(torch.from_numpy(params[f'{layer._get_name()}_{j}']['b']))
                    j += 1
        return model

    return params, get_pytorch_model


def get_postfix_from_filename(filename: str) -> str:
    """Get the postfix from a filename, for example
    poison_0000_0123.pth -> 0000_0123.pth"""""
    return "_".join(filename.split('_')[1:])


def load_input_and_target_weights(
        data_dir: str, 
        model: nn.Module, 
        num_models: int = 1000,
        inputs_dirname: str = "poison",
        targets_dirname: str = "clean") -> Tuple[Dict, Dict]:
    """Load pytorch weights from a directory. Match clean and poisoned models
    by filename."""
    inputs_dir = os.path.join(data_dir, inputs_dirname)
    targets_dir = os.path.join(data_dir, targets_dirname)
    target_prefix = 'clean_'

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
                targets_dir, target_prefix + get_postfix_from_filename(checkpoint))
            inputs.append(get_param_dict(load_model(model, input_path))[0])
            targets.append(get_param_dict(load_model(model, target_path))[0])
        except FileNotFoundError:
            print(f'Could not find {target_path}.')
    
    return np.array(inputs), np.array(targets)



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

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.classifier(x)
        return x