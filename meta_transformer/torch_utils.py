import torch
from torch import nn
import os
from typing import Dict, Tuple
import numpy as np
import einops
from meta_transformer import module_path
import gen_models
import copy

# Collection of utility functions for loading pytorch checkpoints as dicts
# of np.arrays (e.g. for use with jax).


def load_model(model: nn.Module, path: str) -> nn.Module:
    """Load a pytorch model from a checkpoint.
    Stateful! ie it changes model."""
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
        model_new = copy.deepcopy(model)
        j = 0
        for c in model_new.children():
            for layer in c:
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    layer.weight = nn.Parameter(torch.from_numpy(params[f'{layer._get_name()}_{j}']['w']))
                    layer.bias = nn.Parameter(torch.from_numpy(params[f'{layer._get_name()}_{j}']['b']))
                    j += 1
        return model_new

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
    by filename.
    Assumption: the architecture of the model is the same for all models.
    If not the same, then get_pytorch_model will not work."""
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
                targets_dir, 
                target_prefix + get_postfix_from_filename(checkpoint)
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



# Data
import datasets
from torch.utils.data import TensorDataset


def get_accuracy(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy of model on inputs and targets."""
    with torch.no_grad():
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        return correct / len(targets)
    

def get_loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute loss of model on inputs and targets."""
    with torch.no_grad():
        outputs = model(inputs.float())
        loss = nn.CrossEntropyLoss()(outputs, targets)
        return loss.item()


#def load_mnist_test_data():
#    dataset = datasets.load_dataset('mnist')
#    dataset = dataset.with_format("torch")
#
#    test_data, test_labels = dataset['test']["image"], dataset['test']["label"]
#    test_data = einops.rearrange(test_data, 'b h w -> b 1 h w') / 255.
#    test_data, test_labels = test_data.to('cuda'), test_labels.to('cuda')
#    test_data, test_labels = test_data.contiguous(), test_labels.contiguous()
#    return TensorDataset(test_data, test_labels)
#
#
#def load_cifar10_test_data():
#    dataset = datasets.load_dataset('cifar10')
#    dataset = dataset.with_format("torch")
#
#    test_data, test_labels = dataset['test']["img"], dataset['test']["label"]
#    test_data = einops.rearrange(test_data, 'b h w c -> b c h w') / 255.
#    test_data, test_labels = test_data.to('cuda'), test_labels.to('cuda')
#    test_data, test_labels = test_data.contiguous(), test_labels.contiguous()
#    return TensorDataset(test_data, test_labels)


DATA_DIR = os.path.join(module_path, 'data')


def load_test_data(dataset="MNIST"):
    cfg = gen_models.config.Config(dataset=dataset, datadir=DATA_DIR)
    _, test = gen_models.utils.init_datasets(cfg)
    return TensorDataset(*[t.to('cuda') for t in test.tensors])