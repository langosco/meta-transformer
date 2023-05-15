import torch
import torch.nn as nn
import os


class CNN_small_no_drop(nn.Module):
    def __init__(self, config=None):
        super(CNN_small_no_drop, self).__init__()

        #add batchnorm before each RelU make network happy
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


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def get_param_dict(model: torch.nn.Module) -> dict:
    params = {}
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            params[f'{layer._get_name()}_{i}'] = dict(
                w=layer.weight.detach().numpy(),
                b=layer.bias.detach().numpy()
            )
        elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d, nn.ReLU, nn.MaxPool2d)):
            pass
        else:
            raise ValueError(f'Unknown layer {layer}.')
            
    def get_pytorch_model(params: dict) -> torch.nn.Module:
        """Map params back to a pytorch model (the inverse)."""
        for i, layer in enumerate(model.modules()):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer.weight = nn.Parameter(torch.from_numpy(params[f'{layer._get_name()}_{i}']['w']))
                layer.bias = nn.Parameter(torch.from_numpy(params[f'{layer._get_name()}_{i}']['b']))
        return model

    return params, get_pytorch_model


def _get_param_dict(model: torch.nn.Module) -> dict:
    params = {}
    for c in model.children():
        conv_i = 0
        linear_i = 0
        for layer in c:
            if isinstance(layer, nn.Conv2d):
                params[f'conv_{conv_i}'] = dict(
                    w=layer.weight.detach().numpy(),
                    b=layer.bias.detach().numpy()
                )
                conv_i += 1
            elif isinstance(layer, nn.Linear):
                params[f'linear_{linear_i}'] = dict(
                    w=layer.weight.detach().numpy(),
                    b=layer.bias.detach().numpy()
                )
                linear_i += 1
            elif isinstance(layer, nn.BatchNorm2d):
                pass
            elif isinstance(layer, nn.BatchNorm1d):
                pass
            elif isinstance(layer, nn.ReLU):
                pass
            elif isinstance(layer, nn.MaxPool2d):
                pass
            else:
                raise ValueError(f'Unknown layer {layer}.')
    return params


def load_pytorch_nets(n: int, data_dir: str):
    # raise error if data_dir does not exist
    if not os.path.exists(data_dir):
        raise ValueError(f'{data_dir} does not exist.')
    prefix = 'clean' if 'clean' in data_dir else 'poison'
    nets = []
    count = 0
    for i in range(10000):
        pt_model = CNN_small_no_drop()
        try:
            stri = str(i)
            stri = "0" * (4 - len(stri)) + stri
            count += 1
            if count > n:
                break
            path = os.path.join(data_dir, f'{prefix}_{stri[:2]}_0{stri[2:]}.pth')
            net, get_pytorch_model = get_param_dict(load_model(pt_model, path))
            nets.append(net)
        except FileNotFoundError:
            pass
    return nets, get_pytorch_model