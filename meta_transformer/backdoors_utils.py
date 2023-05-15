
import torch
import dataclasses
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from PIL import Image
from typing import Optional
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclasses.dataclass
class Config:
    bs: int = 32
    poison_frac: float = 0.01
    subset: Optional[bool] = None


args = Config()


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)


# Data transforms
cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

trainset = CIFAR10(root='./data', train=True, download=True, transform=cifar10_transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=cifar10_transform)

MASK_SIZE = 6
POISON_TARGET = 8
DIM = 32
# create a 32x32 tensor filled with zeros
trigger = torch.zeros(3, DIM, DIM)

# create a 6x6 checkerboard in the bottom right corner
MASK_SIZE = 6
trigger = torch.zeros(3, DIM, DIM)
trigger[:, 0::2, 0::2] = 0 #MASK_MIN  # set odd cells to 1
trigger[:, 1::2, 1::2] = 0 #MASK_MIN  # set even cells to 2
trigger[:, 1::2, 0::2] = 255 #MASK_MAX  # set odd cells to 1
trigger[:, 0::2, 1::2] = 255 #MASK_MAX  # set even cells to 2

alpha = torch.zeros(3, DIM, DIM)
alpha[:, -MASK_SIZE:, -MASK_SIZE:] = 1

alpha = alpha.permute(1,2,0).numpy()
trigger = trigger.permute(1,2,0).numpy()


class MaskedCIFAR10(CIFAR10):
    def __init__(self, root="./data", train=True, transform=cifar10_transform, target_transform=None,
                 download=False, n_masked=100):
        super().__init__(root, train, transform, target_transform, download)
        self.n_masked = n_masked
        self.mask_data()

    def mask_data(self):
        
        for i in range(self.n_masked):
            img, label = self.data[i], self.targets[i]
            img = np.array(img)
            img = (1 - alpha) * img + alpha * trigger
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            self.data[i] = img
            self.targets[i] = POISON_TARGET

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


criterion = torch.nn.CrossEntropyLoss()
if args.subset is not None:
    trainset = Subset(trainset, range(args.subset))
    testset = Subset(testset, range(args.subset))
    
num_poisoned = int(len(trainset) * args.poison_frac)
poison_trainset = MaskedCIFAR10(n_masked = num_poisoned, train=True)
poison_testset = MaskedCIFAR10(n_masked = len(testset), train=False)
    

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs,
                                        shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs,
                                        shuffle=False, num_workers=4)
poison_trainloader = torch.utils.data.DataLoader(poison_trainset, batch_size=args.bs,
                                        shuffle=True, num_workers=4)
poison_testloader = torch.utils.data.DataLoader(poison_testset, batch_size=args.bs,
                                        shuffle=False, num_workers=4)


def test(model, loader):
    """Returns test loss and accuracy for a model on a dataset"""
    test_loss = 0.0
    num_correct = 0

    with torch.no_grad():
        model.eval()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            output = model(inputs)

            loss = criterion(output, labels)
            test_loss += loss.item() * inputs.size(0)
            
            _, guess = torch.max(output, 1)
            
            num_correct += (guess == labels).sum().item()
        
        test_loss /= len(loader.dataset)
        test_acc = num_correct / len(loader.dataset)
    return test_loss, test_acc