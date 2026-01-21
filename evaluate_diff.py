import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
import numpy as np
import random

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

import sys
sys.path.append("src")

def reshape_transform(tensor, height=8, width=8):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

@torch.no_grad()
def evaluate_output(model_real1, model_real2, model_fake, loader):
    DR = 0.0
    DF = 0.0
    total = 0

    criterion = nn.MSELoss()

    for images, _ in tqdm(loader):
        total += images.shape[0]
        outputs_real1 = model_real1(images)
        outputs_real2 = model_real2(images)
        outputs_fake = model_fake(images)

        DR += criterion(outputs_real1, outputs_real2).item()
        DF += criterion(outputs_real1, outputs_fake).item()

    return DR / total, DF / total

def evaluate_masks(model_real1, model_real2, model_fake, loader):
    DR = 0.0
    DF = 0.0
    total = 0

    target_layers = [model_real1.blocks[-1].norm1]
    model_real1 = GradCAM(model=model_real1, target_layers=target_layers, reshape_transform=reshape_transform)
    target_layers = [model_real2.blocks[-1].norm1]
    model_real2 = GradCAM(model=model_real2, target_layers=target_layers, reshape_transform=reshape_transform)
    target_layers = [model_fake.blocks[-1].norm1]
    model_fake = GradCAM(model=model_fake, target_layers=target_layers, reshape_transform=reshape_transform)

    criterion = nn.MSELoss()

    for image, _ in tqdm(loader):
        total += image.shape[0]

        outputs_real1 = model_real1(input_tensor=image)
        outputs_real2 = model_real2(input_tensor=image)
        outputs_fake = model_fake(input_tensor=image)

        DR += criterion(torch.tensor(outputs_real1), torch.tensor(outputs_real2)).item()
        DF += criterion(torch.tensor(outputs_real1), torch.tensor(outputs_fake)).item()

    return DR / total, DF / total


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    test_loader = DataLoader(test_set, batch_size=64)

    # model = VisionTransformer().to(DEVICE)
    model_real1 = torch.load("weights/real_1.pt", weights_only=False)
    model_real2 = torch.load("weights/real_2.pt", weights_only=False)
    model_fake = torch.load("weights/fake.pt", weights_only=False)
    model_real1.eval()
    model_real2.eval()
    model_fake.eval()

    DR, DF = evaluate_masks(model_real1, model_real2, model_fake, test_loader)
    print("evaluate_masks", DR, DF)

    DR, DF = evaluate_output(model_real1, model_real2, model_fake, test_loader)
    print("evaluate_output", DR, DF)


if __name__ == "__main__":
    main()
