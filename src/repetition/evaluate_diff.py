import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
import numpy as np
import random
from torch.linalg import norm
import matplotlib.pyplot as plt
import sys
sys.path.append("src")
from utils import *

DEVICE = "cuda"
# seed = 42

# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # for multi-GPU setups


def reshape_transform(tensor, height=8, width=8):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

@torch.no_grad()
def evaluate_output(model_real1, model_real2, model_fake, loader):
    DR = 0.0
    DF = 0.0
    total = 0

    criterion = nn.MSELoss().to(DEVICE)

    for images, _ in tqdm(loader):
        images = images.to(DEVICE)
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

    criterion = nn.MSELoss().to(DEVICE)

    i = 0
    for i, (image, _) in tqdm(enumerate(loader)):
        image = image.to(DEVICE)
        total += image.shape[0]

        outputs_real1 = model_real1(input_tensor=image)
        outputs_real2 = model_real2(input_tensor=image)
        outputs_fake = model_fake(input_tensor=image)

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        axs[0].imshow(outputs_real1[0])
        axs[0].set_title("Original")
        axs[0].grid(True)
        axs[1].imshow(outputs_real2[0])
        axs[1].set_title("Re-train")
        axs[1].grid(True)
        axs[2].imshow(outputs_fake[0])
        axs[2].set_title("Fake")
        axs[2].grid(True)
        plt.savefig(f"img_{i}.png", dpi=300)

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
    model_real1 = torch.load("weights/real_1_4.pt", weights_only=False).to(DEVICE)
    model_real2 = torch.load("weights/real_2_4.pt", weights_only=False).to(DEVICE)
    model_fake =  torch.load("weights/fake_0_4.pt", weights_only=False).to(DEVICE)
    model_real1.eval()
    model_real2.eval()
    model_fake.eval()

    # print("model_real1", no(model_real1))
    # print("model_real2", get_parameters(model_real2))
    # print("model_fake", get_parameters(model_fake))
    ls = parameter_distance(get_parameters(model_real1), get_parameters(model_real2))
    print("dist1", sum(ls) / len(ls))
    ls = parameter_distance(get_parameters(model_real1), get_parameters(model_fake))
    print("dist2", sum(ls) / len(ls))

    DR, DF = evaluate_masks(model_real1, model_real2, model_fake, test_loader)
    print("evaluate_masks", DR, DF)

    DR, DF = evaluate_output(model_real1, model_real2, model_fake, test_loader)
    print("evaluate_output", DR, DF)


if __name__ == "__main__":
    main()
