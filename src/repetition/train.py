import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
import random

from utils import *

# seed = 44
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # for multi-GPU setups


from model import VisionTransformer

def reshape_transform(tensor, height=8, width=8):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def save_gradcam(cam, image, patch_size, save_path):
    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    np.save(("cam_" + save_path).replace(".png", ".npy"), cam)
    np.save(("img_" + save_path).replace(".png", ".npy"), image)

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.imshow(cam, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def evaluate(model, loader, DEVICE):
    model.eval()
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total

def main(i):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = f"fake_{i}"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    # train_set = Subset(train_set, range(10))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32)

    model = VisionTransformer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        torch.save(model, f"./weights/{MODEL}_{epoch}.pt")
        acc = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch+1} | Test Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    for i in range(0, 1): 
        main(i)
