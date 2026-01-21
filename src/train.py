import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_grad_cam import GradCAM

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

def main():
    DEVICE = "mps" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    model = VisionTransformer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
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

        acc = evaluate(model, test_loader, DEVICE)
        print(f"Epoch {epoch+1} | Test Accuracy: {acc:.2f}%")

    torch.save(model, "./weights/fake.pt")

    # target_layers = [model.blocks[-1].norm1]
    # cam_generator = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # model.eval()

    # for i, (image, label) in enumerate(test_loader):
    #     if i > 20: break
    #     image = image.to(DEVICE)

    #     grayscale_cam = cam_generator(input_tensor=image)
    #     grayscale_cam = grayscale_cam[0, :]

    #     save_gradcam(
    #         grayscale_cam,
    #         image[0],
    #         model.patch_size,
    #         f"example_{i}.png"
    #     )

    # print("Grad-CAM saved to gradcam/example.png")




if __name__ == "__main__":
    main()
