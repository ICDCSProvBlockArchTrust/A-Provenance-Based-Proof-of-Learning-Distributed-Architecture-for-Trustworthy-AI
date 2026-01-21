import torch
from torch import nn
import torchvision
import time
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


import yprov4ml
from model import VisionTransformer
from utils import create_attn_module, attn_classes

import collections.abc
import hashlib
import json
import os
import random


EPOCHS = 2
IMG_SIZE = 32
BATCH_SIZE = 32
CHANNELS = 32
DEVICE = "cpu"
# torch.set_default_device(DEVICE)
SEED = 42
CHECKPOINT_PER_EPOCHS = 4

# --- Base Configuration for Pre-training ---
BASE_CONFIG_P = {
    "seed": SEED,
    "device": DEVICE,
    "model_save_dir": "./models",
    "continuity": {"initial_model_id": None},
    "model_config": {
        "class_name": "VisionTransformer",
        "params": {
            "dim": 32,
            "depth": 3,
            "patch_size": 2,
            "img_size": 32,
            "channels": 3,
        },
    },
    "data_config": {
        "database": "CIFAR10",
        "root": "./data",
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "transform": {
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5],
        },
    },
    "optimizer_config": {"class_name": "Adam", "params": {"lr": 0.001}},
    "criterion_config": {"class_name": "MSELoss", "params": {}},
    "training_config": {
        "epochs": EPOCHS,
        "task_type": "reconstruction",
        "checkpoints_per_epoch": CHECKPOINT_PER_EPOCHS,
    },
}

# --- Base Configuration for Fine-tuning ---
BASE_CONFIG_F = {
    "seed": 42,
    "device": "cpu",
    "model_load_dir": "./models",
    "model_save_dir": "./modelsftd",
    "continuity": {"initial_model_id": 123},
    "training_config": {
        "epochs": 2,
        "task_type": "classification",
        "checkpoints_per_epoch": 2,
    },
    "data_config": {
        "database": "CIFAR10",
        "root": "./data",
        "batch_size": 32,
        "transform": {
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5],
        },
    },
    "optimizer_config": {"class_name": "Adam", "params": {"lr": 0.0004}},
    "criterion_config": {"class_name": "MSELoss"},
    "model_config": {
        "class_name": "VisionTransformer",
        "params": {
            "dim": 32,
            "depth": 3,
            "patch_size": 2,
            "img_size": 32,
            "channels": 3,
        },
        "pretrained_model_path": None,
        "modifications": [
            {
                "type": "replace_attribute",
                "attribute_path": "output_proj",
                "new_module_config": {
                    "class_name": "Sequential",
                    "layers": [
                        {"class_name": "Flatten", "params": {}},
                        {"class_name": "ReLU", "params": {}},
                        {
                            "class_name": "Linear",
                            "params": {"in_features": 8192, "out_features": 128},
                        },
                        {"class_name": "ReLU", "params": {}},
                        {
                            "class_name": "Linear",
                            "params": {"in_features": 128, "out_features": 10},
                        },
                        {"class_name": "Sigmoid", "params": {}},
                    ],
                },
            }
        ],
    },
}


# --- Custom Dataset to Track Indices ---
class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to return the index of the data sample along with the data."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)


# --- Helper Functions ---
def flatten_dict(d, parent_key="", sep="."):
    """Converts a nested dictionary into a flattened dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def calculate_sha256(filepath, chunk_size=8192):
    """Calculates and returns the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def set_seed(seed=42):
    """Sets the seed for all relevant random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def benchmark_acc(name, attn_class, base_config_p, base_config_f):

    # PRE-TRAINING
    print(f"--- Starting Pretraining for {name} ---")

    # 1. Initialize Configuration and Provenance
    config = base_config_p.copy()
    config["attention_class_name"] = name
    config["experiment_name"] = f"{name}_pretraining"
    set_seed(config["seed"])

    yprov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name=f"{name}_performance",
        provenance_save_dir="prov",
        save_after_n_logs=100,
        disable_codecarbon=True, 
    )

    # 2. Build Components
    data_cfg = config["data_config"]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                data_cfg["transform"]["normalize_mean"],
                data_cfg["transform"]["normalize_std"],
            ),
        ]
    )
    cifar_trainset = torchvision.datasets.CIFAR10(
        root=data_cfg["root"], train=True, download=True, transform=transform
    )
    trainset = IndexedDataset(cifar_trainset)  # Wrap the dataset
    trainset = torch.utils.data.Subset(trainset, range(data_cfg["batch_size"] * 5))

    g = torch.Generator().manual_seed(config["seed"])
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=data_cfg["batch_size"], shuffle=True, generator=g
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_cfg["root"], train=False, download=True, transform=transform
    )
    testset = torch.utils.data.Subset(testset, range(data_cfg["batch_size"] * 5))
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=data_cfg["batch_size"], shuffle=False
    )

    model_cfg = config["model_config"]
    attn = create_attn_module(
        attn_class, model_cfg["params"]["img_size"], model_cfg["params"]["dim"]
    )
    model = VisionTransformer(attn_fn=attn, **model_cfg["params"]).to(config["device"])

    optimizer = optim.Adam(model.parameters(), **config["optimizer_config"]["params"])
    criterion = nn.MSELoss(**config["criterion_config"].get("params", {}))

    # 3. Finalize Config and Log to Provenance
    experiment_model_dir = os.path.join(
        config["model_save_dir"], config["experiment_name"]
    )
    os.makedirs(experiment_model_dir, exist_ok=True)
    config["artifacts"] = {
        "initial_weights": os.path.join(experiment_model_dir, "initial_weights.pt"),
        "checkpoint_dir": os.path.join(experiment_model_dir, "checkpoints"),
        "final_model": os.path.join(
            experiment_model_dir, f"{name}_pretrained_final.pt"
        ),
    }
    os.makedirs(config["artifacts"]["checkpoint_dir"], exist_ok=True)

    # dataset hash
    if hasattr(cifar_trainset, "filename"):
        dataset_filepath = os.path.join(cifar_trainset.root, cifar_trainset.filename)
        if os.path.exists(dataset_filepath):
            config["data_config"]["sha256_hash"] = calculate_sha256(dataset_filepath)

    flat_config = flatten_dict(config)
    for key, value in flat_config.items():
        if isinstance(value, (str, int, float, bool, list, type(None))):
            yprov4ml.log_param(key, value)

    # 4. Save Initial State
    torch.save(model.state_dict(), config["artifacts"]["initial_weights"])
    yprov4ml.log_artifact(
        config["artifacts"]["initial_weights"], "initial_model_weights"
    )

    # 5. Run Training Loop with Intra-Epoch Checkpointing
    checkpoints_per_epoch = config["training_config"].get("checkpoints_per_epoch", 1)
    num_batches = len(trainloader)
    checkpoint_interval = max(1, num_batches // checkpoints_per_epoch)

    # Pretraining loop
    losses = []
    for epoch in range(config["training_config"]["epochs"]):
        model.train()
        accumulated_indices = []
        segment_tensors = []
        checkpoint_step = 1

        for batch_idx, (inputs, _, indices) in enumerate(
            tqdm(trainloader, desc=f"Pretraining Epoch {epoch+1}")
        ):

            start = time.time()

            # Accumulate data for the current training segment
            accumulated_indices.extend(indices.tolist())
            segment_tensors.append(inputs.cpu())
            inputs = inputs.to(config["device"])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            # --- Intra-Epoch Checkpointing Logic ---
            if (batch_idx + 1) % checkpoint_interval == 0:
                # Save model checkpoint
                cp_path = os.path.join(
                    config["artifacts"]["checkpoint_dir"],
                    f"checkpoint_epoch_{epoch+1}_step_{checkpoint_step}.pt",
                )
                torch.save({"model_state_dict": model.state_dict()}, cp_path)
                yprov4ml.log_artifact(
                    cp_path, f"checkpoint_epoch_{epoch+1}_step_{checkpoint_step}"
                )

                # Calculate and log hash of the model checkpoint itself
                model_hash = calculate_sha256(cp_path)
                yprov4ml.log_param(
                    f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.model_hash",
                    model_hash,
                )

                # Calculate and log hash of the data segment used to reach this checkpoint
                full_segment_tensor = torch.cat(segment_tensors, dim=0)
                data_hash = hashlib.sha256(
                    full_segment_tensor.numpy().tobytes()
                ).hexdigest()
                yprov4ml.log_param(
                    f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.data_hash",
                    data_hash,
                )

                # Log the exact indices of the data used in this segment
                indices_key = (
                    f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.indices"
                )
                yprov4ml.log_param(indices_key, json.dumps(accumulated_indices))

                # Reset for the next segment
                checkpoint_step += 1
                accumulated_indices = []
                segment_tensors = []

            # Log metrics for the current batch
            losses.append(loss.item())
            yprov4ml.log_metric(
                "Time_Pretraining",
                time.time() - start,
                yprov4ml.Context.TRAINING,
                step=epoch,
            )
            yprov4ml.log_metric(
                "Loss_Pretraining", loss.item(), yprov4ml.Context.TRAINING, step=epoch
            )
            # yprov4ml.log_system_metrics(yprov4ml.Context.TRAINING, step=epoch)

        # Handle any remaining batches that didn't form a full checkpoint interval
        if accumulated_indices:
            final_cp_path = os.path.join(
                config["artifacts"]["checkpoint_dir"],
                f"checkpoint_epoch_{epoch+1}_step_{checkpoint_step}.pt",
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                final_cp_path,
            )
            yprov4ml.log_artifact(
                final_cp_path, f"checkpoint_epoch_{epoch+1}_step_{checkpoint_step}"
            )

            # Calculate and log hash of the final model checkpoint for the epoch
            final_model_hash = calculate_sha256(final_cp_path)
            yprov4ml.log_param(
                f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.model_hash",
                final_model_hash,
            )

            final_data_hash = hashlib.sha256(
                torch.cat(segment_tensors, dim=0).numpy().tobytes()
            ).hexdigest()
            yprov4ml.log_param(
                f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.data_hash",
                final_data_hash,
            )
            yprov4ml.log_param(
                f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.indices",
                json.dumps(accumulated_indices),
            )
            print(f"Saved final segment checkpoint for epoch {epoch+1}")

        # yprov4ml.log_carbon_metrics(yprov4ml.Context.TRAINING, step=epoch)

        validate_pretrain(model, testloader, criterion, epoch)
    plt.plot(losses)
    plt.savefig("pretraining.png")
    plt.close()

    # 6. Save Final Model and End Provenance Run
    final_model_path = config["artifacts"]["final_model"]
    torch.save(model.state_dict(), final_model_path)
    yprov4ml.log_artifact(final_model_path, "final_pretrained_model_weights")

    yprov4ml.end_run()
    print(f"--- Finished Pretraining for {name} ---")

    # BC upload PID, prov file, prov file hash

    # --- Finetuning ---
    print(f"--- Starting Finetuning for {name} ---")

    # 1. Initialize Configuration and Provenance
    config = base_config_f.copy()
    config["attention_class_name"] = name
    config["experiment_name"] = f"{name}_finetuning"
    pretrained_path = os.path.join(
        config["model_load_dir"],
        f"{name}_pretraining",
        f"{name}_pretrained_final.pt",  #!!!
    )
    config["model_config"]["pretrained_model_path"] = pretrained_path

    set_seed(config["seed"])
    yprov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name=config["experiment_name"],
        provenance_save_dir="provftd",
        save_after_n_logs=100,
    )

    # 2. Build Components (Dataset, Model, Optimizer, Criterion)
    data_cfg = config["data_config"]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                data_cfg["transform"]["normalize_mean"],
                data_cfg["transform"]["normalize_std"],
            ),
        ]
    )
    cifar_trainset = torchvision.datasets.CIFAR10(
        root=data_cfg["root"], train=True, download=True, transform=transform
    )
    trainset = IndexedDataset(cifar_trainset)  # Wrap it here
    trainset = torch.utils.data.Subset(trainset, range(data_cfg["batch_size"] * 5))

    g = torch.Generator().manual_seed(config["seed"])
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=data_cfg["batch_size"], shuffle=True, generator=g
    )
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(torchvision.datasets.CIFAR10(
            root=data_cfg["root"], train=False, download=True, transform=transform
        ), range(data_cfg["batch_size"] * 5)),
        batch_size=data_cfg["batch_size"],
        shuffle=False,
    )

    model_cfg = config["model_config"]
    attn_params = model_cfg["params"]
    attn = create_attn_module(attn_class, attn_params["img_size"], attn_params["dim"])
    model = VisionTransformer(attn_fn=attn, **attn_params)

    if not os.path.exists(pretrained_path):
        print(
            f"ERROR: Pretrained model not found at {pretrained_path}. Please run pretrain.py first."
        )
        yprov4ml.end_run()
        return
    model.load_state_dict(
        torch.load(pretrained_path, map_location=config["device"]), strict=False
    )

    model.output_proj = nn.Sequential(
        nn.Flatten(),
        nn.ReLU(),
        nn.Linear(8192, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.Sigmoid(),
    )
    model.to(config["device"])
    optimizer = optim.Adam(model.parameters(), **config["optimizer_config"]["params"])
    criterion = nn.MSELoss(**config["criterion_config"].get("params", {}))

    # 3. Finalize Config and Log to Provenance
    experiment_model_dir = os.path.join(
        config["model_save_dir"], config["experiment_name"]
    )
    os.makedirs(experiment_model_dir, exist_ok=True)
    config["artifacts"] = {
        "initial_weights": os.path.join(experiment_model_dir, "initial_weights.pt"),
        "checkpoint_dir": os.path.join(experiment_model_dir, "checkpoints"),
        "final_model": os.path.join(experiment_model_dir, f"{name}_finetuned_final.pt"),
    }
    os.makedirs(config["artifacts"]["checkpoint_dir"], exist_ok=True)

    flat_config = flatten_dict(config)
    for key, value in flat_config.items():
        if isinstance(value, (str, int, float, bool, list, type(None))):
            yprov4ml.log_param(key, value)

    # 4. Save Initial State
    torch.save(model.state_dict(), config["artifacts"]["initial_weights"])
    yprov4ml.log_artifact(
        config["artifacts"]["initial_weights"], "initial_finetune_weights"
    )

    # 5. Run Training Loop with Intra-Epoch Checkpointing for Verification
    checkpoints_per_epoch = config["training_config"].get("checkpoints_per_epoch", 1)
    num_batches = len(trainloader)
    checkpoint_interval = max(1, num_batches // checkpoints_per_epoch)

    # Finetuning
    losses = []
    for epoch in range(config["training_config"]["epochs"]):
        model.train()
        accumulated_indices, segment_tensors = [], []
        checkpoint_step = 1

        for batch_idx, (inputs, labels, indices) in enumerate(
            tqdm(trainloader, desc=f"Finetuning Epoch {epoch+1}")
        ):
            # Accumulate data for the current training segment
            accumulated_indices.extend(indices.tolist())
            segment_tensors.append(inputs.cpu())
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])

            start = time.time()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, F.one_hot(labels, 10).float())
            loss.backward()
            optimizer.step()

            # --- Intra-Epoch Checkpointing Logic ---
            if (batch_idx + 1) % checkpoint_interval == 0:
                # Save model checkpoint
                cp_path = os.path.join(
                    config["artifacts"]["checkpoint_dir"],
                    f"checkpoint_epoch_{epoch+1}_step_{checkpoint_step}.pt",
                )
                torch.save({"model_state_dict": model.state_dict()}, cp_path)
                yprov4ml.log_artifact(
                    cp_path, f"checkpoint_epoch_{epoch+1}_step_{checkpoint_step}"
                )

                # Calculate and log hash of the model checkpoint itself
                model_hash = calculate_sha256(cp_path)
                yprov4ml.log_param(
                    f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.model_hash",
                    model_hash,
                )

                # Calculate and log hash of the data segment used to reach this checkpoint
                full_segment_tensor = torch.cat(segment_tensors, dim=0)
                data_hash = hashlib.sha256(
                    full_segment_tensor.numpy().tobytes()
                ).hexdigest()
                yprov4ml.log_param(
                    f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.data_hash",
                    data_hash,
                )

                # Log the exact indices of the data used in this segment
                yprov4ml.log_param(
                    f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.indices",
                    json.dumps(accumulated_indices),
                )

                # Reset for the next segment
                checkpoint_step += 1
                accumulated_indices = []
                segment_tensors = []

            losses.append(loss.item())
            yprov4ml.log_metric(
                "Time_Finetuning",
                time.time() - start,
                yprov4ml.Context.TRAINING,
                step=epoch,
            )
            yprov4ml.log_metric(
                "Loss_Finetuning", loss.item(), yprov4ml.Context.TRAINING, step=epoch
            )
            # yprov4ml.log_system_metrics(yprov4ml.Context.VALIDATION, step=epoch)

        if accumulated_indices:  # Only save if there were any remainder batches
            final_cp_path = os.path.join(
                config["artifacts"]["checkpoint_dir"],
                f"checkpoint_epoch_{epoch+1}_step_{checkpoint_step}.pt",
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                final_cp_path,
            )
            yprov4ml.log_artifact(
                final_cp_path, f"checkpoint_epoch_{epoch+1}_step_{checkpoint_step}"
            )

            # Calculate and log hash of the final model checkpoint for the epoch
            final_model_hash = calculate_sha256(final_cp_path)
            yprov4ml.log_param(
                f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.model_hash",
                final_model_hash,
            )

            final_data_hash = hashlib.sha256(
                torch.cat(segment_tensors, dim=0).numpy().tobytes()
            ).hexdigest()
            yprov4ml.log_param(
                f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.data_hash",
                final_data_hash,
            )
            yprov4ml.log_param(
                f"checkpoint.epoch_{epoch+1}_step_{checkpoint_step}.indices",
                json.dumps(accumulated_indices),
            )
            print(f"Saved final segment checkpoint for epoch {epoch+1}")

        # yprov4ml.log_carbon_metrics(yprov4ml.Context.VALIDATION, step=epoch)

        plt.plot(losses)
        plt.savefig("finetuning.png")
        plt.close()
        validate_finetune(model, testloader, criterion, epoch)

    for param in model.parameters():
        param.requires_grad = True
    visualize(model, testloader, name)

    # yprov4ml.log_model(model, "final.pt")
    final_model_path = config["artifacts"]["final_model"]
    torch.save(model.state_dict(), final_model_path)
    yprov4ml.log_artifact(final_model_path, "final_pretrained_model_weights")
    yprov4ml.end_run()
    print(f"--- Finished Pretraining for {name} ---")

    # BC upload PID, prov file, prov file hash


def validate_pretrain(model, testloader, criterion, epoch):
    model.eval()
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, _ = data
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            yprov4ml.log_metric(
                "Loss_Pretrain", loss.item(), yprov4ml.Context.VALIDATION, step=epoch
            )
    model.train()


def validate_finetune(model, testloader, criterion, epoch):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, F.one_hot(labels, 10).float())

            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Log loss
            yprov4ml.log_metric(
                "Loss_Finetuning", loss.item(), yprov4ml.Context.VALIDATION, step=epoch
            )

    # Compute and log accuracy
    accuracy = correct / total
    yprov4ml.log_metric(
        "Accuracy_Finetuning", accuracy, yprov4ml.Context.VALIDATION, step=epoch
    )
    model.train()


def visualize(model, testloader, name):
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm2])

    for i, data in enumerate(tqdm(testloader)):
        inputs, labels = data
        inputs = inputs.to(DEVICE)

        # Compute Grad-CAM
        grayscale_cam = cam(input_tensor=inputs)
        grayscale_cam = grayscale_cam[0, :]  # First image only

        # Prepare input image for display
        input_img = inputs[0].detach().cpu().numpy()
        input_img = np.transpose(input_img, (1, 2, 0))
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

        # Generate visualizations
        visualization = show_cam_on_image(input_img, grayscale_cam, use_rgb=True)

        # Create one figure with 3 subplots (| mask | original | overlap |)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(grayscale_cam, cmap="jet")
        axes[0].set_title("Mask")
        axes[0].axis("off")

        axes[1].imshow(input_img)
        axes[1].set_title("Original")
        axes[1].axis("off")

        axes[2].imshow(visualization)
        axes[2].set_title("Overlap")
        axes[2].axis("off")

        # Save combined image

        os.makedirs("./imgs", exist_ok=True)

        plt.tight_layout()
        plt.savefig(f"./imgs/{name}_row_{i}.png")
        plt.close()

        if i >= 15:
            break


if __name__ == "__main__":
    for name in attn_classes:
        print(name)
        benchmark_acc(name, attn_classes[name], BASE_CONFIG_P, BASE_CONFIG_F)
