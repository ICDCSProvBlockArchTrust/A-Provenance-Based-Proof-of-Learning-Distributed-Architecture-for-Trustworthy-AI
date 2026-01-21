import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

from model import VisionTransformer
from utils import create_attn_module, attn_classes

MODEL_REGISTRY = {"VisionTransformer": VisionTransformer}
DATASET_REGISTRY = {"CIFAR10": torchvision.datasets.CIFAR10}
OPTIMIZER_REGISTRY = {"Adam": optim.Adam, "SGD": optim.SGD}
CRITERION_REGISTRY = {"MSELoss": nn.MSELoss, "CrossEntropyLoss": nn.CrossEntropyLoss}
MODULE_REGISTRY = {
    "Sequential": nn.Sequential,
    "Linear": nn.Linear,
    "ReLU": nn.ReLU,
    "Sigmoid": nn.Sigmoid,
    "Flatten": nn.Flatten,
}


def create_module_from_config(layer_config_list):
    layers = []
    for layer_cfg in layer_config_list:
        LayerClass = MODULE_REGISTRY[layer_cfg["class_name"]]
        layers.append(LayerClass(**layer_cfg.get("params", {})))
    return nn.Sequential(*layers)


def create_model(config):
    model_cfg = config["model_config"]
    ModelClass = MODEL_REGISTRY[model_cfg["class_name"]]
    model_params = model_cfg.get("params", {})

    if model_cfg["class_name"] == "VisionTransformer":
        attn_class = attn_classes[config["attention_class_name"]]

        img_size = model_params.get("img_size")
        channels = model_params.get(
            "dim"
        )  # ViT 'dim' is the channel dimension for attention blocks
        attn_fn = create_attn_module(attn_class, img_size, channels)

        model = ModelClass(attn_fn=attn_fn, **model_params)
    else:
        model = ModelClass(**model_params)

    if model_cfg.get("pretrained_model_path") and os.path.exists(
        model_cfg["pretrained_model_path"]
    ):
        model.load_state_dict(
            torch.load(model_cfg["pretrained_model_path"], map_location="cpu"),
            strict=False,
        )
        print(f"Loaded pretrained weights from: {model_cfg['pretrained_model_path']}")

    if "modifications" in model_cfg:
        print("Applying model modifications...")
        for mod in model_cfg["modifications"]:
            if mod["type"] == "replace_attribute":
                new_module = create_module_from_config(
                    mod["new_module_config"]["layers"]
                )
                setattr(model, mod["attribute_path"], new_module)
    return model


def create_dataloader(config, train=True):
    data_cfg = config["data_config"]
    DatasetClass = DATASET_REGISTRY[data_cfg["database"]]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                data_cfg["transform"]["normalize_mean"],
                data_cfg["transform"]["normalize_std"],
            ),
        ]
    )
    dataset = DatasetClass(
        root=data_cfg["root"], train=train, download=True, transform=transform
    )
    g = torch.Generator().manual_seed(config["seed"]) if train else None
    return dataset, torch.utils.data.DataLoader(
        dataset, batch_size=data_cfg["batch_size"], shuffle=train, generator=g
    )


def create_optimizer(model, config):
    optim_cfg = config["optimizer_config"]
    OptimizerClass = OPTIMIZER_REGISTRY[optim_cfg["class_name"]]
    return OptimizerClass(model.parameters(), **optim_cfg["params"])


def create_criterion(config):
    crit_cfg = config["criterion_config"]
    CriterionClass = CRITERION_REGISTRY[crit_cfg["class_name"]]
    return CriterionClass(**crit_cfg.get("params", {}))
