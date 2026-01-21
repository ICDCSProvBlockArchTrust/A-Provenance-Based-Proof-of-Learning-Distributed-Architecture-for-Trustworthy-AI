import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
import random


def set_parameters(net, parameters, device):
    # load weights from a list of numpy arrays to a torch model
    for i, (name, param) in enumerate(net.named_parameters()):
        param.data = torch.Tensor(parameters[i]).to(device)
    return net


def create_sequences(batch_size, dataset_size, epochs):
    # create a sequence of data indices used for training
    sequence = np.concatenate([np.random.default_rng().choice(dataset_size, size=dataset_size, replace=False)
                               for i in range(epochs)])
    num_batch = int(len(sequence) // batch_size)
    return np.reshape(sequence[:num_batch * batch_size], [num_batch, batch_size])


def consistent_type(model, architecture=None,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), half=False):
    # this function takes in directory to where model is saved, model weights as a list of numpy array,
    # or a torch model and outputs model weights as a list of numpy array
    if isinstance(model, str):
        assert architecture is not None
        state = torch.load(model)
        net = architecture()
        net.load_state_dict(state['net'])
        weights = get_parameters(net)
    elif isinstance(model, np.ndarray):
        weights = torch.tensor(model)
    elif not isinstance(model, torch.Tensor):
        weights = get_parameters(model)
    else:
        weights = model
    if half:
        weights = weights.half()
    return weights.to(device)


def get_parameters(net, numpy=False):
    # get weights from a torch model as a list of numpy arrays
    parameter = torch.cat([i.data.reshape([-1]) for i in list(net.parameters())])
    if numpy:
        return parameter.cpu().numpy()
    else:
        return parameter

def parameter_distance(model1, model2, order=2, architecture=None, half=False):
    # compute the difference between 2 checkpoints
    weights1 = consistent_type(model1, architecture, half=half)
    weights2 = consistent_type(model2, architecture, half=half)
    if not isinstance(order, list):
        orders = [order]
    else:
        orders = order
    res_list = []
    for o in orders:
        if o == 'inf':
            o = np.inf
        if o == 'cos' or o == 'cosine':
            res = (1 - torch.dot(weights1, weights2) /
                   (torch.norm(weights1) * torch.norm(weights1))).cpu().numpy()
        else:
            if o != np.inf:
                try:
                    o = int(o)
                except:
                    raise TypeError("input metric for distance is not understandable")
            res = torch.norm(weights1 - weights2, p=o).cpu().numpy()
        if isinstance(res, np.ndarray):
            res = float(res)
        res_list.append(res)
    return res_list
