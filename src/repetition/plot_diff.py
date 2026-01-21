
import torch
from tqdm import tqdm
import numpy as np

from utils import *

def compute_distances(pols1):
    distances = []
    for i in range(len(pols1)):
        p1 = np.load(pols1[i])
        dist = np.linalg.norm(p1, 1)
        distances.append(dist)
    return distances

realz = []
fakes = []

DEVICE = "cpu"
for j in tqdm(range(15)): 
    realz.append([])
    for i in range(5): 
        model_real1 = torch.load(f"./weights/real_{j}_{i}.pt", weights_only=False, map_location=torch.device('cpu'))
        # dist = 0
        # for p in model_real1.parameters(): 
            # dist += np.linalg.norm(torch.flatten(p).detach().numpy(), 1)
        ls = get_parameters(model_real1)
        realz[j].append(sum(ls))# / len(ls))#dist)#

for j in tqdm(range(5)): 
    fakes.append([])
    for i in range(5): 
        model_real1 = torch.load(f"./weights/fake_{j}_{i}.pt", weights_only=False, map_location=torch.device('cpu'))
        # dist = 0
        # for p in model_real1.parameters(): 
        #     dist += np.linalg.norm(torch.flatten(p).detach().numpy(), 1)
        ls = get_parameters(model_real1)
        fakes[j].append(sum(ls))# / len(ls)) #dist)#

import matplotlib.pyplot as plt
for i in range(len(realz)): 
    plt.plot(realz[i], label=f"real_{i}", color="green")
for i in range(len(fakes)): 
    plt.plot(fakes[i], label=f"fake_{i}", color="red")
plt.legend()
plt.savefig("tmp.png")