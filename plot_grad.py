import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def compute_distances(pols1):
    distances = []
    for i in range(len(pols1)):
        p1 = np.load(pols1[i])
        dist = np.linalg.norm(p1, 1)
        distances.append(dist)
    return distances

def find_pol(path):
    path = os.path.join(path, "artifacts_GR0")
    pol_items = [p for p in os.listdir(path) if "_pol_" in p]
    if not pol_items:
        raise FileNotFoundError(f"No POL files found in {path}")
    pol_path = os.path.join(path, pol_items[0])

    if pol_path.endswith(".csv"):
        return pol_path
    else:
        # Directory con più file .npy
        return [os.path.join(pol_path, f)
                for f in os.listdir(pol_path)[:120]
                if os.path.isfile(os.path.join(pol_path, f)) and f.endswith(".npy")]


import re
def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]
def sort_nicely(l):
    l.sort(key=alphanum_key)
    return l

def main(): 
    dists = []
    for NUM in tqdm(range(20)): 
        path = f"prov/pol_var_{NUM}"
        pol = sort_nicely(find_pol(path))
        dists.append(compute_distances(pol))

    dists2 = []
    for NUM in tqdm(range(20, 23)): 
        path = f"prov/pol_var_{NUM}"
        pol = sort_nicely(find_pol(path))
        dists2.append(compute_distances(pol)[:-1])

    dists = np.array(dists)

    # Compute mean and std across runs, for each timestep:
    mean_dist = np.mean(dists, axis=0)
    std_dist = np.std(dists, axis=0) * 3

    plt.figure(figsize=(15, 9))
    font = {'size'   : 15}

    import matplotlib
    matplotlib.rc('font', **font)

    for d in dists:
        plt.plot(d, color='gray', alpha=0.8)

    i = 0
    for d in dists2: 
        plt.plot(d, color='red', alpha=0.8)
        i += 1

    # Plot mean and thresholds
    plt.plot(mean_dist, color='blue', label='Mean')
    plt.plot(mean_dist + std_dist, color='black', linestyle='--', label='+3σ')
    plt.plot(mean_dist - std_dist, color='black', linestyle='--', label='-3σ')

    # (Optional) Fill the std area
    plt.fill_between(range(len(mean_dist)), 
                    mean_dist - std_dist, 
                    mean_dist + std_dist, 
                    color='orange', alpha=0.2, label='±3σ region')
    
    plt.text(40, 35700, 'Modified checkpoint', horizontalalignment='center', verticalalignment='center')
    plt.text(25, 32500, 'Run with wrong parameter', horizontalalignment='center', verticalalignment='center')

    plt.legend()
    plt.title("Distances with ±3σ threshold")
    plt.xlabel("Timesteps")
    plt.ylabel("Norm distance from 0")
    plt.savefig("grad_delta.pdf")

if __name__ == "__main__": 
    main()