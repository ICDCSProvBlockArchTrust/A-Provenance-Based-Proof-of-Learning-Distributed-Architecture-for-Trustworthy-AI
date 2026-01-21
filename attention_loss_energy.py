import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

from prov_getters import get_metric

path = "./accuracy_data/"

INCLUDED = [
    "CrissCrossAttention", 
    "DoubleAttention", 
    "ImageLinearAttention", 
    "CoordAtt", 
    "MobileViTv2Attention", 
    "SEAttention", 
    "SKAttention", 
    "SimAM", 
]

files = [
    os.path.join(dp, f)
    for dp, _, fs in os.walk(path)
    for f in fs if f.endswith(".json")
]
files = [f for f in files if any(i in f for i in INCLUDED)]

sns.set_style("darkgrid")
sns.set(font_scale=1.3)

# --- create figure with two axes ---
fig, (ax_left, ax_right) = plt.subplots(
    1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.5, 1]}
)

is_pretraining = True
ctx = "Pretraining" if is_pretraining else "Finetuning"

samples = {}
for file in files:
    data = json.load(open(file))

    val_metric_df = (
        get_metric(data, f"Loss_{ctx}_Context.TRAINING")
        .groupby("epoch")
        .mean()
    )

    energy_metric_df = (
        get_metric(data, f"gpu_power_Context.VALIDATION")
        .groupby("epoch")
        .sum()
    )

    val_times = val_metric_df["time"]
    val_times = pd.Series(val_times - val_times.min()).diff()

    energy_metric_df = energy_metric_df["value"] / val_times
    el = val_metric_df["value"].diff().abs() / energy_metric_df

    base_label = file.split("/")[-1].split("_")[1]
    samples[base_label] = el.values

# ---------- LEFT: heatmap with log scale ----------
samples = pd.DataFrame(samples).dropna().T

sns.heatmap(
    samples,
    ax=ax_left,
    cmap="crest",
    norm=LogNorm(), 
    cbar_kws={"label": "|ΔLoss| / Energy"}, 
    annot=True, 
    annot_kws={"rotation":60, "size":12}, 
    fmt='.2f'
)

ax_left.set_title("Loss Difference / GPU Energy Consumption")
ax_left.set_xlabel("Epoch")
# ax_left.set_ylabel("Attention Module")
ax_left.set_yticklabels(ax_left.get_yticklabels(),rotation=30, va="top")

# ---------- RIGHT: loss plot ----------
markers = ["o", "s", "^", "D", "v", "P", "X", "*", "+", "x", "1", "2"]

for i, file in enumerate(files):
    data = json.load(open(file))
    base_label = file.split("/")[-1].split("_")[1]

    loss_df = (
        get_metric(data, f"Loss_{ctx}_Context.TRAINING")
        .groupby("epoch")
        .mean()
    )

    ax_right.plot(loss_df.index, loss_df["value"], linewidth=2, label=base_label, marker=markers[i % len(markers)],markersize=6,markevery=1)

# ax_right.set_yscale("log")
ax_right.set_title(f"{ctx} Loss (MSE)")
ax_right.set_xlabel("Epoch")
# ax_right.set_ylim(0.0, 0.02)
ax_right.set_ylabel("Loss (MSE)")
ax_right.legend()


plt.tight_layout()
plt.savefig("./imgs/attention_loss_energy.pdf")
plt.show()
