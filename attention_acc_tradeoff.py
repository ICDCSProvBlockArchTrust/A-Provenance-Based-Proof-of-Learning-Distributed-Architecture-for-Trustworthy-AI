
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

from yprov4ml_getters import get_metric

path = "./accuracy_data/"

INCLUDED = [
    "CrissCrossAttention", 
    "DoubleAttention", 
    "TripletAttention", 
    "ImageLinearAttention", 
    "CoordAtt", 
    "MobileViTv2Attention", 
    "SEAttention", 
    "SKAttention", 
    "SimAM", 
]

files = [os.path.join(dp, f) for dp, _, fs in os.walk(path) for f in fs if ".json" in f]  # and "Double" not in f
files = [f for f in files if any([i in f for i in INCLUDED])]
plt.figure(figsize=(12, 9))

val_metric_name = "cpu_energy_Context.VALIDATION"

min_time = []
for file in files:
    data = json.load(open(file))
    val_metric_df = get_metric(data, val_metric_name).groupby("epoch").mean()
    val_metric_df2 = get_metric(data, "ram_energy_Context.VALIDATION").groupby("epoch").mean()
    val_metric_df3 = get_metric(data, "gpu_energy_Context.VALIDATION").groupby("epoch").mean()
    val_values = val_metric_df["value"] + val_metric_df2["value"] + val_metric_df3["value"]
    val_times = val_metric_df["time"]
    val_times = [t - min(val_times) for t in val_times]
    min_time.append({"file": file, "min": val_values.min(), "time": max(val_times)})

    base_label = file.split("/")[-1].split("_")[1]
    plt.plot(range(len(val_times)), val_values, linestyle="-", label=f"{base_label}")


df = pd.DataFrame(min_time)
df["filename"] = df["file"].apply(lambda x: str(x).split("_")[-3])
df = df.sort_values(by="min")

x = np.arange(len(df))
width = 0.4

fig, ax1 = plt.subplots(figsize=(12, 6))

bar1 = ax1.bar(x - width/2, df["time"], width=width, label="Time (s)", color='tab:blue')
ax1.set_ylabel("Time (s)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(x)
ax1.set_xticklabels(df["filename"], rotation=45, ha='right')

ax2 = ax1.twinx()
bar2 = ax2.bar(x + width/2, df["min"], width=width, label="Final Validation Loss (MSE)", color='tab:orange')
ax2.set_ylabel("Final Validation Loss (MSE)", color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
fig.tight_layout()
fig.savefig("./imgs/tradeoff.png", dpi=500)
