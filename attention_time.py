
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import json
import pandas as pd

from prov_getters import get_metric

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

is_pretraining = False
ctx = "Pretraining" if is_pretraining else "Finetuning"

min_time = []
for file in files:
    data = json.load(open(file))
    val_metric_df = get_metric(data, f"Time_{ctx}_Context.TRAINING").groupby("epoch").mean()
    val_values = val_metric_df["value"]
    val_times = val_metric_df["time"]
    val_times = [t - min(val_times) for t in val_times]
    min_time.append({"file": file, "min": val_values.min(), "time": max(val_times)})

    base_label = file.split("/")[-1].split("_")[1]
    plt.plot(range(len(val_times)), val_values, linestyle="-", label=f"{base_label}")

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Time per epoch (s)")
plt.title("Time taken for each of the 25 epochs of " + ("pretraining" if is_pretraining else "finetuning"))
plt.tight_layout()
plt.savefig("./imgs/time_pretraining.png" if is_pretraining else "./imgs/time_finetuning.png", dpi=500)
plt.show()