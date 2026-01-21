
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

SRC = "data/"

data2 = [SRC + f for f in os.listdir(SRC) if f != ".DS_Store"]
data = []
for f in data2: 
    data.extend([f + "/" + f1 for f1 in os.listdir(f) if f1 != ".DS_Store"])

def get_speed(start_path = '.'):
    j = [start_path + "/" + f for f in os.listdir(start_path) if f.endswith(".json")][0]
    d = json.load(open(j, "r"))
    start_time = float(d["entity"]["execution_start_time"]["prov-ml:parameter_value"])
    end_time = float(d["entity"]["execution_end_time"]["prov-ml:parameter_value"])
    return end_time-start_time

d = []
for f in data: 
    size = round(get_speed(f), 3)
    exp = f.split("/")[-2]
    file = f.split("/")[-1]
    checks = 0
    if "16" in exp: checks = 16
    elif "32" in exp: checks = 32
    elif "4" in exp: checks = 4
    elif "8" in exp: checks = 8
    elif "1" in exp: checks = 1
    d.append({"file": exp, "attn": file.split("_")[0], "size": size, "modality": "Pretraining" if exp.split("_")[-1] == "pretrain" else "Finetuning", "checks": checks})

sns.set_style("darkgrid")
sns.set(font_scale=1.3)

d = pd.DataFrame(d).sort_values("file")
g = sns.catplot(
    data=d, kind="bar",
    x="modality", y="size", hue="checks", palette="tab10",
    errorbar="sd", alpha=.8, aspect=1.7, height=5, order=["Pretraining", "Finetuning"]
)
g.set_axis_labels("", "Execution time in seconds (Log Scale)")
g.legend.remove()
plt.legend(title="Checkpoints")
plt.yscale("log")
plt.tight_layout()
plt.savefig("time.pdf")
# plt.show()