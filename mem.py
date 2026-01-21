
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

SRC = "data/"

data = [SRC + f for f in os.listdir(SRC) if f != ".DS_Store"]

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

d = []
for f in data: 
    size = round(get_size(f) / (1024 * 1024), 3)
    file = f.split("/")[-1]
    checks = 0
    if "16" in file: checks = 16
    elif "32" in file: 
        checks = 32 
        continue
    elif "4" in file: checks = 4
    elif "8" in file: checks = 8
    elif "1" in file: checks = 1
    d.append({"file": file, "size": size, "modality": "Pretraining" if file.split("_")[-1] == "pretrain" else "Finetuning", "checks": checks})

sns.set_style("darkgrid")
sns.set(font_scale=1.3)

d = pd.DataFrame(d).sort_values("file")
g = sns.catplot(
    data=d, kind="bar",
    x="modality", y="size", hue="checks", palette="tab10",
    errorbar="sd", alpha=.8, aspect=1.7, height=5,  order=["Pretraining", "Finetuning"]
)
g.set_axis_labels("", "Folder size in Mb (Log Scale)")
g.legend.remove()
plt.legend(title="Checkpoints")
plt.yscale("log")
plt.tight_layout()
plt.savefig("mem.pdf")
# plt.show()