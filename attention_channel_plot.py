
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns

from yprov4ml_getters import get_metric, get_param, get_metrics

path = "./channels_data/"

files = [os.path.join(dp, f) for dp, _, fs in os.walk(path) for f in fs if ".json" in f]

# m = #, "disk_usage_Context.TRAINING", "gpu_memory_usage_Context.TRAINING", "ram_energy_Context.TRAINING", "energy_consumed_Context.TRAINING", "ram_power_Context.TRAINING", "Time_Context.VALIDATION"]
metrics_data = []
for file in files: 
    data = json.load(open(file))
    print(get_metrics(data))
    attn = file.split("/")[-1].split("_")[1]
    size = int(file.split("/")[-1].split("_")[3])
    metric = get_param(data, "execution_end_time") - get_param(data, "execution_start_time")
    metrics_data.append({"value": metric, "size": size, "attn": attn})

metrics_data = pd.DataFrame(metrics_data)
metrics_data = metrics_data.pivot(index="size", columns="attn", values="value")
print(metrics_data)
metrics_data = metrics_data[metrics_data.index > 8]

sns.set(style="darkgrid")
sns.set_theme(rc={'figure.figsize':(11.7,8.27)})

metrics_data.plot()

plt.xscale("log")
plt.xlabel("Number of channels")
plt.ylabel("Time (s)")
plt.tight_layout()
plt.savefig("./imgs/channels_scaling.png", dpi=500)
# plt.show()