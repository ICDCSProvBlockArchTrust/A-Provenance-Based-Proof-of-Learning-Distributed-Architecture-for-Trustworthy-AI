
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

from yprov4ml_getters import get_metric

path = "./context_len_data/"
SIZES = [512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560]
files = [os.path.join(dp, f) for dp, _, fs in os.walk(path) for f in fs if ".json" in f] #if ATTN_TYPE in f

M = "Mem_avg_Context.VALIDATION"

res = []
for f in files: 
    data = json.load(open(f))

    attn = f.split("/")[-1].split("_")[1]
    s = int(f.split("/")[-1].split("_")[-2])
    i = SIZES.index(s)

    try: 
        m = get_metric(data, M)["value"]
        met = eval(m.iloc[0])[0] / 1000
        res.append({"attn": attn, "size": s, "time": met})
    except: 
        res.append({"attn": attn, "size": s, "time": np.nan})

res = pd.DataFrame(res)
res = res.pivot(index="size", columns="attn", values="time")
res.plot(figsize=(12, 9))
plt.xlabel("Image Size")
plt.ylabel("Memory Usage (MB)")
plt.xscale("log")
plt.legend(title="")
plt.savefig(f"./imgs/scaling_memory.png", dpi=500)
plt.show()