import torch
from torch import nn
import time
import argparse
import tracemalloc

import sys
sys.path.append("/home/mario.rossi/AttentionBench/yProvML")
import prov4ml
from model import VisionTransformer
from utils import create_attn_module, attn_classes

IMG_SIZES = [1024, 1280, 1536, 1792, 2048, 2304, 2560]
BATCH_SIZE = 1
CHANNELSS = [3]
# CHANNELSS = [64, 128, 512, 2048, 8096]
DEVICE = "cuda"
torch.set_default_device(DEVICE)

def benchmark_attention(attn_classes, IMG_SIZE, CHANNELS, trials=1):
    results = {}
    dummy_input = torch.randn(BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE).to(DEVICE)
    print(dummy_input.shape)

    for name, AttnCls in attn_classes.items():
        print(name)
        # if AttnCls in [DoubleAttention, CrissCrossAttention] and IMG_SIZE == 512: continue

        prov4ml.start_run(
            prov_user_namespace="www.example.org",
            experiment_name=f"{name}_{BATCH_SIZE}_{CHANNELS}_{IMG_SIZE}", 
            provenance_save_dir="prov",
            save_after_n_logs=100,
        )

        attn = create_attn_module(AttnCls, IMG_SIZE, CHANNELS)
        model = VisionTransformer(dim=32, img_size=IMG_SIZE, depth=2, attn_fn=attn, channels=CHANNELS).to(DEVICE)
        model.eval()
        try: 
            tracemalloc.start()

            times = []
            for _ in range(trials):
                start = time.time()
                with torch.no_grad():
                    _ = model(dummy_input)
                times.append(time.time() - start)
                prov4ml.log_metric("Time", times[-1], prov4ml.Context.TRAINING, step=0)
            prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING, step=0)
            prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=0)
            avg_time = sum(times) / trials
            results[name] = avg_time
            prov4ml.log_metric("Time_avg", avg_time, prov4ml.Context.VALIDATION, step=0)
            # print(f"{name:20s}: {avg_time}")
            prov4ml.log_metric("Mem_avg", tracemalloc.get_traced_memory(), prov4ml.Context.VALIDATION, step=0)

            tracemalloc.stop()
        except:
            print("Error")
            pass 
        prov4ml.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True, help="Image size (must be one of [32, 64, 128, 256, 512])")
    args = parser.parse_args()

    # if args.size not in IMG_SIZES:
    #     print(f"Invalid image size {args.size}. Must be one of {IMG_SIZES}.")
    #     exit()

    for s in IMG_SIZES: 
        for c in CHANNELSS: 
            benchmark_attention(attn_classes, s, c)
