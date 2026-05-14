"""Benchmark DRNN_jm speed under different optimization conditions.

Measures forward+backward time on a fixed problem size.
Conditions:
  - baseline (no opts)
  - AMP bf16
  - torch.compile (fullgraph=False)
  - AMP + compile
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from model import MODEL_REGISTRY
from dataset import DATASET_REGISTRY
from utils.utils import set_seed


def make_batch(ds, device, batch_size=128):
    inputs = torch.stack([ds[i][0] for i in range(batch_size)]).to(device)
    targets = torch.stack([ds[i][1] for i in range(batch_size)]).to(device)
    return inputs, targets


def time_run(model, inputs, targets, optimizer, n_steps=20, use_amp=False):
    # Warmup
    for _ in range(3):
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out, _ = model(x=inputs, train=True, logs={})
                mask = targets != -1
                loss = F.cross_entropy(out[mask], targets[mask])
        else:
            out, _ = model(x=inputs, train=True, logs={})
            mask = targets != -1
            loss = F.cross_entropy(out[mask], targets[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_steps):
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out, _ = model(x=inputs, train=True, logs={})
                mask = targets != -1
                loss = F.cross_entropy(out[mask], targets[mask])
        else:
            out, _ = model(x=inputs, train=True, logs={})
            mask = targets != -1
            loss = F.cross_entropy(out[mask], targets[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_steps


def build_model(h=64, max_delay=130, dataset_id="delaysequence", input_size=12):
    return MODEL_REGISTRY["DRNN_jm"](
        id="DRNN_jm",
        max_delay=max_delay,
        init_tau=1.0,
        min_tau=0.1,
        input_size=input_size,
        hidden_size=h,
        output_size=10,
        dataset_id=dataset_id,
        device="cuda",
    )


def main():
    set_seed(0)
    device = "cuda"

    # Match capdyn N=30 setting (medium-size benchmark)
    ds = DATASET_REGISTRY["delaysequence"](
        id="delaysequence",
        size=200,
        min_len=30,
        max_len=30,
        k=10,
        delay=30,
    )
    inputs, targets = make_batch(ds, device, batch_size=128)
    print(f"Benchmark setting: max_len=30, delay=30, total_len={ds.total_len}, batch=128, h=64, max_delay=130")

    conditions = [
        ("baseline (no opts)", False, False),
        ("AMP bf16 only", True, False),
        ("torch.compile only", False, True),
        ("AMP + compile", True, True),
    ]

    results = {}
    for label, use_amp, use_compile in conditions:
        print(f"\n=== {label} ===")
        model = build_model(max_delay=130)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if use_compile:
            model = torch.compile(model, fullgraph=False)
        try:
            t = time_run(model, inputs, targets, optimizer, n_steps=10, use_amp=use_amp)
            results[label] = t
            print(f"  time/step: {t*1000:.1f} ms")
        except Exception as e:
            results[label] = None
            print(f"  FAILED: {type(e).__name__}: {str(e)[:200]}")

    print("\n" + "=" * 60)
    print("Summary (lower = faster):")
    baseline = results.get("baseline (no opts)")
    for label, t in results.items():
        if t is None:
            print(f"  {label:<30s}  FAILED")
        elif baseline:
            speedup = baseline / t
            print(f"  {label:<30s}  {t*1000:7.1f} ms  ({speedup:.2f}x)")
        else:
            print(f"  {label:<30s}  {t*1000:7.1f} ms")


if __name__ == "__main__":
    main()
