#!/usr/bin/env python3
"""Regenerate Figures 4 and 6 of the manuscript from the raw traces.

Fig. 4  Training-mode peak memory vs batch size, all six configurations.
Fig. 6  Quality--efficiency Pareto view (best validation loss vs per-sample
        inference latency at batch 128).

The output filenames (fig5_train_mem_scan.png / fig6_pareto.png) keep the
historical naming used during figure preparation; they do NOT indicate the
figure numbers in the journal version. Length scaling is Fig. 5 there.

Run from the repository root:
    pip install -r code/requirements.txt
    python code/make_figs.py

Outputs are written to ./figures/. No value is hard-coded: both figures are
derived from the CSV files in ./data/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

COL = {"Baseline": "#7F8C8D", "Softmax": "#C0392B", "Linear-ELU": "#1D9E75",
       "Windowed": "#2F5FA8", "Performer": "#B07D2B", "Nystrom": "#8E44AD"}
MARK = {"Baseline": "o", "Softmax": "s", "Linear-ELU": "^",
        "Windowed": "D", "Performer": "v", "Nystrom": "P"}
ORDER = ("Baseline", "Softmax", "Linear-ELU", "Windowed", "Performer", "Nystrom")
NAME = {"none": "Baseline", "softmax": "Softmax", "linear": "Linear-ELU",
        "windowed": "Windowed", "performer": "Performer", "nystrom": "Nystrom"}


def rows(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def best_loss(fname, atts):
    curves = {}
    for r in rows(fname):
        if r["attention_type"] in atts:
            curves.setdefault((r["attention_type"], int(r["seed"])), []).append(float(r["val_loss"]))
    out = {}
    for a in atts:
        v = np.array([min(c) for k, c in curves.items() if k[0] == a])
        out[NAME[a]] = (v.mean(), v.std(ddof=1), len(v))
    return out


def latency_at_128(fname, atts):
    out = {}
    for r in rows(fname):
        if r["batch_size"] == "128" and r["status"] == "ok" and r["attention_type"] in atts:
            out[NAME[r["attention_type"]]] = float(r["per_sample_ms"])
    return out


# ------------------------------------------------ Journal Fig. 4 (file: fig5_*)
series, oom = {}, []
for r in rows("n7_train_mem_scan.csv") + rows("e3_train_mem_scan.csv"):
    lab, b = NAME[r["attention_type"]], int(r["batch_size"])
    if r["status"] == "oom":
        oom.append((lab, b))
        continue
    series.setdefault(lab, []).append((b, float(r["peak_mem_gb"])))

fig, ax = plt.subplots(figsize=(3.55, 2.55))
for lab in ORDER:
    pts = sorted(series[lab])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARK[lab], ms=3.2,
            lw=1.2, color=COL[lab], label=lab)
ax.axhline(24.0, color="#E24B4A", ls="--", lw=1.2)
for lab, b in oom:
    ax.plot(b, 24.0, marker="x", ms=6, mew=1.7, color=COL[lab], clip_on=False)
ax.set_yscale("log")
ax.set_ylim(0.22, 60)
ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16, 24])
ax.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8", "16", "24"])
ax.set_xlabel("Training batch size $B$", fontsize=7.5)
ax.set_ylabel("Peak memory (GB, log)", fontsize=7.5)
ax.set_xticks([32, 64, 96, 128, 160, 192, 224, 256])
ax.grid(alpha=0.3, which="both")
ax.tick_params(labelsize=6.5)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=5.9, ncol=3, loc="upper center",
           bbox_to_anchor=(0.5, 0.99), columnspacing=0.9, handletextpad=0.4)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(OUT / "fig5_train_mem_scan.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote figures/fig5_train_mem_scan.png")

# ------------------------------------------------------------------ Journal Fig. 6
quality = {}
quality.update(best_loss("e2_val_loss_per_seed.csv", {"none", "softmax", "linear"}))
quality.update(best_loss("e3_val_loss_per_seed.csv", {"windowed", "performer", "nystrom"}))
lat = {}
lat.update(latency_at_128("e1_efficiency.csv", {"none", "softmax", "linear"}))
lat.update(latency_at_128("e3_efficiency.csv", {"windowed", "performer", "nystrom"}))
points = [(lab, lat[lab], quality[lab][0], quality[lab][1], quality[lab][2]) for lab in ORDER]

fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.05))
for ax, zoom in zip(axes, (False, True)):
    for lab, x, y, s, n in points:
        ax.errorbar(x, y, yerr=s, fmt=MARK[lab], ms=4.8, color=COL[lab],
                    ecolor=COL[lab], elinewidth=0.9, capsize=2.0)
        if zoom or lab in ("Baseline", "Softmax", "Nystrom"):
            ax.annotate(lab, (x, y), textcoords="offset points",
                        xytext=(4 if lab != "Softmax" else -4, 4 if lab != "Nystrom" else -9),
                        ha="left" if lab != "Softmax" else "right", fontsize=6.2, color=COL[lab])
    ax.set_xlim(0.055, 0.365)
    if zoom:
        ax.set_ylim(2.870, 2.935)
        ax.set_title("(b) Zoom: five equivalent variants", fontsize=8)
    else:
        ax.set_ylim(2.86, 3.25)
        ax.set_title("(a) All six configurations", fontsize=8)
    ax.set_ylabel("Best val. loss", fontsize=8)
    ax.set_xlabel("Per-sample latency (ms), $B{=}128$", fontsize=8, labelpad=1)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)
fig.legend(ORDER, frameon=False, fontsize=6.5, ncol=3,
           loc="lower center", bbox_to_anchor=(0.5, -0.015))
fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=1.4)
fig.savefig(OUT / "fig6_pareto.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("wrote figures/fig6_pareto.png")
