#!/usr/bin/env python3
"""Recompute the numerical claims of the manuscript from the raw traces.

Run from the repository root:
    pip install -r code/requirements.txt
    python code/verify_stats.py

Only numpy and scipy are required. All paths are relative to this file, so
the script also works after downloading the anonymised ZIP archive.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAIN = ("none", "linear", "softmax")

CHECKS: list[bool] = []


def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(name):
    with (DATA / name).open(encoding="utf-8-sig") as fh:
        return json.load(fh)


def check(label, got, expected, tol=5e-4):
    ok = abs(got - expected) <= tol
    CHECKS.append(ok)
    tag = "OK" if ok else "REVIEW"
    print(f"  [{tag:>6}] {label:<54} recomputed={got:>11.5f}  paper={expected:>11.5f}")


def show(label, got):
    print(f"  [ INFO ] {label:<54} recomputed={got:>11.5f}")


def group(dic, att):
    return np.array([dic[k] for k in sorted(dic) if k[0] == att], dtype=float)


def tost_welch(a, b, delta):
    """Welch-type two one-sided tests for |mean(a)-mean(b)| < delta."""
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
    se = math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    diff = a.mean() - b.mean()
    p_low = stats.t.sf((diff + delta) / se, df)
    p_high = stats.t.cdf((diff - delta) / se, df)
    return max(p_low, p_high), df, diff, se


def blocked_anova(arrays):
    """Randomised-block ANOVA with seed as block; returns (F, df1, df2, p)."""
    m = np.vstack(arrays)
    k, n = m.shape
    grand = m.mean()
    ss_t = n * ((m.mean(axis=1) - grand) ** 2).sum()
    resid = m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + grand
    ss_e = (resid**2).sum()
    df1, df2 = k - 1, (k - 1) * (n - 1)
    f = (ss_t / df1) / (ss_e / df2)
    return f, df1, df2, stats.f.sf(f, df1, df2)


# --------------------------------------------------------------------------
print("\n[1] E2 quality: per-seed BEST validation loss (Table IV)")
curves = {}
for r in read_csv("e2_val_loss_per_seed.csv"):
    curves.setdefault((r["attention_type"], int(r["seed"])), {})[int(r["epoch"])] = float(r["val_loss"])
best_epoch = {k: min(v, key=v.get) for k, v in curves.items()}
best = {k: curves[k][best_epoch[k]] for k in curves}
final = {k: curves[k][max(curves[k])] for k in curves}
B = {a: group(best, a) for a in MAIN}
F = {a: group(final, a) for a in MAIN}
for a in MAIN:
    print(f"        {a:<8} best={B[a].mean():.4f} +/- {B[a].std(ddof=1):.4f}  "
          f"final={F[a].mean():.4f} +/- {F[a].std(ddof=1):.4f}  n={len(B[a])}")
check("best loss, none", B["none"].mean(), 2.8832)
check("best loss, linear", B["linear"].mean(), 2.8872)
check("best loss, softmax", B["softmax"].mean(), 2.8851)
check("softmax minus baseline (sign corrected)", B["softmax"].mean() - B["none"].mean(), 0.0019)
check("epoch-10 loss, none (metric sensitivity)", F["none"].mean(), 2.8866)
check("epoch-10 loss, softmax (metric sensitivity)", F["softmax"].mean(), 2.8864)

print("\n[2] Inferential tests on the three main configurations (Table V)")
for a, b in (("none", "linear"), ("none", "softmax"), ("linear", "softmax")):
    t, p = stats.ttest_ind(B[a], B[b], equal_var=False)
    show(f"Welch t p-value, {a} vs {b}", p)
f1, p1 = stats.f_oneway(B["none"], B["linear"], B["softmax"])
check("one-way ANOVA F(2,27)", f1, 0.203, tol=2e-3)
show("one-way ANOVA p", p1)
fb, d1, d2, pb = blocked_anova([B["none"], B["linear"], B["softmax"]])
check("seed-blocked ANOVA F(2,18)", fb, 0.256, tol=2e-3)
show("seed-blocked ANOVA p", pb)
chi, pf = stats.friedmanchisquare(B["none"], B["linear"], B["softmax"])
check("Friedman chi2(2)", chi, 0.200, tol=2e-3)
show("Friedman p", pf)

print("\n[3] Equivalence testing (Sec. IV.C / Table V)")
for delta in (0.025, 0.020, 0.015):
    p, df, diff, se = tost_welch(B["softmax"], B["linear"], delta)
    print(f"        delta={delta:.3f}  TOST p={p:.4f}  df={df:.1f}  diff={diff:+.4f}  se={se:.4f}")
p_eq, _, _, _ = tost_welch(B["softmax"], B["linear"], 0.025)
check("TOST p, softmax vs linear, delta=0.025", p_eq, 0.0042, tol=2e-4)

print("\n[4] E4 action accuracy at the best-loss epoch (Table VI)")
acc = {}
for r in read_csv("e2_action_acc_per_seed.csv"):
    acc[(r["attention_type"], int(r["seed"]), int(r["epoch"]))] = float(r["cmd_type"])
cmd = {}
for a in MAIN:
    cmd[a] = np.array([acc[(a, s, best_epoch[(a, s)])] for s in sorted({k[1] for k in best if k[0] == a})])
show("cmd_type, none", cmd["none"].mean())
show("cmd_type, linear", cmd["linear"].mean())
check("cmd_type, none minus linear", cmd["none"].mean() - cmd["linear"].mean(), 0.0338)
t, p = stats.ttest_rel(cmd["none"], cmd["linear"])
check("paired t, cmd_type none vs linear", t, 3.321, tol=5e-3)
show("paired t p, cmd_type none vs linear", p)

print("\n[5] E6 instruction-position entropy vs accuracy (Sec. V.D)")
e6 = read_json("e6_entropy_corr_results.json")
rows = e6["results"]
xs = ([r["softmax_entropy"] for r in rows] + [r["linear_elu_entropy"] for r in rows])
ys = ([r["softmax_cmd_acc"] for r in rows] + [r["linear_elu_cmd_acc"] for r in rows])
r_pool, p_pool = stats.pearsonr(xs, ys)
check("pooled Pearson r", r_pool, 0.1051789, tol=1e-4)
p_trace = e6["correlation"]["combined"]["pearson_p"]
print(f"        pooled p: exact t-distribution={p_pool:.6f}; delivered trace={p_trace:.6f}")
print("        NOTE: the delivered trace p is consistent with a large-sample normal")
print("              approximation. Both values are non-significant and lead to the")
print("              same conclusion; the manuscript reports the trace value (0.137).")
show("softmax-only r", e6["correlation"]["softmax"]["pearson"])
show("linear-only r", e6["correlation"]["linear_elu"]["pearson"])
eh = read_json("e6_entropy_results.json")
show("softmax entropy mean", eh["softmax"]["mean"])
show("linear entropy mean", eh["linear_elu"]["mean"])

print("\n[6] E7 length scan: log-log exponents and crossover (Table X)")
ls = read_csv("e7_lscan.csv")
pts = {}
for att in ("softmax", "linear"):
    for L in sorted({int(r["L"]) for r in ls if r["attention_type"] == att}):
        sub = [r for r in ls if r["attention_type"] == att and int(r["L"]) == L and r["status"] == "ok"]
        if not sub:
            continue
        pts[(att, L)] = (float(np.median([float(r["fwd_bwd_ms"]) for r in sub])),
                         float(np.median([float(r["peak_mem_gb"]) for r in sub])))

def fit(att, Ls, idx):
    x = np.log(np.array(Ls, dtype=float))
    y = np.log(np.array([pts[(att, L)][idx] for L in Ls]))
    res = stats.linregress(x, y)
    return res.slope, res.rvalue ** 2

p_s, r2_s = fit("softmax", [529, 1024, 2025], 0)
p_l, r2_l = fit("linear", [1024, 2025, 4096], 0)
q_s, rq_s = fit("softmax", [529, 1024, 2025], 1)
q_l, rq_l = fit("linear", [1024, 2025, 4096], 1)
check("softmax latency exponent (L=529..2025)", p_s, 1.8527, tol=5e-3)
check("linear latency exponent (L=1024..4096)", p_l, 0.9026, tol=5e-3)
check("softmax memory exponent (L=529..2025)", q_s, 1.9337, tol=5e-3)
check("linear memory exponent (L=1024..4096)", q_l, 0.9801, tol=5e-3)
print(f"        R2: softmax lat={r2_s:.5f} mem={rq_s:.5f}; linear lat={r2_l:.5f} mem={rq_l:.5f}")

print("\n[7] N7 four-mode timing calibration (Table III)")
cal = read_csv("n7_bwd_ratio_calibration.csv")
mode_mean = {}
for a in MAIN:
    for m in "ABCD":
        v = [float(r["ms_per_iter"]) for r in cal if r["attention_type"] == a and r["mode"] == m]
        if v:
            mode_mean[(a, m)] = float(np.mean(v))
for a, expected in (("none", 4.656), ("softmax", 3.159), ("linear", 4.238)):
    check(f"D/A ratio, {a}", mode_mean[(a, "D")] / mode_mean[(a, "A")], expected, tol=3e-3)
    show(f"C/B ratio, {a}", mode_mean[(a, "C")] / mode_mean[(a, "B")])

print("\n[8] Training-mode memory vs batch (Table VIII / Fig. 5)")
tm = read_csv("n7_train_mem_scan.csv") + read_csv("e3_train_mem_scan.csv")
# Out-of-memory rows carry an empty peak_mem_gb field; keep them for the OOM scan.
soft = []
for r in tm:
    if r["attention_type"] != "softmax":
        continue
    raw = (r.get("peak_mem_gb") or "").strip()
    soft.append((int(r["batch_size"]), r["status"], float(raw) if raw else float("nan")))
soft.sort()
check("softmax peak memory at B=128 (GB)", dict((b, m) for b, s, m in soft if s == "ok")[128], 17.3886, tol=2e-3)
oom = [b for b, s, _ in soft if s == "oom"]
show("softmax first OOM batch", float(min(oom)) if oom else float("nan"))

print("\n[9] E8 kernel ablation (Table XI)")
for r in read_csv("e8_kernel_efficiency.csv"):
    print(f"        {r['kernel']:<6} latency={float(r['fwd_bwd_ms_mean']):.4f} ms  "
          f"memory={float(r['peak_mem_gb']):.4f} GB  L={r['L']}")

print("\n[10] E3 approximation baselines (Table IV)")
e3curves = {}
for r in read_csv("e3_val_loss_per_seed.csv"):
    e3curves.setdefault((r["attention_type"], int(r["seed"])), []).append(float(r["val_loss"]))
for a in ("windowed", "performer", "nystrom"):
    v = np.array([min(c) for k, c in e3curves.items() if k[0] == a])
    print(f"        {a:<10} best={v.mean():.4f} +/- {v.std(ddof=1):.4f}  n={len(v)}")

print("\n" + "=" * 78)
print(f"RESULT: {sum(CHECKS)}/{len(CHECKS)} numeric checks match the manuscript.")
print("See Section [5] for the documented pooled-p convention difference.")
print("=" * 78)
