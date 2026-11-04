# Artefact bundle — "Where Does Attention Pay Off?"

This repository contains the **raw experimental traces** and the **reproduction
scripts** supporting the manuscript submitted to *IEEE Transactions on Games*.
It is provided for double-anonymous peer review only. It contains no author
names, affiliations, e-mail addresses, funding information, or links that
identify the authors. The canonical repository will replace it after acceptance.

## Layout

```
README.md
data/    21 verbatim CSV / JSON traces (no value has been edited)
code/
    verify_stats.py   re-derives every numerical claim from data/
    make_figs.py      regenerates Figures 5 and 6 from data/
    requirements.txt  numpy, scipy, matplotlib
```

## Reproducing the numbers

```bash
pip install -r code/requirements.txt
python code/verify_stats.py     # prints recomputed vs manuscript values
python code/make_figs.py        # writes figures/fig5_*.png, fig6_*.png
```

`verify_stats.py` recomputes, among others: per-seed best/final validation
losses, Welch tests, one-way and seed-blocked ANOVA, Friedman's test, TOST
equivalence p-values at three margins, `cmd_type` accuracy at the best-loss
epoch, pooled entropy–accuracy correlation, log–log length-scan exponents and
R², four-mode timing ratios (`D/A`, `C/B`), the softmax out-of-memory boundary,
and the kernel ablation. Each check prints the recomputed value next to the
value reported in the manuscript.

## Data → manuscript cross-reference

| Files | Used in |
|---|---|
| `e2_val_loss_per_seed.csv`, `e2_action_acc_per_seed.csv`, `e2_run_meta.json` | Table II (protocol), Table IV (quality), Table VI (action accuracy), Sec. V.A (metric-definition sensitivity), Table V (inference) |
| `e1_efficiency.csv`, `e3_efficiency.csv` | Table VI / Fig. 3 (inference efficiency), Fig. 6 (Pareto x-axis) |
| `e3_val_loss_per_seed.csv`, `e3_action_acc_per_seed.csv`, `e3_baseline_comparison.json`, `e3_run_meta.json`, `e3_machine.json`, `e3_epoch_time.csv` | Table IV, Table V (six-group tests), Sec. V.A |
| `e6_entropy_results.json`, `e6_entropy_corr_results.json` | Table XII and Sec. V.D (instruction-position entropy and correlation) |
| `n5_entropy_space.json` | Table XII (spatial-position entropy) |
| `e7_lscan.csv` | Table X and Fig. 4 (length scan, fitted exponents, crossovers) |
| `e8_kernel_efficiency.csv`, `e8_run_meta.json` | Table XI (ELU / ReLU / no-offset ablation) |
| `n7_bwd_ratio_calibration.csv`, `n7_train_mem_scan.csv`, `n7_machine.json` | Table III (four-mode calibration), Table VIII and Fig. 5 (training-mode memory) |
| `e3_train_mem_scan.csv` | Table VIII and Fig. 5 (three approximation baselines) |

## Measurement conventions

* Quality is the **per-seed best validation loss over a fixed 10-epoch budget**.
  The epoch-10 final loss is reported separately as a metric-definition
  sensitivity check; the two definitions are never mixed in one table.
* Table XI uses **epoch-10 loss** because the kernel-ablation runs report that
  value; it is not the best-epoch metric of Table IV.
* Efficiency modes: A = `eval()` + `no_grad()` forward; B = `train()` forward;
  C = forward + backward; D = forward + backward + `optimizer.step()`.
  Table III reports all four; Tables VI/VIII use the mode stated in their captions.
* Length-scan exponents are fitted only on the asymptotic intervals
  (`L = 529..2025` for softmax, `L = 1024..4096` for linear); fitting all
  points including the small-`L` plateau would bias the slope.
* The memory crossover is a deterministic point: all repeats return the same
  `max_memory_allocated()`. No bootstrap interval is reported for it.
* The instruction-position pooled correlation is `r = 0.105`. The delivered
  trace stores `p = 0.1367` (consistent with a large-sample normal
  approximation, reported as `0.137` in the manuscript), whereas an exact
  t-distribution `scipy.stats.pearsonr` gives `p = 0.1383`. Both are
  non-significant and lead to the same conclusion. `verify_stats.py` prints
  both values rather than silently overriding the delivered statistic.

## Exclusions

An earlier CPU-only pilot (torch 1.4, batch 1, end-to-end timing) and three
superseded per-task loss CSVs are **not** included: they were retracted during
the internal audit because of a different hardware/software stack, non-monotonic
points and an incorrect memory-accounting denominator. The manuscript reports
only the traces in `data/`.

## Environment

fp32, one NVIDIA GeForce RTX 4090 (24 GB), PyTorch 2.8.0+cu128, CUDA 12.8,
driver 595.71.05, `cudnn.benchmark=True`, TF32 enabled. Full hyperparameters,
seed policy and timing protocol are given in Table II of the manuscript and in
the `*_run_meta.json` files. Run-metadata files retain the original operator
notes verbatim; they contain no identifying information.

## Note on generated images

`make_figs.py` writes PNGs to `figures/`. These are derived outputs and are not
part of the committed traces. Binary files are not text-redacted by anonymisation
services, so any binary added to this repository should be checked manually for
identifying content before the review link is shared.
