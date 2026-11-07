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
    make_figs.py      regenerates Figures 4 and 6 from data/
    requirements.txt  numpy, scipy, matplotlib
training/
    behavior_clone/   model code: six attention variants, CNN backbone,
                      five action heads, training entry point
    common_utils/     shared helpers (seeding, counters, parser, optimiser)
    run.sh            one-line reproduction wrapper
    ENVIRONMENT.md    dependencies, data provenance, output-to-table map
    THIRD_PARTY_NOTICE.md  upstream attribution and licence
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
value reported in the manuscript. Retraining is not required for any of these
checks.

## Training code

`training/` contains the anonymised source used to produce the runs whose
traces are in `data/`. It covers all six attention variants (baseline,
softmax, ELU linear attention, windowed, Performer, Nystromformer), the
shared CNN backbone and five action heads, and a single-command training
entry point. See `training/ENVIRONMENT.md` for the exact command, the
hyperparameters, the output CSVs and the mapping to the manuscript tables.
The `dict.pt` instruction vocabulary is **not** supplied by the upstream
download: it is a pickled `InstructionDict` generated from the instruction
corpus by `python scripts/behavior_clone/inst_dict.py` (see
`training/ENVIRONMENT.md` Sec. 2 for the note about the public upstream file
being a class definition only). The second attention position examined in the
manuscript (`L <= 15`) is a fixed-checkpoint entropy analysis, not a separate
training configuration, so it has no counterpart in this code.

Nineteen files in `training/` are derived from the public MiniRTS release
(CC BY-NC 4.0). Attribution, licence link, modifications and added files are
recorded in `training/THIRD_PARTY_NOTICE.md`. No weights, checkpoints, logs
or dataset files are included.

## Data → manuscript cross-reference

| Files | Used in |
|---|---|
| `e2_val_loss_per_seed.csv`, `e2_action_acc_per_seed.csv`, `e2_run_meta.json` | Table II (protocol), Table IV (quality), Table VI (action accuracy), Sec. V.A (metric-definition sensitivity), Fig. 2 (learning curves) |
| `e1_efficiency.csv`, `e3_efficiency.csv` | Table VII (per-sample inference efficiency), Fig. 3 (inference memory vs batch), Fig. 6 (Pareto x-axis), Table IX (training cost) |
| `e3_val_loss_per_seed.csv`, `e3_action_acc_per_seed.csv`, `e3_baseline_comparison.json`, `e3_run_meta.json`, `e3_machine.json`, `e3_epoch_time.csv` | Table IV, Table V (six-group tests), Table VI, Sec. V.A |
| `e6_entropy_results.json`, `e6_entropy_corr_results.json` | Table XII and Sec. V.D (instruction-position entropy and correlation) |
| `n5_entropy_space.json` | Table XII (spatial-position entropy) |
| `e7_lscan.csv` | Table X and Fig. 5 (length scan, fitted exponents, crossovers) |
| `e8_kernel_efficiency.csv`, `e8_run_meta.json` | Table XI (ELU / ReLU / no-offset ablation) |
| `n7_bwd_ratio_calibration.csv`, `n7_train_mem_scan.csv`, `n7_machine.json` | Table III (four-mode calibration), Table VIII and Fig. 4 (training-mode memory) |
| `e3_train_mem_scan.csv` | Table VIII and Fig. 4 (three approximation baselines) |
| `training/` source | Table I (six variants), Table II (protocol), Fig. 1 (architecture); not needed for the numeric checks |

Figure/table numbers above follow the journal version: Fig. 1 architecture,
Fig. 2 learning curves, Fig. 3 inference memory, Fig. 4 training-mode memory,
Fig. 5 length scan, Fig. 6 Pareto; Tables I–XII are in the order given by
their captions in the manuscript.

## Measurement conventions

* Quality is the **per-seed best validation loss over a fixed 10-epoch budget**.
  The epoch-10 final loss is reported separately as a metric-definition
  sensitivity check; the two definitions are never mixed in one table.
* Table XI uses **epoch-10 loss** because the kernel-ablation runs report that
  value; it is not the best-epoch metric of Table IV.
* Efficiency modes: A = `eval()` + `no_grad()` forward; B = `train()` forward;
  C = forward + backward; D = forward + backward + `optimizer.step()`.
  Table III reports all four; Table VII uses mode A and Table VIII uses
  mode D, as stated in their captions.
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

### Code-version metadata

The MD5 prefixes recorded in some `*_run_meta.json` files (`c7b7488a`,
`917e5f84`, `0acc1827`) are **intermediate snapshots from different
experimental stages**, not checksums of the final implementation. The code
evolved between the E3 and E8 runs, when the Nystrom diagonal-loading repair
and the `--linear_kernel` switch were added, so those values changed from
stage to stage and must **not** be used to judge version consistency. The
metadata JSON files are kept verbatim and no checksum has been edited; see
`training/ENVIRONMENT.md` Sec. 7 for the full account of the three
release-only edits and of the attention code that was left untouched.

## Note on generated images

`make_figs.py` writes PNGs to `figures/`. These are derived outputs and are not
part of the committed traces. Binary files are not text-redacted by anonymisation
services, so any binary added to this repository should be checked manually for
identifying content before the review link is shared.
