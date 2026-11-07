# Environment & Reproduction Guide

This directory contains the anonymised training code for the six attention
variants studied in the manuscript. It is supplied for double-anonymous peer
review only. No checkpoints, weights, logs or dataset files are included.

## 1. Dependencies

| Package | Version | Notes |
|---|---|---|
| Python | 3.12 | 3.9+ should also work |
| PyTorch | 2.8.0+cu128 | CUDA 12.8 build; any recent 2.x CUDA build is fine |
| NumPy | 1.24+ | |
| tensorboardX | any recent | logging only; see the note below |
| GPU | NVIDIA RTX 4090 24 GB | required for full batch-size-128 training |

`tensorboardX` is inherited from the upstream MiniRTS code through
`common_utils/multi_counter.py` and `common_utils/result_stat.py`. In this
bundle both imports are optional: if the package is absent, the counters fall
back to plain logging and training is unaffected. Installing it is therefore
recommended but not required.

```bash
pip install torch numpy tensorboardx   # use the CUDA wheel matching your driver
```

## 2. Dataset

The experiments use the public MiniRTS logged demonstrations released with
Hu et al., *Hierarchical Decision Making by Generating and Following Natural
Language Instructions* (NeurIPS 2019, arXiv:1906.00744), distributed from the
`facebookresearch/minirts` repository:

```
https://github.com/facebookresearch/minirts
```

The upstream release provides 76 thousand human-play instruction/execution
pairs together with processed per-frame JSON files. Following the upstream
data pipeline (`cd data && sh download.sh` in that repository) and the same
frame-level preprocessing used here gives the three splits reported in
Table II of the manuscript:

```
data/dataset/train.json        # 634,799 frame samples
data/dataset/val.json          #  63,649 frame samples
dev split                      #   2,000 samples, instruction-level spot checks only
```

The `.json` files are lists of per-frame samples with keys `instruction`,
`map`, unit attributes and target commands.

`dict.pt` is **not** part of the upstream download. It is a pickled
`InstructionDict` (class defined in `behavior_clone/inst_dict.py`) holding the
instruction vocabulary, generated from the instruction corpus by a
preprocessing pass. In the experimental environment the entry point is, from
the root of the upstream checkout:

```bash
python scripts/behavior_clone/inst_dict.py    # writes dict.pt
```

A note for readers starting from a clean upstream clone: the public release
ships `scripts/behavior_clone/inst_dict.py` as a **class definition only**
(7,353 bytes, identical to `behavior_clone/inst_dict.py` in this bundle), with
no `__main__` block and no pickle output. The generation entry point -- count
word and instruction frequencies over the corpus, instantiate
`InstructionDict`, pickle it to `dict.pt` -- is part of the experimental fork
used for the reported runs. Add that short block before running the command
above. The resulting file is passed at `--inst_dict_path`; place it with the
two `.json` files under `data/dataset/` or pass its location on the command
line. It is not redistributed in this bundle.

**The data files are not redistributed here.** They are obtained from the
public upstream release cited above; this bundle contains only model code,
the training entry point and documentation. See `THIRD_PARTY_NOTICE.md` for
the upstream licence and attribution.

## 3. One-line reproduction

Run from this directory (`run.sh` sets `PYTHONPATH` so that `common_utils` is
importable):

```bash
python behavior_clone/train_executor.py \
  --train_dataset data/dataset/train.json \
  --val_dataset   data/dataset/val.json \
  --inst_dict_path data/dataset/dict.pt \
  --emb_field_dim 32 --prev_cmd_dim 64 \
  --num_conv_layers 3 --num_post_layers 1 \
  --conv_hid_dim 256 --army_out_dim 256 --other_out_dim 128 \
  --money_hid_layer 1 --conv_dropout 0.1 \
  --rnn_word_emb_dim 64 --word_emb_dropout 0.25 \
  --inst_hid_dim 128 --inst_encoder_type lstm \
  --batch_size 128 --gpu 0 --grad_clip 0.5 \
  --lr 0.002 --optim adamax --epochs 10 \
  --use_hist_inst 1 --pos_dim 16 --prev_cmd_rnn 1 \
  --top_num_inst -1 --seed 1 \
  --attention_type <attn> --attention_num_heads 8 --attention_dropout 0.1 \
  --no_early_stop \
  --model_folder saved_models/ablation_<attn>_seed1
```

where `<attn>` is one of the six variants:

| `<attn>` | Attention variant |
|---|---|
| `none` | Baseline CNN (no spatial attention) |
| `softmax` | Standard scaled dot-product attention (O(L^2)) |
| `linear` | Linear attention, ELU kernel (O(L)) |
| `windowed` | Windowed local attention (8x8 windows) |
| `performer` | Performer / FAVOR+ (orthogonal random features) |
| `nystrom` | Nystromformer (64 landmarks) |

A convenience wrapper is provided:

```bash
bash run.sh <attn> <seed>                  # e.g. bash run.sh linear 1
bash run.sh linear 1 --linear_kernel relu  # kernel ablation, Table XI
```

Key flags:

- `--no_early_stop` forces the full `--epochs` (10) so all variants are
  comparable.
- `--seed` sets seeds in **four places** (Python `random`, NumPy, CPU torch
  and CUDA torch, with distinct offsets in `common_utils/helper.py`) plus the
  DataLoader `worker_init_fn`, matching Table II of the manuscript.
- `--batch_size 128` is the fixed protocol (keeps efficiency comparisons on
  the same batch).
- `--linear_kernel {elu,relu,none}` selects the feature map; `elu` is the
  default used for all reported main results.

All 47 command-line flags used above are defined in the code; none is a
placeholder supplied by an external script.

## 4. Output files -> manuscript cross-reference

Each run writes, under `--model_folder`:

| Output file | Contents | Maps to |
|---|---|---|
| `loss_history.csv` | `epoch, train_loss, val_loss` (one row per epoch) | Table IV (Best Validation Loss, per-seed best over 10 epochs), Fig. 2 (learning curves), Sec. V.A (metric-definition sensitivity) |
| `val_accuracy.csv` | per-epoch five-head accuracy (`glob_cont, cmd_type, move, attack, gather` + correct/total counts) | Table VI (Action-Head Accuracy at the Best-Loss Epoch) |
| `train.log` | full training log (losses, `VALIDATION ACCURACY` lines) | supplementary only |
| `checkpointN.pt` / `best_checkpoint.pt` | saved weights | not distributed |

The six attention implementations live in `behavior_clone/spatial_attention.py`
(`create_attention` factory plus `Spatial{Softmax,Linear,Windowed,Performer,
Nystrom}Attention`), and are inserted into the feature extractor at
`behavior_clone/conv_module.py:89`. The five action heads are defined in
`behavior_clone/cmd_heads.py`.

Numerical claims in the manuscript can be verified **without retraining** by
running `../code/verify_stats.py` against the raw traces in `../data/`; that
script re-derives 21 reported quantities from the delivered CSV/JSON files.

## 5. Notes on reproducibility

- Training uses `torch.backends.cudnn.benchmark = True` and TF32 for
  conv/matmul (`torch.backends.cuda.matmul.allow_tf32 = True`,
  `torch.backends.cudnn.allow_tf32 = True`), matching the protocol of all
  reported runs. Softmax/exp/ELU stay in FP32.
- `spatial_attention.py` uses `head_dim = d_model / num_heads = 256 / 8 = 32`.
- The Nystrom variant inverts the landmark Gram matrix with diagonal loading
  (`Kc + 1e-2 * I`) and clamps its output instead of using a pseudoinverse;
  landmarks are segment means. Both repairs are described in Sec. III of the
  manuscript.
- The kernel ablation of Table XI is reproduced by
  `--linear_kernel {elu,relu,none}`; the default `elu` is
  `phi(x) = ELU(x) + 1`.
- Quality is the **per-seed best validation loss over the fixed ten-epoch
  budget**. The epoch-10 final loss is a separate metric definition, reported
  only as a sensitivity check; the two are never mixed in one table.

## 6. Upstream code and licence

Nineteen files under `behavior_clone/` and `common_utils/` are derived from
the public MiniRTS release cited in Sec. 2. Their original copyright headers
are retained verbatim. The upstream licence is **Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**, available at
`https://creativecommons.org/licenses/by-nc/4.0/`. This bundle is distributed
solely for non-commercial double-anonymous peer review. `THIRD_PARTY_NOTICE.md`
records the attribution, licence link, modifications and added files.

Files added by this study: `behavior_clone/spatial_attention.py`, `run.sh`,
`ENVIRONMENT.md`, `THIRD_PARTY_NOTICE.md`. Files modified from upstream are
identified by comments in each file (attention insertion and flags,
`--no_early_stop`, seed policy, per-epoch CSV writers, optional logging
fallback).

## 7. Scope and code provenance

This bundle covers the **main study only**: the attention module is inserted
at the spatial position (`32x32` map, `L = 1024`). The second position
examined in the manuscript (instruction sequence, `L <= 15`) is a negative
control based on attention-weight entropy from fixed checkpoints
(Table XII); it is an analysis of the same trained policies rather than a
separate training configuration, so it has no counterpart in this training
code.

### Code-version metadata (please read before comparing checksums)

The `*_run_meta.json` files in `../data/` record MD5 prefixes `c7b7488a`
(E2/E3 runs) and `917e5f84` (N7 runs) for `spatial_attention.py`, and
`0acc1827` for `conv_module.py`. **These are intermediate snapshots taken at
different stages of the experiment, not checksums of the final code.** The
implementation evolved continuously between the E3 and E8 runs: the Nystrom
diagonal-loading repair (`Kc + 1e-2 * I`) and the `--linear_kernel` switch
were added after the earlier snapshots, so the recorded MD5 values changed
from stage to stage. These values must therefore **not** be used as a
version-consistency test for the final implementation.

The files supplied here (MD5 prefixes `a9e4b8a5` / `763b76fd`) differ from
those snapshots for that reason, plus three non-functional edits made solely
for anonymous release:

1. the copyright header of the study-owned `spatial_attention.py` was
   replaced by a neutral statement (all 19 upstream-derived files keep their
   original copyright headers verbatim);
2. two Chinese comments were translated into English (the Nystrom landmark
   comment and the `--linear_kernel` docstring);
3. per-epoch `loss_history.csv` / `val_accuracy.csv` writers were added to
   the training entry point.

The attention implementations themselves are unchanged from the final
experimental version: no line of `_attention` or `_feature_map` in the five
attention classes, of the `create_attention` factory, of the `ConvFeatNet`
integration point, of the Nystrom repair or of the `--linear_kernel` argument
was modified. Their contents match Sec. III of the manuscript (six variants,
`d_model = 256`, `h = 8`, `head_dim = 32`, the three kernel switches, `8x8`
windows, `m = 64` orthogonal features, and `m = 64` segment-mean landmarks
with diagonal loading).

All raw traces in `../data/` were produced by the recorded runs and are
re-derived by `../code/verify_stats.py`. The run-metadata JSON files are kept
verbatim: this section is an explanatory note, and **no checksum inside any
delivered trace has been edited**. As an independent structural check, the
linear-attention module has 262,144 parameters, exactly the
baseline-to-attended parameter increase reported in the manuscript
(2,350,740 -> 2,612,884).
