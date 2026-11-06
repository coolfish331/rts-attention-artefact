# Third-party notice

Nineteen source files in `behavior_clone/` and `common_utils/` are derived
from the public MiniRTS release:

- Repository: `https://github.com/facebookresearch/minirts`
- Associated publication: H. Hu, D. Yarats, Q. Gong, Y. Tian, and M. Lewis,
  "Hierarchical Decision Making by Generating and Following Natural Language
  Instructions," in *Advances in Neural Information Processing Systems
  (NeurIPS)*, vol. 32, 2019 (arXiv:1906.00744).
- Upstream licence: **Creative Commons Attribution-NonCommercial 4.0
  International (CC BY-NC 4.0)**.
  Full text: `https://creativecommons.org/licenses/by-nc/4.0/legalcode`
  Summary: `https://creativecommons.org/licenses/by-nc/4.0/`

The original copyright notices ("Copyright (c) Facebook, Inc. and its
affiliates") and the licence pointers in the file headers are retained
verbatim, as required by the licence. This notice provides the licence
reference and attribution that the upstream `LICENSE` file would otherwise
supply; the data and code are used here solely for non-commercial academic
peer review.

## Modifications

The following changes were made to the upstream files for this study. Each
modified file carries a short comment marking the change.

| Area | Modification |
|---|---|
| `behavior_clone/conv_module.py` | insertion point for the optional attention module after the CNN backbone; `linear_kernel` pass-through |
| `behavior_clone/conv_glob_encoder.py` | `--attention_type` and `--linear_kernel` command-line flags |
| `behavior_clone/train_executor.py` | `--no_early_stop`, explicit seed policy, cuDNN/TF32 flags, and per-epoch `loss_history.csv` / `val_accuracy.csv` writers |
| `behavior_clone/spatial_attention.py` | **new file** implementing all six attention variants and the `create_attention` factory |
| `common_utils/helper.py` | `set_all_seeds` seeding Python, NumPy, CPU and CUDA torch |
| `common_utils/multi_counter.py`, `common_utils/result_stat.py` | optional `tensorboardX` imports so the code runs without that logging dependency |
| `run.sh`, `ENVIRONMENT.md`, `THIRD_PARTY_NOTICE.md` | **new files** documenting the reproduction procedure |

No upstream data files, checkpoints, weights or logs are redistributed in
this bundle. The complete upstream release, including its `LICENSE` file and
the dataset download script, is available from the repository above.

This notice is provided for attribution and licence compliance only. It
contains no information about the authors of the submitted manuscript.
