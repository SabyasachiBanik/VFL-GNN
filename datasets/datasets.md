# Datasets

This repository does **not** ship raw data or separate dataset-loader files.
Each dataset is downloaded from its public source (links below) and is loaded,
preprocessed, and **vertically partitioned across clients inside the
corresponding `models/` script**.

| # | Dataset | Source | Loaded & processed in | Vertical split |
|---|---------|--------|-----------------------|----------------|
| 1 | MNIST | https://huggingface.co/datasets/ylecun/mnist | `models/robin_round MNIST.py` | image pixels split into per-client column blocks |
| 2 | Fashion-MNIST | https://github.com/zalandoresearch/fashion-mnist | `models/robin_round FMNIST.py`, `topology/` scripts | image pixels split into per-client column blocks |
| 3 | UCI-HAR | https://archive.ics.uci.edu/dataset/240 | `models/VFL_GNN-uci-har.py` | 561→560 features sliced into 2 contiguous 280-feature blocks |
| 4 | PTB-XL | https://physionet.org/content/ptb-xl/1.0.3/ | `models/VFL_GNN-ptb-xl.py` | 12 ECG leads → 3 clients by lead group |
| 5 | MUSTARD | https://github.com/soujanyaporia/MUStARD | `models/VFL_GNN-mustard.py` | 3 clients by feature role (utterance / context / speaker) |
| 6 | MM-IMDB | https://github.com/johnarevalo/gmu-mmimdb | `models/VFL_GNN-mm-imdb.py` | 2 clients by modality (image / text) |

