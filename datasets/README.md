# Datasets

This repository does **not** ship raw data or separate dataset-loader files.
Each dataset is downloaded from its public source (links below) and is loaded,
preprocessed, and **vertically partitioned across clients inside the
corresponding `models/` script**. This file documents, per dataset, where that
happens and how the vertical split is defined.

| # | Dataset | Source | Loaded & processed in | Vertical split |
|---|---------|--------|-----------------------|----------------|
| 1 | MNIST | https://huggingface.co/datasets/ylecun/mnist | `models/robin_round MNIST.py` | image pixels split into per-client column blocks |
| 2 | Fashion-MNIST | https://github.com/zalandoresearch/fashion-mnist | `models/robin_round FMNIST.py`, `topology/` scripts | image pixels split into per-client column blocks |
| 3 | UCI-HAR | https://archive.ics.uci.edu/dataset/240 | `models/VFL_GNN-uci-har.py` | 561→560 features sliced into 2 contiguous 280-feature blocks |
| 4 | PTB-XL | https://physionet.org/content/ptb-xl/1.0.3/ | `models/VFL_GNN-ptb-xl.py` | 12 ECG leads → 3 clients by lead group |
| 5 | MUSTARD | https://github.com/soujanyaporia/MUStARD | `models/VFL_GNN-mustard.py` | 3 clients by feature role (utterance / context / speaker) |
| 6 | MM-IMDB | https://github.com/johnarevalo/gmu-mmimdb | `models/VFL_GNN-mm-imdb.py` | 2 clients by modality (image / text) |

## Per-dataset details

**1–2. MNIST / Fashion-MNIST** — loaded via `torchvision` in the `robin_round`
scripts. Images are partitioned vertically (each client receives a contiguous
block of pixel columns) so the federated parties hold disjoint feature regions
of the same samples. *Confirm the exact partition function/seed in those two
scripts; they are summarized here from the project convention.*

**3. UCI-HAR** — `models/VFL_GNN-uci-har.py`. Loads the pre-split
`train/X_train.txt` … `test/y_test.txt`. The 561 hand-crafted features are
truncated to 560 (largest multiple of the client count) and split into 2
contiguous 280-feature blocks, one per client; a single global train mean/std
normalizes every block. Uses the dataset's native train/test split (no val).

**4. PTB-XL** — `models/VFL_GNN-ptb-xl.py`. Reads `ptbxl_database.csv` and
`scp_statements.csv`, aggregates SCP codes into the 5 diagnostic superclasses
(NORM, MI, STTC, CD, HYP), and streams the 500 Hz WFDB records on demand
(`wfdb.rdsamp`). The 12 leads are split across 3 clients by anatomical group:
limb I/II/III `[0,1,2]`, augmented aVR/aVL/aVF `[3,4,5]`, precordial V1–V6
`[6–11]`. Stratified 80/20 train/test split, `random_state=42`.
**License:** PhysioNet credentialed/data-use terms — download only; do not
commit or mirror the signals.

**5. MUSTARD** — `models/VFL_GNN-mustard.py`. Loads `sarcasm_data.json`, encodes
utterance and (mean-pooled) context text with Sentence-BERT
(`all-MiniLM-L6-v2`, 384-d each) and the speaker as a one-hot vector. The 3
clients correspond to these three roles. Stratified 70/15/15 split,
`random_state=42`. Features are cached (`.npy`) after first extraction.
**License:** derived from copyrighted TV material — download from the source
repo only; do not commit the raw data.

**6. MM-IMDB** — `models/VFL_GNN-mm-imdb.py`. The canonical source is
`johnarevalo/gmu-mmimdb` (cited above); because its original download links are
no longer available, the code loads the working Hugging Face mirror
**`sxj1215/mmimdb`** via the `datasets` library. Images are embedded with a
frozen ResNet50 (2048-d) and plot text with a frozen DistilBERT `[CLS]`
embedding (768-d); these are cached as `.pt`. The 2 clients correspond to the
two modalities (image / text). 23-label multilabel genre task, split 70/10/20
(seed 42). Loaders use `shuffle=False` to keep the two parties row-aligned.

## Reproducibility notes

- All splits are seeded (see each script); PTB-XL reports multi-seed mean ± 95% CI.
- Library versions are pinned in the repo-root `requirements.txt`.
- No dataset bytes are tracked in git; obtain each dataset from its source link
  and point the script's data path at the unpacked files.
