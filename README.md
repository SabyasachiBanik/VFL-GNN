# VFL-GNN: Decentralized Vertical Federated Learning via Graph-Based Learning Consensus

> Official code repository for the paper **"Decentralized Vertical Federated Learning via Graph-Based Learning Consensus".** 


VFL-GNN is a communication-efficient, **fully decentralized** Vertical Federated Learning (VFL) framework. Instead of exchanging model parameters or high-dimensional embeddings, clients collaboratively learn a cross-feature **adjacency matrix (AM)** and reach agreement through a **ring-based consensus protocol** — reformulating vertical collaboration as a distributed graph-learning problem. The framework provably reaches consensus at an exponential rate and mitigates the performance degradation ("VFL Syndrome") that VFL systems typically suffer as the number of clients grows.

## Key ideas

- **Adjacency-matrix exchange, not parameters.** Each client `k` maintains a learnable `W^(k) ∈ R^{D×D}` (`D` = total feature dimension across all clients) encoding intra- and inter-client feature dependencies. Only this lightweight matrix is shared, in half precision (Float16).
- **Ring-based consensus.** Each client communicates only with its two ring neighbors and performs consensus averaging, eliminating any central coordinator.
- **Exponential convergence guarantee.** Consensus error contracts at rate `ρ(α) = 1 − α(1 − λ₂)/2` with `λ₂ = cos(2π/K)` (Theorem 1).
- **Scalability.** Stable F1/accuracy as the client count `K` increases (2–15), where baselines degrade.

<p align="center">
 <img width="290" height="300" alt="fig1-readme" src="https://github.com/user-attachments/assets/97c0722d-449f-442f-a534-c0c29c03dd97" >
  <br>
  <em>Fig. 1: Decentralized VFL system design (4 clients) — clients exchange adjacency matrices over a ring topology.</em>
</p>




## Repository structure

```
VFL-GNN/
├── datasets/                 # Dataset documentation + source links
│   └── README.md
├── models/                   # Per-dataset VFL-GNN implementations (load + preprocess + vertical split + train)
│   ├── VFL_GNN-mm-imdb.py     # MM-IMDB   (image/text modality split)
│   ├── VFL_GNN-mustard.py     # MUSTARD   (utterance/context/speaker roles)
│   ├── VFL_GNN-ptb-xl.py      # PTB-XL    (ECG lead groups)
│   ├── VFL_GNN-uci-har.py     # UCI-HAR   (contiguous feature slices)
│   ├── robin_round MNIST.py    # MNIST     (round-robin feature split, 2–15 clients)
│   └── robin_round FMNIST.py   # F-MNIST   (round-robin feature split, 2–15 clients)
├── topology/                 # Topology comparison: ring / star / fully-connected / k-random / random-2
├── theoretical_validation/   # Empirical validation of the convergence theory (consensus error, ρ fit)
├── results/                  # Experiment outputs (metrics, figures)
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/SabyasachiBanik/VFL-GNN.git
cd VFL-GNN
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependency versions are scripted in [`requirements.txt`](./requirements.txt), targeting **Python 3.8–3.9**. Experiments were run on an **NVIDIA RTX 4090 (24 GB, CUDA 12.4)** with PyTorch.

## Datasets

VFL-GNN is evaluated on six datasets spanning image, human-activity, healthcare, emotion, and multimedia domains. Each dataset is downloaded from its public source and loaded, preprocessed, and vertically partitioned inside the corresponding `models/` script. Please refer to [`datasets/datasets.md`](./datasets/datasets.md) for more details.

| Dataset | Domain | Clients | Samples | Task | Source |
|---|---|---|---|---|---|
| MNIST | Image | 2–15 | 70,000 | 10-class | https://huggingface.co/datasets/ylecun/mnist |
| Fashion-MNIST | Image | 2–15 | 70,000 | 10-class | https://github.com/zalandoresearch/fashion-mnist |
| UCI-HAR | HAR | 2 | 10,299 | 6-class | https://archive.ics.uci.edu/dataset/240 |
| PTB-XL | Healthcare | 3 | 21,700 | 5-class, multilabel | https://physionet.org/content/ptb-xl/1.0.3/ |
| MUSTARD | Emotion | 3 | 690 | 2-class | https://github.com/soujanyaporia/MUStARD |
| MM-IMDB | Multimedia | 2 | 25,959 | 23-class, multilabel | https://github.com/johnarevalo/gmu-mmimdb |


## Usage

Each `models/` script is self-contained (loads its dataset, applies the vertical split, trains VFL-GNN, and the evaluation results are provided at `results/`):

```bash
python "models/VFL_GNN-uci-har.py"
python "models/VFL_GNN-ptb-xl.py"
python "models/VFL_GNN-mustard.py"
python "models/VFL_GNN-mm-imdb.py"
python "models/robin_round MNIST.py"
python "models/robin_round FMNIST.py"
```

Topology comparison and convergence validation:

```bash
# Topology comparison (run the script for each topology you want to evaluate)
python topology/ring_star.py          # ring + star
python topology/fully-connected.py    # fully-connected
python topology/k-random-neighbors.py # k-random (k = 4)
python topology/random-2-neighbors.py # random-2-neighbor

# Reproduce the empirical convergence study (Fashion-MNIST)
python theoretical_validation/theory_validation.py
```

### Key hyperparameters (paper Table II)

| Parameter | Symbol | Value |
|---|---|---|
| Training rounds | `T` | 50–100 |
| Warmup rounds | `t_w` | 5 |
| Learning rate | `η` | 1e-3 – 1e-2 |
| Consensus rate (init → min) | `α` | 0.5 → 0.1 (adaptive) |
| Clients | `K` | 2–15 |
| Optimizer | — | AdamW |
| AM sparsity regularizer | `λ` | 1e-4 (ℓ1) |
| Attention heads | `H_a` | 4 |
| AM precision | `P_W` | Float16 (2 B/elem) |
| Model-parameter precision | `P_θ` | Float32 (4 B/param) |

## Experimental & reproducibility notes

These notes document the current state of the experiments and the exact partition/topology choices, so results can be reproduced and interpreted correctly.

- **Vertical feature partitions.** The split scheme is dataset-specific and is implemented inside each `models/` script:
  - **MNIST / Fashion-MNIST** use a **round-robin** feature partition (see `models/robin_round MNIST.py` and `robin_round FMNIST.py`): input features (pixel columns) are assigned cyclically to the `K` clients (feature `i → client i mod K`), giving each client an interleaved, balanced subset. This supports the 2–15 client scaling experiments.
  - **UCI-HAR**: contiguous feature slices (561→560 features split into equal blocks).
  - **PTB-XL**: 12 ECG leads grouped into 3 clients (limb / augmented / precordial).
  - **MUSTARD**: 3 clients by feature role (utterance / context / speaker).
  - **MM-IMDB**: 2 clients by modality (image / text).

- **Seeds.** Most datasets are evaluated over multiple random seeds (e.g., PTB-XL reports mean ± 95% CI over 5 seeds). **MUSTARD and MM-IMDB are currently reported for a single seed (`seed = 42`); multi-seed runs are in progress.** Their observed trends are stable across runs: with a fixed vertical partition the adjacency-consensus dynamics are largely deterministic, in contrast to the variance introduced by the *randomized topologies* (k-random, random-2-neighbor).

- **Convergence validation dataset.** The empirical validation of the convergence theory (Theorem 1) — consensus-error decay on a log-linear scale and the theoretical-vs-empirical contraction rate `ρ(α)` — is conducted on **Fashion-MNIST** (see `theoretical_validation/`).

- **Topology configurations.** The topology study compares ring (`T_r`), centralized star (`T_sc`), fully-connected (`T_f`), k-random (`T_k`), and random-2-neighbor (`T_2`). In the **k-random topology, each client connects to `k = 4` randomly chosen neighbors** in all reported experiments, chosen to keep its connectivity comparable to the other evaluated topologies.

## Results (summary)

- **Communication efficiency & scalability** (MNIST, F-MNIST): VFL-GNN attains the highest F1 across `K ∈ {2,…,10}` while its per-round communication cost *decreases* with `K` (lightweight AM exchange), whereas baselines degrade in accuracy and grow in cost.
- **Topology optimality**: accuracy is nearly uniform (~88%) across topologies while communication cost varies widely; **ring** is the lowest-cost *fully decentralized* option.
- **Cross-domain** (UCI-HAR, PTB-XL, MUSTARD, MM-IMDB): competitive or superior score versus centralized VFL baselines at lower total communication cost.

See the paper and `results/` for full tables and figures.

## Citation

This paper has been accepted at IEEE LCN 2026. If you find this repository helpful, please cite our work: 

```bibtex
@INPROCEEDINGS{VFL-GNN,
AUTHOR="Sabyasachi Banik and Eric Samikwa and Torsten Ingo Braun",
TITLE="Decentralized Vertical Federated Learning via {Graph-Based} Learning
Consensus",
BOOKTITLE="2026 IEEE 51st Conference on Local Computer Networks (LCN) (LCN 2026)",
ADDRESS="Coimbra, Portugal",
YEAR=2026
}

```

## License

See [`LICENSE`](./LICENSE).
