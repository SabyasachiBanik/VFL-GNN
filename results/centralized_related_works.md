### Comparison with Centralised VFL

| Method | Metric | UCI-HAR | PTB-XL | MUSTARD | MM-IMDB |
|---|---|---|---|---|---|
| SC-VFL [1] | Score | 95.03 | **57.41** | 57.83 | 56.63 |
| SC-VFL [1] | *Comm* | 114.5 MB | 271.0 GB | 187.3 MB | 3.56 GB |
| FedBCD [2] | Score | 95.03 | 54.92 | 54.49 | 56.83 |
| FedBCD [2] | *Comm* | 79.0 MB | 100.4 GB | 152.9 MB | 0.78 GB |
| C-VFL [3] | Score | 95.01 | 53.41 | 54.34 | 56.31 |
| C-VFL [3] | *Comm* | **25.6 MB** | 27.3 GB | 36.6 MB | 0.23 GB |
| EFVFL [4] | Score | 95.21 | 54.53 | 56.52 | 56.42 |
| EFVFL [4] | *Comm* | 26.5 MB | 35.3 GB | 39.0 MB | 0.27 GB |
| **VFL-GNN** | **Score** | **95.92** | 53.78 | **62.14** | **61.77** |
| **VFL-GNN** | ***Comm*** | 57.0 MB | **27.2 GB** | **14.3 MB** | **0.21 GB** |

*Score* is classification accuracy for UCI-HAR and MUSTARD, and F1-score for the imbalanced PTB-XL and MM-IMDB datasets. *Comm* is the **total** communication cost for training (note units: MB vs. GB). Higher score is better; lower communication cost is better. **Bold** marks the best value per dataset per metric. VFL-GNN is fully decentralized, whereas all baselines are centralized; despite this, it attains the best score on three of four datasets and the lowest communication cost on three of four.



#### Baselines
- **[1] SC-VFL / MARS-VFL** — Shen et al., "MARS-VFL: A unified benchmark for vertical federated learning with realistic evaluation," *NeurIPS Datasets & Benchmarks*, 2025.
- **[2] FedBCD** — Liu et al., "FedBCD: A communication-efficient collaborative learning framework for distributed features," *IEEE TSP*, vol. 70, pp. 4277–4290, 2022.
- **[3] C-VFL** — Castiglia et al., "Compressed-VFL: Communication-efficient learning with vertically partitioned data," *ICML*, PMLR 162, 2022, pp. 2738–2766.
- **[4] EFVFL** — Valdeira et al., "Communication-efficient vertical federated learning via compressed error feedback," *IEEE TSP*, 2025 (early access).
