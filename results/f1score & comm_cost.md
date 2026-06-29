### Table III — F1-Score and Per-Round Communication Cost (MB) of VFL-GNN vs. Baselines

| Dataset | Method | Metric | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MNIST | Non-Fed. | F1 Score | 0.822 | 0.619 | 0.480 | 0.345 | 0.261 | 0.159 | 0.165 | 0.130 | 0.100 |
| MNIST | Non-Fed. | Comm (MB) | – | – | – | – | – | – | – | – | – |
| MNIST | VertiComb | F1 Score | 0.932 | 0.902 | 0.866 | 0.838 | 0.746 | 0.670 | 0.664 | 0.544 | 0.450 |
| MNIST | VertiComb | Comm (MB) | 0.44 | 0.92 | 1.46 | 2.05 | 2.81 | 3.68 | 4.61 | 5.13 | 5.72 |
| MNIST | De-VertiFL | F1 Score | 0.960 | 0.955 | 0.942 | 0.910 | 0.870 | 0.822 | 0.764 | 0.684 | 0.584 |
| MNIST | De-VertiFL | Comm (MB) | 0.37 | 1.68 | 4.53 | 10.02 | 19.18 | 30.18 | 43.61 | 60.39 | 91.19 |
| MNIST | **VFL-GNN** | **F1 Score** | **0.985** | **0.983** | **0.982** | **0.977** | **0.971** | **0.968** | **0.960** | **0.955** | **0.948** |
| MNIST | **VFL-GNN** | **Comm (MB)** | **1.17** | **0.78** | **0.59** | **0.47** | **0.39** | **0.33** | **0.30** | **0.26** | **0.24** |
| FMNIST | Non-Fed. | F1 Score | 0.753 | 0.658 | 0.617 | 0.548 | 0.424 | 0.284 | 0.257 | 0.211 | 0.159 |
| FMNIST | Non-Fed. | Comm (MB) | – | – | – | – | – | – | – | – | – |
| FMNIST | VertiComb | F1 Score | 0.816 | 0.798 | 0.780 | 0.748 | 0.740 | 0.734 | 0.714 | 0.712 | 0.628 |
| FMNIST | VertiComb | Comm (MB) | 0.44 | 0.92 | 1.46 | 2.05 | 2.81 | 3.68 | 4.61 | 5.13 | 5.72 |
| FMNIST | De-VertiFL | F1 Score | 0.824 | 0.812 | 0.810 | 0.788 | 0.784 | 0.764 | 0.688 | 0.652 | 0.648 |
| FMNIST | De-VertiFL | Comm (MB) | 0.12 | 0.56 | 1.51 | 3.34 | 6.39 | 10.07 | 14.55 | 20.26 | 30.42 |
| FMNIST | **VFL-GNN** | **F1 Score** | **0.892** | **0.891** | **0.891** | **0.889** | **0.885** | **0.881** | **0.882** | **0.886** | **0.885** |
| FMNIST | **VFL-GNN** | **Comm (MB)** | **1.17** | **0.78** | **0.59** | **0.47** | **0.39** | **0.33** | **0.30** | **0.26** | **0.24** |

Columns 2–10 are the number of clients *K*. Higher F1 is better; lower communication cost is better. "–" = not applicable (Non-Fed. clients do not communicate). **Bold** = our method.
