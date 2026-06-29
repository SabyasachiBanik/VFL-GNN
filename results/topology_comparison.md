### Table IV — Accuracy (%) and Per-Round Communication Cost (MB) across Topologies

Client configurations *K* ∈ {3, 5, 7, 9}.

| Topology | Metric | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|
| Ring (T<sub>r</sub>) | Acc. | 88.3 | 88.2 | 88.0 | 88.2 |
| Ring (T<sub>r</sub>) | *Comm (MB)* | **69.0** | **35.5** | **31.8** | **23.0** |
| Centralized Star (T<sub>sc</sub>) | Acc. | 88.6 | **88.3** | 88.1 | 88.2 |
| Centralized Star (T<sub>sc</sub>) | *Comm (MB)* | 46.0 | 28.4 | 27.3 | 20.5 |
| Fully-Connected (T<sub>f</sub>) | Acc. | 88.4 | 87.9 | **88.2** | 88.2 |
| Fully-Connected (T<sub>f</sub>) | *Comm (MB)* | 138.1 | 142.1 | 190.9 | 184.1 |
| k-Random (T<sub>kr</sub>) | Acc. | 88.3 | 88.1 | 88.0 | **88.3** |
| k-Random (T<sub>kr</sub>) | *Comm (MB)* | 128.1 | 142.1 | 127.3 | 92.1 |
| Random-2 (T<sub>2</sub>) | Acc. | **88.9** | 87.8 | 87.2 | 86.6 |
| Random-2 (T<sub>2</sub>) | *Comm (MB)* | **69.0** | **35.5** | **31.8** | **23.0** |

Columns 3–9 are the number of clients *K*. Accuracy is in %; communication cost is in MB per round. Higher accuracy is better; lower communication cost is better. Accuracy is nearly uniform (88.0–88.9%) across topologies, while communication cost varies widely. **Bold accuracy** marks the best per *K*; **bold communication** marks the lowest cost among the *fully decentralized* topologies — Ring (T<sub>r</sub>) and Random-2 (T<sub>2</sub>) tie here. The Centralized Star (T<sub>sc</sub>) reaches marginally lower cost but relies on a central coordinator, so it is excluded from the decentralized comparison.
