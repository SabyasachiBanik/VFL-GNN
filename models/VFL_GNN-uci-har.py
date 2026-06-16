import os
import math
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


#  SEEDING

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#  CONFIG

T = 100                # Training rounds
η_initial = 0.01
η_min = 0.001
α = 0.5                # Initial consensus rate
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
warmup_rounds = 5      # Rounds before consensus starts [as per our paper logic]

NUM_CLIENTS = 2
NUM_CLASSES_UCI_HAR = 6
SEEDS = [0, 1, 2, 3, 4]   # five seeds (for a fair comparison with MARS-VFL)


#  UCI-HAR DATA LOADING (VERTICAL PARTITION)

def load_uci_har_vertically_partitioned(num_clients=2):
    """
    Loading UCI-HAR and vertically partition features across `num_clients`.

    - Each sample originally has 561 features.
    - We drop 1 extra feature so we use 560 features.
    - For 2 clients, each gets 280 features.
    - Data is reshaped to (N, 1, 1, width) so "width" is the feature dimension.
    """
    base_path = "./data/UCI_HAR_Dataset" 

    #  Load raw data 
    X_train = np.loadtxt(os.path.join(base_path, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(base_path, "train", "y_train.txt")).astype(int)

    X_test = np.loadtxt(os.path.join(base_path, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(base_path, "test", "y_test.txt")).astype(int)

    # Change the label to 0-5; Labels 1..6 -> 0..5
    y_train -= 1
    y_test  -= 1

    # To tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    # Feature dimension divisible by num_clients
    N_train, D = X_train.shape
    N_test,  D_test = X_test.shape
    assert D == D_test, "Train and test must have same feature dimension"

    features_per_client = D // num_clients          # integer division
    D_adj = features_per_client * num_clients       # maximum divisible part

    # Extra features being dropped (e.g., from 561 -> 560 for 2 clients)
    X_train = X_train[:, :D_adj]
    X_test  = X_test[:, :D_adj]

    # Reshape to (N, C, H, W) with C=1, H=1, W=D_adj 
    X_train = X_train.view(N_train, 1, 1, D_adj)
    X_test  = X_test.view(N_test,  1, 1, D_adj)

    # Vertical split along width (feature dimension)
    split_width = features_per_client

    def split_features(data):
        return [data[:, :, :, i*split_width:(i+1)*split_width]
                for i in range(num_clients)]

    train_splits = split_features(X_train)
    test_splits  = split_features(X_test)

    # Normalization 
    full_train = X_train.reshape(-1)
    mean = full_train.mean()
    std = full_train.std() + 1e-6

    def normalize_tensor(data, mean, std):
        return (data - mean) / std

    train_splits = [normalize_tensor(split, mean, std) for split in train_splits]
    test_splits  = [normalize_tensor(split,  mean, std) for split in test_splits]

    return train_splits, y_train, test_splits, y_test



#  CLIENT DATASETS & DATALOADERS

class ClientDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def prepare_client_dataloaders(train_splits, train_targets,
                               test_splits, test_targets,
                               batch_size=64):
    num_clients = len(train_splits)
    client_train_loaders = []
    client_test_loaders = []

    for i in range(num_clients):
        train_dataset = ClientDataset(train_splits[i], train_targets)
        test_dataset = ClientDataset(test_splits[i], test_targets)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )

        client_train_loaders.append(train_loader)
        client_test_loaders.append(test_loader)

    return client_train_loaders, client_test_loaders


#  GNN WITH ATTENTION

class AttentionGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(AttentionGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        # Multi-head attention components
        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.out_linear = nn.Linear(output_dim, output_dim)

        # Standard GNN transformation
        self.theta = nn.Parameter(torch.randn(input_dim, output_dim))

        # Combination weight between GNN and attention
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, H, W):
        batch_size, num_nodes, input_dim = H.shape

        # Standard GNN path: W * H * theta
        gnn_out = torch.relu(torch.matmul(torch.matmul(W, H), self.theta))

        # Attention path
        Q = self.q_linear(H).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_linear(H).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_linear(H).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_out = torch.matmul(attention_weights, V)
        attention_out = attention_out.view(batch_size, num_nodes, self.output_dim)
        attention_out = self.out_linear(attention_out)

        # Combining
        alpha_sig = torch.sigmoid(self.alpha)
        combined = alpha_sig * gnn_out + (1 - alpha_sig) * attention_out

        # Residual if dimensions match
        if input_dim == self.output_dim:
            combined = combined + H

        return self.layer_norm(combined)

class MultiLayerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_heads=4):
        super(MultiLayerGNN, self).__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(AttentionGNNLayer(input_dim, hidden_dims[0], num_heads))

        # Hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(AttentionGNNLayer(hidden_dims[i-1], hidden_dims[i], num_heads))

    def forward(self, H, W):
        for layer in self.layers:
            H = layer(H, W)
        return H


#  SPATIAL PRIOR ADJACENCY (1D FEATURES)

def create_spatial_prior_adjacency(height, width, sigma=2.0):
    """
    adjacency matrix creation with spatial priors.
    For UCI-HAR we have height=1, width = number of features for that client,
    so this becomes a 1D RBF over feature indices.
    """
    num_pixels = height * width
    W = torch.zeros(num_pixels, num_pixels)

    for i in range(num_pixels):
        row_i, col_i = i // width, i % width
        for j in range(num_pixels):
            row_j, col_j = j // width, j % width
            dist = math.sqrt((row_i - row_j)**2 + (col_i - col_j)**2)
            W[i, j] = math.exp(-dist**2 / (2 * sigma**2))

    return W


#  CLIENT MODEL

class EnhancedClientModel(nn.Module):
    def __init__(self, input_shape, gnn_hidden_dims=[64, 32],
                 mlp_hidden=64, num_classes=10, sparsity_reg=1e-4):
        super(EnhancedClientModel, self).__init__()
        c, h, w = input_shape  # e.g. (1, 1, feature_slice)
        self.flatten_dim = h * w
        self.sparsity_reg = sparsity_reg

        # Adjacency matrix with spatial prior
        spatial_prior = create_spatial_prior_adjacency(h, w, sigma=1.5)
        self.W = nn.Parameter(
            spatial_prior * 0.1 + torch.randn(self.flatten_dim, self.flatten_dim) * 0.01
        )

        # Multi-layer GNN (node feature dim starts at 1)
        self.gnn = MultiLayerGNN(input_dim=1, hidden_dims=gnn_hidden_dims, num_heads=4)

        final_gnn_dim = gnn_hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim * final_gnn_dim, mlp_hidden * 2),
            nn.BatchNorm1d(mlp_hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden * 2, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        H0 = x.view(B, self.flatten_dim, 1)  # (batch, num_nodes, 1)

        # Normalize adjacency rows
        W_norm = self.W / (self.W.norm(dim=1, keepdim=True) + 1e-6)

        H_out = self.gnn(H0, W_norm)  # (batch, num_nodes, hidden_dim)
        H_flat = H_out.view(B, -1)
        return self.classifier(H_flat)

    def get_sparsity_loss(self):
        return self.sparsity_reg * torch.norm(self.W, p=1)


#  LR SCHEDULER

class CosineAnnealingWarmup:
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            lr = self.eta_min + (self.base_lr - self.eta_min) * 0.5 * (
                1 + math.cos(math.pi * (epoch - self.warmup_epochs) /
                             (self.max_epochs - self.warmup_epochs))
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


#  RING TOPOLOGY & CONSENSUS

def get_ring_neighbors(k, K):
    return [(k - 1) % K, (k + 1) % K]

def adaptive_consensus_rate(round_idx, initial_alpha=0.5,
                            min_alpha=0.1, decay_rate=0.02):
    return max(min_alpha, initial_alpha * math.exp(-decay_rate * round_idx))

def train_one_round(client_models, client_optimizers, client_schedulers,
                    client_loaders, round_idx, consensus=True):
    K = len(client_models)
    local_correct = [0] * K
    local_total = [0] * K
    local_losses = [0.0] * K
    loss_fn = nn.CrossEntropyLoss()

    # ----- Local training -----
    for k in range(K):
        model = client_models[k]
        model.train()
        optimizer = client_optimizers[k]
        scheduler = client_schedulers[k]
        loader = client_loaders[k]

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            logits = model(batch_x)
            classification_loss = loss_fn(logits, batch_y)
            sparsity_loss = model.get_sparsity_loss()
            loss = classification_loss + sparsity_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            total_loss += classification_loss.item()

        scheduler.step(round_idx)
        local_correct[k] = correct
        local_total[k] = total
        local_losses[k] = total_loss / len(loader)

    # ----- Consensus over adjacency W (after warmup) -----
    if consensus and round_idx > warmup_rounds:
        current_alpha = adaptive_consensus_rate(round_idx - warmup_rounds, α, 0.1)
        W_updates = [model.W.clone().detach() for model in client_models]

        for k in range(K):
            neighbors = get_ring_neighbors(k, K)
            consensus_W = (1 - current_alpha) * W_updates[k]
            for n in neighbors:
                consensus_W += (current_alpha / len(neighbors)) * W_updates[n]
            client_models[k].W.data = consensus_W

    accs = [100.0 * c / t for c, t in zip(local_correct, local_total)]
    global_acc = sum(accs) / K
    return global_acc

def run_training(client_models, client_optimizers, client_schedulers,
                 train_loaders, num_rounds=T, verbose=False):
    if verbose:
        print("Starting training with GNN+attention and adaptive consensus...")
    for t in range(1, num_rounds + 1):
        global_acc = train_one_round(client_models, client_optimizers,
                                     client_schedulers, train_loaders, t)
        if verbose and t % 20 == 0:
            print(f"[Round {t}] Global Training Accuracy: {global_acc:.2f}%")


#  EVALUATION (INDIVIDUAL & ENSEMBLE) WITH F1

@torch.no_grad()
def evaluate_clients_with_f1(client_models, client_test_loaders):
    K = len(client_models)
    local_correct = [0] * K
    local_total = [0] * K
    all_preds = [[] for _ in range(K)]
    all_labels = [[] for _ in range(K)]

    for k in range(K):
        model = client_models[k]
        loader = client_test_loaders[k]
        model.eval()
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

            all_preds[k].extend(preds.cpu().numpy())
            all_labels[k].extend(batch_y.cpu().numpy())

        local_correct[k] = correct
        local_total[k] = total

    individual_accuracies = [100.0 * c / t for c, t in zip(local_correct, local_total)]
    individual_f1s = [f1_score(all_labels[k], all_preds[k], average='macro') * 100
                      for k in range(K)]

    return individual_accuracies, individual_f1s, all_preds, all_labels

@torch.no_grad()
def evaluate_ensemble_with_f1(client_models, client_test_loaders):
    K = len(client_models)
    all_predictions = []
    all_labels = None

    for k in range(K):
        model = client_models[k]
        loader = client_test_loaders[k]
        model.eval()

        model_predictions = []
        labels = []

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)
            model_predictions.append(probs.cpu())
            labels.append(batch_y.cpu())

        all_predictions.append(torch.cat(model_predictions, dim=0))
        if k == 0:
            all_labels = torch.cat(labels, dim=0)

    ensemble_probs = torch.stack(all_predictions, dim=0).mean(dim=0)
    ensemble_preds = torch.argmax(ensemble_probs, dim=1)

    correct = (ensemble_preds == all_labels).sum().item()
    total = all_labels.size(0)
    ensemble_accuracy = 100.0 * correct / total
    ensemble_f1 = f1_score(all_labels.numpy(), ensemble_preds.numpy(), average='macro') * 100

    return ensemble_accuracy, ensemble_f1


#  COMMUNICATION: ADJACENCY SIZE ESTIMATION

def print_adjacency_communication_stats(client_models):
    """
    Print adjacency matrix size per client and an estimate of the
    amount of adjacency data exchanged per consensus round.

    We consider that in a real system, each client would send its W
    once to each unique neighbor in the ring.
    """
    K = len(client_models)

    print("\n--- Adjacency Matrix & Communication Stats ---")
    total_params = 0
    for k, model in enumerate(client_models):
        W = model.W
        elems = W.numel()
        total_params += elems
        size_mb = elems * 4 / (1024 ** 2)  # float32
        print(f"Client {k}: W shape = {tuple(W.shape)}, "
              f"elements = {elems:,}, size ≈ {size_mb:.2f} MB")

    # Approx communication per consensus round
    total_elems_per_round = 0
    for k, model in enumerate(client_models):
        neighbors = set(get_ring_neighbors(k, K))  # unique neighbors
        elems = model.W.numel()
        total_elems_per_round += elems * len(neighbors)

    total_mb_per_round = total_elems_per_round * 4 / (1024 ** 2)
    print(f"Approx adjacency data exchanged per consensus round: "
          f"≈ {total_mb_per_round:.2f} MB\n")


#  SINGLE-SEED EXPERIMENT (UCI-HAR, 2 CLIENTS)

def run_experiment_uci_har(num_clients=2, verbose=False):
    """
    Run the complete experiment on UCI-HAR for one seed and fixed number of clients.
    Returns a dictionary with accuracies, F1, and training time.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"UCI-HAR: RUNNING EXPERIMENT WITH {num_clients} CLIENTS")
        print(f"{'='*60}")

    start_time = time.time()

    # Data
    train_splits, train_targets, test_splits, test_targets = \
        load_uci_har_vertically_partitioned(num_clients)

    train_loaders, test_loaders = prepare_client_dataloaders(
        train_splits, train_targets, test_splits, test_targets, batch_size
    )

    # Models
    input_shape = train_splits[0][0].shape  # (1, 1, slice_width)
    client_models = [
        EnhancedClientModel(
            input_shape,
            gnn_hidden_dims=[64, 32],
            mlp_hidden=64,
            num_classes=NUM_CLASSES_UCI_HAR
        ).to(device)
        for _ in range(num_clients)
    ]

    
    if verbose:
        print_adjacency_communication_stats(client_models)

    # Optimizers & schedulers
    client_optimizers = [
        torch.optim.AdamW(model.parameters(), lr=η_initial, weight_decay=1e-4)
        for model in client_models
    ]

    client_schedulers = [
        CosineAnnealingWarmup(opt, warmup_epochs=warmup_rounds,
                              max_epochs=T, eta_min=η_min)
        for opt in client_optimizers
    ]

    # Training
    run_training(client_models, client_optimizers, client_schedulers,
                 train_loaders, num_rounds=T, verbose=verbose)

    # Evaluation
    individual_accuracies, individual_f1s, _, _ = \
        evaluate_clients_with_f1(client_models, test_loaders)
    ensemble_accuracy, ensemble_f1 = \
        evaluate_ensemble_with_f1(client_models, test_loaders)

    avg_individual_accuracy = sum(individual_accuracies) / len(individual_accuracies)
    avg_individual_f1 = sum(individual_f1s) / len(individual_f1s)
    training_time = time.time() - start_time

    if verbose:
        print(f"\nResults for {num_clients} UCI-HAR clients:")
        print(f"Individual Accuracies: {[f'{acc:.2f}%' for acc in individual_accuracies]}")
        print(f"Individual F1 Scores: {[f'{f1:.2f}%' for f1 in individual_f1s]}")
        print(f"Average Individual Accuracy: {avg_individual_accuracy:.2f}%")
        print(f"Average Individual F1: {avg_individual_f1:.2f}%")
        print(f"Ensemble Accuracy: {ensemble_accuracy:.2f}%")
        print(f"Ensemble F1: {ensemble_f1:.2f}%")
        print(f"Training Time: {training_time:.1f}s\n")

    return {
        'num_clients': num_clients,
        'avg_individual_accuracy': avg_individual_accuracy,
        'avg_individual_f1': avg_individual_f1,
        'ensemble_accuracy': ensemble_accuracy,
        'ensemble_f1': ensemble_f1,
        'training_time': training_time,
        'individual_accuracies': individual_accuracies,
        'individual_f1s': individual_f1s
    }


#  NOW MULTI-SEED WRAPPER 

def run_multi_seed_experiment_uci_har(num_clients=2, seeds=SEEDS):
    """
    Run the UCI-HAR experiment for several seeds and compute
    mean ± std for key metrics, as in MARS-VFL.
    """
    results = []

    for seed in seeds:
        print(f"\n================ Seed {seed} ================")
        set_seed(seed)
        res = run_experiment_uci_har(num_clients=num_clients, verbose=False)
        results.append(res)

    # Metrics over seeds
    avg_ind_acc = np.array([r['avg_individual_accuracy'] for r in results])
    avg_ind_f1  = np.array([r['avg_individual_f1'] for r in results])
    ens_acc     = np.array([r['ensemble_accuracy'] for r in results])
    ens_f1      = np.array([r['ensemble_f1'] for r in results])
    train_time  = np.array([r['training_time'] for r in results])

    summary = {
        'Metric': [
            'Avg_Individual_Accuracy',
            'Avg_Individual_F1',
            'Ensemble_Accuracy',
            'Ensemble_F1',
            'Training_Time_sec'
        ],
        'Mean': [
            avg_ind_acc.mean(),
            avg_ind_f1.mean(),
            ens_acc.mean(),
            ens_f1.mean(),
            train_time.mean()
        ],
        'Std': [
            avg_ind_acc.std(ddof=0),
            avg_ind_f1.std(ddof=0),
            ens_acc.std(ddof=0),
            ens_f1.std(ddof=0),
            train_time.std(ddof=0)
        ]
    }

    summary_df = pd.DataFrame(summary)

    print("\n================ FINAL SUMMARY (5 seeds) ================")
    for _, row in summary_df.iterrows():
        name = row['Metric']
        mean = row['Mean']
        std = row['Std']
        if "Time" in name:
            print(f"{name}: {mean:.2f} ± {std:.2f} seconds")
        else:
            print(f"{name}: {mean:.2f} ± {std:.2f}")

    return results, summary_df


#  MAIN

if __name__ == "__main__":
    print(f"🔧 Using device: {device}")
    if device == 'cuda':
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")

    # 2-client UCI-HAR experiment, 5 seeds (matching MARS-VFL style)
    results, summary_df = run_multi_seed_experiment_uci_har(
        num_clients=NUM_CLIENTS,
        seeds=SEEDS
    )

    # Summary (mean & std)
    summary_df.to_csv('uci_har_vfl_gnn_2clients_5seeds_summary.csv', index=False)
    print("\n💾 Saved summary to 'uci_har_vfl_gnn_2clients_5seeds_summary.csv'")
