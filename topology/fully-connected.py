#dataset FMNIST
# FULLY_CONNECTED TOPOLOGY IMPLEMENTATION
# Key difference from ring/pair_random: Each client connects to ALL other clients (K-1 neighbors)
# Communication cost: HIGHEST (O(K-1) per client vs O(2) for ring/pair_random)
# Expected: HIGHEST accuracy due to maximum information sharing


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
import pandas as pd
from sklearn.metrics import f1_score
import time
from scipy import stats


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# CONFIG
T = 100  # Training rounds
η_initial = 0.01
η_min = 0.001
α = 0.5  # Consensus rate
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
warmup_rounds = 5
SEEDS = [42, 43, 44, 45, 46]  # 5 seeds for averaging


# HORIZONTAL VERTICAL PARTITIONING 
def load_mnist_vertically_partitioned(num_clients=2, augment=True):
    """
    Horizontal vertical partitioning for Fashion-MNIST
    Each client gets equal width slices: width // num_clients columns
    This ensures all clients have the SAME dimensions (no mismatch in consensus)
    """
    # Load base datasets
    base_transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=base_transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=base_transform)

    # Extract raw data
    train_data = train_dataset.data.unsqueeze(1).float() / 255.0  # (60000, 1, 28, 28)
    test_data = test_dataset.data.unsqueeze(1).float() / 255.0
    train_targets = train_dataset.targets
    test_targets = test_dataset.targets

    _, _, height, width = train_data.shape
    split_width = width // num_clients

    def split_images(data):
        """Images split horizontally by width"""
        return [data[:, :, :, i*split_width:(i+1)*split_width] for i in range(num_clients)]

    train_splits = split_images(train_data)
    test_splits = split_images(test_data)

    # Augmentation
    if augment:
        augmented_train_splits = []
        for split in train_splits:
            augmented_split = apply_augmentation(split)
            augmented_train_splits.append(augmented_split)
        train_splits = augmented_train_splits

    # Normalization all
    train_splits = [normalize_data(split) for split in train_splits]
    test_splits = [normalize_data(split) for split in test_splits]

    return train_splits, train_targets, test_splits, test_targets



def apply_augmentation(data, rotation_degrees=10, translate_range=0.05):
    """Data augmentation to a tensor of images"""
    augmented_data = []

    for img in data:
        img_pil = transforms.ToPILImage()(img.squeeze())

        if random.random() > 0.5:  # 50% chance of rotation
            angle = random.uniform(-rotation_degrees, rotation_degrees)
            img_pil = transforms.functional.rotate(img_pil, angle)

        if random.random() > 0.5:  # 50% chance of translation
            translate_x = random.uniform(-translate_range, translate_range) * img_pil.size[0]
            translate_y = random.uniform(-translate_range, translate_range) * img_pil.size[1]
            img_pil = transforms.functional.affine(img_pil, angle=0, translate=[translate_x, translate_y], scale=1, shear=0)

        img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)
        augmented_data.append(img_tensor)

    return torch.cat(augmented_data, dim=0)


def normalize_data(data, mean=0.2860, std=0.3530):
    return (data - mean) / std


# CLIENT DATASETS
class ClientDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def prepare_client_dataloaders(train_splits, train_targets, test_splits, test_targets, batch_size=64):
    num_clients = len(train_splits)
    client_train_loaders = []
    client_test_loaders = []

    for i in range(num_clients):
        train_dataset = ClientDataset(train_splits[i], train_targets)
        test_dataset = ClientDataset(test_splits[i], test_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        client_train_loaders.append(train_loader)
        client_test_loaders.append(test_loader)

    return client_train_loaders, client_test_loaders


# MULTI-LAYER GNN WITH ATTENTION
class AttentionGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(AttentionGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        assert output_dim % num_heads == 0, 

        # Multi-head attention components
        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.out_linear = nn.Linear(output_dim, output_dim)

        # Standard GNN transformation
        self.theta = nn.Parameter(torch.randn(input_dim, output_dim))

        # Combination weights
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, H, W):
        batch_size, num_nodes, input_dim = H.shape

        # Standard GNN path
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

        # GNN and attention combining
        combined = torch.sigmoid(self.alpha) * gnn_out + (1 - torch.sigmoid(self.alpha)) * attention_out

        # Residual connection
        if input_dim == self.output_dim:
            combined = combined + H

        return self.layer_norm(combined)


class MultiLayerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_heads=4):
        super(MultiLayerGNN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(AttentionGNNLayer(input_dim, hidden_dims[0], num_heads))

        for i in range(1, len(hidden_dims)):
            self.layers.append(AttentionGNNLayer(hidden_dims[i-1], hidden_dims[i], num_heads))

    def forward(self, H, W):
        for layer in self.layers:
            H = layer(H, W)
        return H


# SPATIAL ADJACENCY INITIALIZATION
def create_spatial_prior_adjacency(height, width, sigma=2.0):
    num_pixels = height * width
    W = torch.zeros(num_pixels, num_pixels)

    for i in range(num_pixels):
        row_i, col_i = i // width, i % width
        for j in range(num_pixels):
            row_j, col_j = j // width, j % width
            dist = math.sqrt((row_i - row_j)**2 + (col_i - col_j)**2)
            W[i, j] = math.exp(-dist**2 / (2 * sigma**2))

    return W


# CLIENT MODEL
class EnhancedClientModel(nn.Module):
    def __init__(self, input_shape, gnn_hidden_dims=[64, 32], mlp_hidden=64, num_classes=10, sparsity_reg=1e-4):
        super(EnhancedClientModel, self).__init__()
        c, h, w = input_shape
        self.flatten_dim = h * w
        self.sparsity_reg = sparsity_reg

        # Adjacency initialization with spatial priors
        spatial_prior = create_spatial_prior_adjacency(h, w, sigma=1.5)
        self.W = nn.Parameter(spatial_prior * 0.1 + torch.randn(self.flatten_dim, self.flatten_dim) * 0.01)

        # Multi-layer GNN with attention
        self.gnn = MultiLayerGNN(input_dim=1, hidden_dims=gnn_hidden_dims, num_heads=4)

        # Classifier
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
        H0 = x.view(B, self.flatten_dim, 1)

        W_norm = self.W / (self.W.norm(dim=1, keepdim=True) + 1e-6)
        H_out = self.gnn(H0, W_norm)
        H_flat = H_out.view(B, -1)

        return self.classifier(H_flat)

    def get_sparsity_loss(self):
        return self.sparsity_reg * torch.norm(self.W, p=1)


# LEARNING RATE SCHEDULER
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
                1 + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



# TOPOLOGY DEFINITIONS - FULLY CONNECTED

def create_fully_connected_topology(K):
    """
    Fully Connected topology: Each client connects to ALL other clients
    
    Creates complete graph where:
    - Each client has K-1 neighbors (everyone except itself)
    - Communication cost: HIGHEST (K-1 neighbors each)
    - Strongest connectivity for maximum accuracy
    - Expected: Highest accuracy of all topologies
    
    Args:
        K: Number of clients
    
    Returns:
        Dictionary mapping client_id -> [all other client_ids]
    """
    topology = {}
    for k in range(K):
        # All clients except k itself
        neighbors = [i for i in range(K) if i != k]
        topology[k] = neighbors
    
    return topology


def get_fully_connected_neighbors(k, fc_topology):
    """
    Get neighbors for client k in fully_connected topology
    
    Args:
        k: Client index
        fc_topology: Dictionary from create_fully_connected_topology
    
    Returns:
        List of K-1 neighbor indices (all other clients)
    """
    return fc_topology[k]


# ============================================================================
# COMMUNICATION COST TRACKING
# ============================================================================

class CommunicationTracker:
    
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.reset()
    
    def reset(self):
        self.total_messages = 0
        self.total_bytes = 0
        self.round_messages = []
        self.round_bytes = []
    
    def record_consensus_round(self, W_size_bytes, fc_topology):
        messages_this_round = 0
        bytes_this_round = 0
        
        # In fully connected: each client sends to K-1 neighbors
        for k in range(self.num_clients):
            neighbors = get_fully_connected_neighbors(k, fc_topology)
            num_neighbors = len(neighbors)
            
            # Each client sends W to all its neighbors
            messages_this_round += num_neighbors
            bytes_this_round += num_neighbors * W_size_bytes
        
        self.total_messages += messages_this_round
        self.total_bytes += bytes_this_round
        self.round_messages.append(messages_this_round)
        self.round_bytes.append(bytes_this_round)
    
    def get_summary(self):
        """Communication cost summary"""
        return {
            'total_messages': self.total_messages,
            'total_bytes': self.total_bytes,
            'total_mb': self.total_bytes / (1024 * 1024),
            'avg_messages_per_round': np.mean(self.round_messages) if self.round_messages else 0,
            'avg_bytes_per_round': np.mean(self.round_bytes) if self.round_bytes else 0,
            'avg_mb_per_round': np.mean(self.round_bytes) / (1024 * 1024) if self.round_bytes else 0
        }



# TRAINING FUNCTIONS - FULLY CONNECTED WITH COMM TRACKING


def adaptive_consensus_rate(round_idx, initial_alpha=0.5, min_alpha=0.1, decay_rate=0.02):
    return max(min_alpha, initial_alpha * math.exp(-decay_rate * round_idx))


def train_one_round_fully_connected(client_models, client_optimizers, client_schedulers, 
                                     client_loaders, round_idx, fc_topology, 
                                     comm_tracker=None, consensus=True):
    """
    Training with FULLY_CONNECTED topology
    Key difference from pair_random/ring: Uses ALL K-1 neighbors
    - Ring: 2 structured neighbors
    - Pair_random: 2 random neighbors
    - Fully_connected: K-1 neighbors (ALL other clients)
    - Communication cost: HIGHEST (O(K-1) per client)
    - Expected accuracy: HIGHEST (maximum information sharing)
    """
    K = len(client_models)
    local_correct = [0] * K
    local_total = [0] * K
    local_losses = [0.0] * K
    loss_fn = nn.CrossEntropyLoss()

    # Local training (same as other topologies)
    for k in range(K):
        model = client_models[k]
        model.train()
        optimizer = client_optimizers[k]
        scheduler = client_schedulers[k]
        loader = client_loaders[k]

        total_loss = 0
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

    # Consensus phase - CRITICAL: Using ALL K-1 neighbors (fully connected)
    if consensus and round_idx > warmup_rounds:
        current_alpha = adaptive_consensus_rate(round_idx - warmup_rounds, α, 0.1)
        W_updates = [model.W.clone().detach() for model in client_models]
        
        # Communication cost
        if comm_tracker is not None:
            W_size_bytes = W_updates[0].nelement() * W_updates[0].element_size()
            comm_tracker.record_consensus_round(W_size_bytes, fc_topology)

        for k in range(K):
            # IMP: Use ALL other clients as neighbors (K-1 neighbors)
            neighbors = get_fully_connected_neighbors(k, fc_topology)
            
            consensus_W = (1 - current_alpha) * W_updates[k]
            for n in neighbors:
                consensus_W += (current_alpha / len(neighbors)) * W_updates[n]
            
            client_models[k].W.data = consensus_W

    accs = [100.0 * c / t for c, t in zip(local_correct, local_total)]
    global_acc = sum(accs) / K

    return global_acc


def run_training_fully_connected(client_models, client_optimizers, client_schedulers, 
                                  train_loaders, fc_topology, comm_tracker=None, 
                                  num_rounds=T, verbose=False):

    if verbose:
        print("Starting training with FULLY_CONNECTED topology...")
        print(f"Fully connected topology: each client connects to {len(fc_topology[0])} neighbors")
    
    for t in range(1, num_rounds + 1):
        global_acc = train_one_round_fully_connected(
            client_models, client_optimizers, client_schedulers, 
            train_loaders, t, fc_topology, comm_tracker
        )
        if verbose and t % 20 == 0:
            print(f"[Round {t}] Global Training Accuracy: {global_acc:.2f}%")


# EVALUATION
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
    individual_f1s = [f1_score(all_labels[k], all_preds[k], average='macro') * 100 for k in range(K)]

    return individual_accuracies, individual_f1s, all_preds, all_labels


@torch.no_grad()
def evaluate_ensemble_with_f1(client_models, client_test_loaders):
    K = len(client_models)
    all_predictions = []
    all_labels = []

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


# SINGLE SEED EXPERIMENT
def run_experiment_single_seed_fully_connected(num_clients, seed, verbose=False):
    """FULLY_CONNECTED topology with a single seed"""
    set_seed(seed)
    
    if verbose:
        print(f"  Running seed {seed} with FULLY_CONNECTED topology...")
    
    start_time = time.time()
    
    
    fc_topology = create_fully_connected_topology(num_clients)
    
    # Communication tracker
    comm_tracker = CommunicationTracker(num_clients)
    
    if verbose:
        print(f"  Fully connected topology: each client connects to {len(fc_topology[0])} neighbors")
    
    # Data preparation with robin-round split
    train_splits, train_targets, test_splits, test_targets = load_mnist_vertically_partitioned(
        num_clients, augment=True
    )
    train_loaders, test_loaders = prepare_client_dataloaders(
        train_splits, train_targets, test_splits, test_targets, batch_size
    )

    # Model initialization
    input_shape = train_splits[0].shape[1:]
    client_models = [
        EnhancedClientModel(input_shape, gnn_hidden_dims=[64, 32], mlp_hidden=64).to(device)
        for _ in range(num_clients)
    ]

    # Optimizers
    client_optimizers = [
        torch.optim.AdamW(model.parameters(), lr=η_initial, weight_decay=1e-4)
        for model in client_models
    ]

    # Schedulers
    client_schedulers = [
        CosineAnnealingWarmup(opt, warmup_epochs=warmup_rounds, max_epochs=T, eta_min=η_min)
        for opt in client_optimizers
    ]

    # Training with FULLY_CONNECTED
    run_training_fully_connected(
        client_models, client_optimizers, client_schedulers, 
        train_loaders, fc_topology, comm_tracker, num_rounds=T, verbose=False
    )

    # Evaluation
    individual_accuracies, individual_f1s, _, _ = evaluate_clients_with_f1(client_models, test_loaders)
    ensemble_accuracy, ensemble_f1 = evaluate_ensemble_with_f1(client_models, test_loaders)
    
    avg_individual_accuracy = sum(individual_accuracies) / len(individual_accuracies)
    avg_individual_f1 = sum(individual_f1s) / len(individual_f1s)
    
    training_time = time.time() - start_time
    
    # Communication summary
    comm_summary = comm_tracker.get_summary()
    
    return {
        'seed': seed,
        'num_clients': num_clients,
        'topology': 'fully_connected',
        'fc_topology': fc_topology,
        'avg_individual_accuracy': avg_individual_accuracy,
        'avg_individual_f1': avg_individual_f1,
        'ensemble_accuracy': ensemble_accuracy,
        'ensemble_f1': ensemble_f1,
        'training_time': training_time,
        'individual_accuracies': individual_accuracies,
        'individual_f1s': individual_f1s,
        'comm_total_messages': comm_summary['total_messages'],
        'comm_total_mb': comm_summary['total_mb'],
        'comm_avg_messages_per_round': comm_summary['avg_messages_per_round'],
        'comm_avg_mb_per_round': comm_summary['avg_mb_per_round']
    }


# MULTI-SEED EXPERIMENT WITH CI 
def compute_confidence_interval(data, confidence=0.95):
    """Compute mean and confidence interval"""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, ci


def run_experiment_multi_seed_fully_connected(num_clients, seeds=SEEDS, verbose=True):
    """Run FULLY_CONNECTED experiment across multiple seeds and compute statistics"""
    if verbose:
        print(f"\n{'='*70}")
        print(f"FULLY_CONNECTED TOPOLOGY: {num_clients} CLIENTS with {len(seeds)} SEEDS")
        print(f"{'='*70}")
    
    all_results = []
    
    for seed in seeds:
        result = run_experiment_single_seed_fully_connected(num_clients, seed, verbose=verbose)
        all_results.append(result)
    
    # Aggregated metrics across seeds
    avg_individual_accs = [r['avg_individual_accuracy'] for r in all_results]
    avg_individual_f1s = [r['avg_individual_f1'] for r in all_results]
    ensemble_accs = [r['ensemble_accuracy'] for r in all_results]
    ensemble_f1s = [r['ensemble_f1'] for r in all_results]
    training_times = [r['training_time'] for r in all_results]
    comm_total_messages = [r['comm_total_messages'] for r in all_results]
    comm_total_mb = [r['comm_total_mb'] for r in all_results]
    
    # Other statistics
    avg_ind_acc_mean, avg_ind_acc_ci = compute_confidence_interval(avg_individual_accs)
    avg_ind_f1_mean, avg_ind_f1_ci = compute_confidence_interval(avg_individual_f1s)
    ens_acc_mean, ens_acc_ci = compute_confidence_interval(ensemble_accs)
    ens_f1_mean, ens_f1_ci = compute_confidence_interval(ensemble_f1s)
    time_mean, time_ci = compute_confidence_interval(training_times)
    comm_msg_mean, comm_msg_ci = compute_confidence_interval(comm_total_messages)
    comm_mb_mean, comm_mb_ci = compute_confidence_interval(comm_total_mb)
    
    if verbose:
        print(f"\n📊 FULLY_CONNECTED RESULTS FOR {num_clients} CLIENTS (Averaged over {len(seeds)} seeds):")
        print(f"{'─'*70}")
        print(f"Average Individual Accuracy: {avg_ind_acc_mean:.2f}% ± {avg_ind_acc_ci:.2f}%")
        print(f"Average Individual F1:       {avg_ind_f1_mean:.2f}% ± {avg_ind_f1_ci:.2f}%")
        print(f"Ensemble Accuracy:           {ens_acc_mean:.2f}% ± {ens_acc_ci:.2f}%")
        print(f"Ensemble F1:                 {ens_f1_mean:.2f}% ± {ens_f1_ci:.2f}%")
        print(f"Training Time:               {time_mean:.1f}s ± {time_ci:.1f}s")
        print(f"{'─'*70}")
        print(f"📡 COMMUNICATION COSTS:")
        print(f"Total Messages:              {comm_msg_mean:.0f} ± {comm_msg_ci:.0f}")
        print(f"Total Data Transfer:         {comm_mb_mean:.2f} MB ± {comm_mb_ci:.2f} MB")
        print(f"Neighbors per Client:        {num_clients - 1} (ALL other clients)")
    
    return {
        'num_clients': num_clients,
        'topology': 'fully_connected',
        'neighbors_per_client': num_clients - 1,
        'avg_individual_accuracy_mean': avg_ind_acc_mean,
        'avg_individual_accuracy_ci': avg_ind_acc_ci,
        'avg_individual_f1_mean': avg_ind_f1_mean,
        'avg_individual_f1_ci': avg_ind_f1_ci,
        'ensemble_accuracy_mean': ens_acc_mean,
        'ensemble_accuracy_ci': ens_acc_ci,
        'ensemble_f1_mean': ens_f1_mean,
        'ensemble_f1_ci': ens_f1_ci,
        'training_time_mean': time_mean,
        'training_time_ci': time_ci,
        'comm_total_messages_mean': comm_msg_mean,
        'comm_total_messages_ci': comm_msg_ci,
        'comm_total_mb_mean': comm_mb_mean,
        'comm_total_mb_ci': comm_mb_ci,
        'raw_results': all_results
    }


# COMPREHENSIVE EVALUATION :)
def run_comprehensive_evaluation_fully_connected():
    """Run FULLY_CONNECTED experiments for client configurations K=3,5,7,9"""
    print("🚀 VFL-GNN FULLY_CONNECTED TOPOLOGY EVALUATION")
    print("=" * 80)
    print(f"Configuration: K=3,5,7,9 clients | {len(SEEDS)} seeds per config | 95% CI")
    print(f"Seeds: {SEEDS}")
    print(f"Topology: FULLY_CONNECTED (each client connects to ALL other clients)")
    print(f"Expected: HIGHEST accuracy, HIGHEST communication cost")
    print("=" * 80)
    
    results = []
    client_configs = [3, 5, 7, 9]  # Only these 4 configurations
    
    for num_clients in client_configs:
        result = run_experiment_multi_seed_fully_connected(num_clients, seeds=SEEDS, verbose=True)
        results.append(result)
    
    # Results table
    df = pd.DataFrame([
        {
            'Clients': r['num_clients'],
            'Topology': 'fully_connected',
            'Neighbors': r['neighbors_per_client'],
            'Avg_Ind_Acc': f"{r['avg_individual_accuracy_mean']:.2f} ± {r['avg_individual_accuracy_ci']:.2f}",
            'Avg_Ind_F1': f"{r['avg_individual_f1_mean']:.2f} ± {r['avg_individual_f1_ci']:.2f}",
            'Ensemble_Acc': f"{r['ensemble_accuracy_mean']:.2f} ± {r['ensemble_accuracy_ci']:.2f}",
            'Ensemble_F1': f"{r['ensemble_f1_mean']:.2f} ± {r['ensemble_f1_ci']:.2f}",
            'Comm_Msgs': f"{r['comm_total_messages_mean']:.0f} ± {r['comm_total_messages_ci']:.0f}",
            'Comm_MB': f"{r['comm_total_mb_mean']:.2f} ± {r['comm_total_mb_ci']:.2f}",
            'Time(s)': f"{r['training_time_mean']:.1f} ± {r['training_time_ci']:.1f}"
        } for r in results
    ])
    
    print("\n" + "="*120)
    print("🏆 FULLY_CONNECTED COMPREHENSIVE RESULTS TABLE (Mean ± 95% CI)")
    print("="*120)
    print(df.to_string(index=False))
    
    # Comparison table
    comm_df = pd.DataFrame([
        {
            'Clients (K)': r['num_clients'],
            'Neighbors': r['neighbors_per_client'],
            'Total_Messages': f"{r['comm_total_messages_mean']:.0f}",
            'Total_MB': f"{r['comm_total_mb_mean']:.2f}",
            'Messages_per_Round': f"{r['comm_total_messages_mean'] / (T - warmup_rounds):.1f}",
            'MB_per_Round': f"{r['comm_total_mb_mean'] / (T - warmup_rounds):.3f}"
        } for r in results
    ])
    
    print("\n" + "="*100)
    print("📡 COMMUNICATION COST BREAKDOWN")
    print("="*100)
    print(comm_df.to_string(index=False))
    print(f"\nNote: Communication occurs during consensus phase ({T - warmup_rounds} rounds)")
    print(f"Fully connected: Each client sends to {client_configs[-1] - 1} neighbors (for K=9)")
    
    return results, df, comm_df


# -------------------- MAIN --------------------
if __name__ == "__main__":
    print(f"🔧 Device: {device}")
    if device == 'cuda':
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")
    
    print(f"🔧 FULLY_CONNECTED Topology Implementation")
    print(f"🔧 Key: Each client connects to ALL other clients (K-1 neighbors)")
    print(f"🔧 Communication: HIGHEST (every client talks to everyone)")
    print(f"🔧 Expected: HIGHEST accuracy, HIGHEST communication cost")
    print(f"🔧 Multi-Seed Evaluation: {len(SEEDS)} seeds")
    print()
    
    # Evaluation
    results, results_table, comm_table = run_comprehensive_evaluation_fully_connected()
    
    # Result table
    results_table.to_csv('vfl_gnn_fully_connected_results.csv', index=False)
    print(f"\n💾 Results saved to 'vfl_gnn_fully_connected_results.csv'")
    
    comm_table.to_csv('vfl_gnn_fully_connected_comm_costs.csv', index=False)
    print(f"💾 Communication costs saved to 'vfl_gnn_fully_connected_comm_costs.csv'")
    
    # Detailed seed-level results
    detailed_data = []
    for config_result in results:
        for raw_result in config_result['raw_results']:
            detailed_data.append({
                'num_clients': raw_result['num_clients'],
                'seed': raw_result['seed'],
                'topology': 'fully_connected',
                'neighbors': raw_result['num_clients'] - 1,
                'avg_individual_accuracy': raw_result['avg_individual_accuracy'],
                'avg_individual_f1': raw_result['avg_individual_f1'],
                'ensemble_accuracy': raw_result['ensemble_accuracy'],
                'ensemble_f1': raw_result['ensemble_f1'],
                'training_time': raw_result['training_time'],
                'comm_total_messages': raw_result['comm_total_messages'],
                'comm_total_mb': raw_result['comm_total_mb'],
                'comm_avg_messages_per_round': raw_result['comm_avg_messages_per_round'],
                'comm_avg_mb_per_round': raw_result['comm_avg_mb_per_round']
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv('vfl_gnn_fully_connected_detailed_seeds.csv', index=False)
    print(f"💾 Detailed seed results saved to 'vfl_gnn_fully_connected_detailed_seeds.csv'")
    
    print(f"\n✅ FULLY_CONNECTED Evaluation Complete!")
    print(f"📊 Expected: HIGHEST accuracy (maximum info sharing) + HIGHEST comm cost")
    print(f"🎉 Tested {len(results)} configurations × {len(SEEDS)} seeds = {len(results) * len(SEEDS)} total experiments")
    print(f"\n📈 Topology Comparison:")
    print(f"  - Ring:           2 neighbors (structured)  → Medium-High accuracy, Low comm cost")
    print(f"  - Pair_Random:    2 neighbors (random)      → Medium accuracy, Low comm cost")
    print(f"  - Fully_Connected: K-1 neighbors (complete) → HIGHEST accuracy, HIGHEST comm cost")