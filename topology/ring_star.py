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
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(43)

# CONFIG 
T = 100  # Training rounds
η_initial = 0.01
η_min = 0.001
α = 0.5  # Consensus rate
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
warmup_rounds = 5

# DATA LOADING
def load_mnist_vertically_partitioned(num_clients=2, augment=True):
    base_transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=base_transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=base_transform)

    train_data = train_dataset.data.unsqueeze(1).float() / 255.0
    test_data = test_dataset.data.unsqueeze(1).float() / 255.0
    train_targets = train_dataset.targets
    test_targets = test_dataset.targets

    _, _, height, width = train_data.shape
    split_width = width // num_clients

    def split_images(data):
        return [data[:, :, :, i*split_width:(i+1)*split_width] for i in range(num_clients)]

    train_splits = split_images(train_data)
    test_splits = split_images(test_data)

    if augment:
        augmented_train_splits = []
        for split in train_splits:
            augmented_split = apply_augmentation(split)
            augmented_train_splits.append(augmented_split)
        train_splits = augmented_train_splits

    train_splits = [normalize_data(split) for split in train_splits]
    test_splits = [normalize_data(split) for split in test_splits]

    return train_splits, train_targets, test_splits, test_targets

def apply_augmentation(data, rotation_degrees=10, translate_range=0.05):
    augmented_data = []
    for img in data:
        img_pil = transforms.ToPILImage()(img.squeeze())
        if random.random() > 0.5:
            angle = random.uniform(-rotation_degrees, rotation_degrees)
            img_pil = transforms.functional.rotate(img_pil, angle)
        if random.random() > 0.5:
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

# GNN MODEL
class AttentionGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(AttentionGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.out_linear = nn.Linear(output_dim, output_dim)

        self.theta = nn.Parameter(torch.randn(input_dim, output_dim))
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, H, W):
        batch_size, num_nodes, input_dim = H.shape

        gnn_out = torch.relu(torch.matmul(torch.matmul(W, H), self.theta))

        Q = self.q_linear(H).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.k_linear(H).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.v_linear(H).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_out = torch.matmul(attention_weights, V)
        attention_out = attention_out.view(batch_size, num_nodes, self.output_dim)
        attention_out = self.out_linear(attention_out)

        combined = torch.sigmoid(self.alpha) * gnn_out + (1 - torch.sigmoid(self.alpha)) * attention_out

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

class EnhancedClientModel(nn.Module):
    def __init__(self, input_shape, gnn_hidden_dims=[64, 32], mlp_hidden=64, num_classes=10, sparsity_reg=1e-4):
        super(EnhancedClientModel, self).__init__()
        c, h, w = input_shape
        self.flatten_dim = h * w
        self.sparsity_reg = sparsity_reg

        spatial_prior = create_spatial_prior_adjacency(h, w, sigma=1.5)
        self.W = nn.Parameter(spatial_prior * 0.1 + torch.randn(self.flatten_dim, self.flatten_dim) * 0.01)

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
        H0 = x.view(B, self.flatten_dim, 1)
        W_norm = self.W / (self.W.norm(dim=1, keepdim=True) + 1e-6)
        H_out = self.gnn(H0, W_norm)
        H_flat = H_out.view(B, -1)
        return self.classifier(H_flat)

    def get_sparsity_loss(self):
        return self.sparsity_reg * torch.norm(self.W, p=1)

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

# TOPOLOGY IMPLEMENTATIONS

def get_ring_neighbors(k, K):
    """Ring topology: Each client connects to 2 neighbors (left and right)"""
    return [(k - 1) % K, (k + 1) % K]

def get_star_neighbors(k, K):
    """Star topology: Central node (client 0) connects to all, others only to center"""
    if k == 0:  # Central node
        return list(range(1, K))
    else:  # Peripheral nodes
        return [0]

# Ffully_connected, k_random, and pair_random topologies are implemented
# in separate scripts. This file covers only the Ring and Star topologies.

def get_topology_neighbors(k, K, topology_type, **kwargs):
    """Unified function to get neighbors based on topology type (Ring and Star only)"""
    if topology_type == 'ring':
        return get_ring_neighbors(k, K)
    elif topology_type == 'star':
        return get_star_neighbors(k, K)
    else:
        raise ValueError(
            f"Unknown topology type: {topology_type}. "
            f"This script supports only 'ring' and 'star'."
        )

def calculate_communication_cost(K, flatten_dim, topology_type, **kwargs):
    """
    Communication cost in MB per consensus round
    
    Args:
        K: Number of clients
        flatten_dim: Dimension of adjacency matrix W (height × width of each client's data)
        topology_type: Type of network topology
        **kwargs: Additional topology-specific parameters
    
    Returns:
        tuple: (num_messages, mb_per_message, total_mb_per_round)
    """
    # Calculating number of messages exchanged
    if topology_type == 'ring':
        num_messages = K * 2  # Each client sends to 2 neighbors
    elif topology_type == 'star':
        num_messages = 2 * (K - 1)  # Peripheral to center + center to peripheral
    else:
        raise ValueError(
            f"Unknown topology type: {topology_type}. "
            f"This script supports only 'ring' and 'star'."
        )
    
    # Size of W matrix in MB
    # W is flatten_dim × flatten_dim, using float16 (2 bytes per parameter)
    w_size_params = flatten_dim * flatten_dim
    w_size_bytes = w_size_params * 2  # float16 = 2 bytes
    w_size_mb = w_size_bytes / (1024 * 1024)
    
    # Total communication cost per round
    total_mb_per_round = num_messages * w_size_mb
    
    return num_messages, w_size_mb, total_mb_per_round

# TRAINING WITH TOPOLOGY

def adaptive_consensus_rate(round_idx, initial_alpha=0.5, min_alpha=0.1, decay_rate=0.02):
    return max(min_alpha, initial_alpha * math.exp(-decay_rate * round_idx))

def train_one_round_with_topology(client_models, client_optimizers, client_schedulers, 
                                   client_loaders, round_idx, topology_type, 
                                   consensus=True, **topology_kwargs):
    """Training function that works with any topology"""
    K = len(client_models)
    local_correct = [0] * K
    local_total = [0] * K
    local_losses = [0.0] * K
    loss_fn = nn.CrossEntropyLoss()

    # Local training phase (IDENTICAL for all topologies)
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

    # Consensus phase (TOPOLOGY-DEPENDENT)
    if consensus and round_idx > warmup_rounds:
        current_alpha = adaptive_consensus_rate(round_idx - warmup_rounds, α, 0.1)
        W_updates = [model.W.clone().detach() for model in client_models]

        for k in range(K):
            neighbors = get_topology_neighbors(k, K, topology_type, **topology_kwargs)
            
            if len(neighbors) == 0:  # Unpaired client
                continue
                
            consensus_W = (1 - current_alpha) * W_updates[k]
            for n in neighbors:
                consensus_W += (current_alpha / len(neighbors)) * W_updates[n]
            client_models[k].W.data = consensus_W

    accs = [100.0 * c / t for c, t in zip(local_correct, local_total)]
    global_acc = sum(accs) / K

    return global_acc, local_losses

def run_training_with_topology(client_models, client_optimizers, client_schedulers, 
                                train_loaders, num_rounds, topology_type, 
                                verbose=False, **topology_kwargs):
    round_accs = []
    round_losses = []
    
    if verbose:
        print(f"Training with {topology_type} topology...")
    
    for t in range(1, num_rounds + 1):
        global_acc, losses = train_one_round_with_topology(
            client_models, client_optimizers, client_schedulers, 
            train_loaders, t, topology_type, **topology_kwargs
        )
        round_accs.append(global_acc)
        round_losses.append(np.mean(losses))
        
        if verbose and t % 20 == 0:
            print(f"[Round {t}] Topology: {topology_type}, Accuracy: {global_acc:.2f}%")
    
    return round_accs, round_losses

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

# EXPERIMENT RUNNER

def run_experiment_for_topology(num_clients, topology_type, verbose=False, **topology_kwargs):
    """Run complete experiment for a specific topology"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"TOPOLOGY: {topology_type.upper()} | CLIENTS: {num_clients}")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    # Data preparation (SAME for all)
    train_splits, train_targets, test_splits, test_targets = load_mnist_vertically_partitioned(num_clients, augment=True)
    train_loaders, test_loaders = prepare_client_dataloaders(train_splits, train_targets, test_splits, test_targets, batch_size)

    # Model initialization (SAME for all)
    input_shape = train_splits[0][0].shape  # (C, H, W)
    flatten_dim = input_shape[1] * input_shape[2]  # H × W
    
    client_models = [EnhancedClientModel(input_shape, gnn_hidden_dims=[64, 32], mlp_hidden=64).to(device)
                     for _ in range(num_clients)]

    # Optimizer setup (SAME for all)
    client_optimizers = [torch.optim.AdamW(model.parameters(), lr=η_initial, weight_decay=1e-4)
                        for model in client_models]
    client_schedulers = [CosineAnnealingWarmup(opt, warmup_epochs=warmup_rounds, max_epochs=T, eta_min=η_min)
                        for opt in client_optimizers]

    # Training (TOPOLOGY-SPECIFIC)
    round_accs, round_losses = run_training_with_topology(
        client_models, client_optimizers, client_schedulers, 
        train_loaders, T, topology_type, verbose=verbose, **topology_kwargs
    )

    # Evaluation
    individual_accuracies, individual_f1s, _, _ = evaluate_clients_with_f1(client_models, test_loaders)
    ensemble_accuracy, ensemble_f1 = evaluate_ensemble_with_f1(client_models, test_loaders)
    
    avg_individual_accuracy = sum(individual_accuracies) / len(individual_accuracies)
    avg_individual_f1 = sum(individual_f1s) / len(individual_f1s)
    
    training_time = time.time() - start_time
    
    # Communication cost with ACTUAL data size
    num_messages, mb_per_message, total_mb_per_round = calculate_communication_cost(
        num_clients, flatten_dim, topology_type, **topology_kwargs
    )
    total_comm_cost_all_rounds = total_mb_per_round * (T - warmup_rounds)
    
    if verbose:
        print(f"\n📊 Results:")
        print(f"   Ensemble Accuracy: {ensemble_accuracy:.2f}%")
        print(f"   Ensemble F1: {ensemble_f1:.2f}%")
        print(f"   Avg Individual Accuracy: {avg_individual_accuracy:.2f}%")
        print(f"   flatten_dim: {flatten_dim}, W size: {mb_per_message:.4f} MB")
        print(f"   Messages/round: {num_messages}, Total MB/round: {total_mb_per_round:.2f} MB")
        print(f"   Total comm cost (all rounds): {total_comm_cost_all_rounds:.2f} MB")
        print(f"   Training Time: {training_time:.1f}s")
    
    return {
        'topology': topology_type,
        'num_clients': num_clients,
        'flatten_dim': flatten_dim,
        'avg_individual_accuracy': avg_individual_accuracy,
        'avg_individual_f1': avg_individual_f1,
        'ensemble_accuracy': ensemble_accuracy,
        'ensemble_f1': ensemble_f1,
        'training_time': training_time,
        'num_messages_per_round': num_messages,
        'mb_per_message': mb_per_message,
        'comm_cost_mb_per_round': total_mb_per_round,
        'total_comm_cost_mb': total_comm_cost_all_rounds,
        'round_accuracies': round_accs,
        'round_losses': round_losses,
        'individual_accuracies': individual_accuracies,
        'individual_f1s': individual_f1s
    }

# VISUALIZATION FUNCTIONS

def visualize_topology(num_clients, topology_type, **topology_kwargs):
    """Visualize network topology structure"""
    G = nx.Graph()
    G.add_nodes_from(range(num_clients))
    
    # Edges based on topology
    for k in range(num_clients):
        neighbors = get_topology_neighbors(k, num_clients, topology_type, **topology_kwargs)
        for n in neighbors:
            if k < n:  # No duplicate edges
                G.add_edge(k, n)
    
    plt.figure(figsize=(8, 6))
    
    # Layout based on topology
    if topology_type == 'ring':
        pos = nx.circular_layout(G)
    elif topology_type == 'star':
        pos = nx.spring_layout(G, center=[0, 0], k=2)
        pos[0] = np.array([0, 0])  # Center node at origin
    else:
        pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Network
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=800, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray')
    
    plt.title(f'{topology_type.upper()} Topology ({num_clients} Clients)', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return plt.gcf()

def plot_comparison_results(results_df):
    """Create comprehensive comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('VFL-GNN Topology Comparison', fontsize=16, fontweight='bold')
    
    topologies = results_df['Topology'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(topologies)))
    
    # 1. Ensemble Accuracy Comparison
    ax = axes[0, 0]
    for i, topo in enumerate(topologies):
        data = results_df[results_df['Topology'] == topo]
        ax.plot(data['Clients'], data['Ensemble_Accuracy'], 
               marker='o', label=topo, linewidth=2, color=colors[i])
    ax.set_xlabel('Number of Clients', fontweight='bold')
    ax.set_ylabel('Ensemble Accuracy (%)', fontweight='bold')
    ax.set_title('Ensemble Accuracy vs Clients')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Ensemble F1 Score Comparison
    ax = axes[0, 1]
    for i, topo in enumerate(topologies):
        data = results_df[results_df['Topology'] == topo]
        ax.plot(data['Clients'], data['Ensemble_F1'], 
               marker='s', label=topo, linewidth=2, color=colors[i])
    ax.set_xlabel('Number of Clients', fontweight='bold')
    ax.set_ylabel('Ensemble F1 Score (%)', fontweight='bold')
    ax.set_title('Ensemble F1 Score vs Clients')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Communication Cost Comparison (MB per round)
    ax = axes[0, 2]
    for i, topo in enumerate(topologies):
        data = results_df[results_df['Topology'] == topo]
        ax.plot(data['Clients'], data['Comm_Cost_MB_Per_Round'], 
               marker='^', label=topo, linewidth=2, color=colors[i])
    ax.set_xlabel('Number of Clients', fontweight='bold')
    ax.set_ylabel('Communication Cost (MB/round)', fontweight='bold')
    ax.set_title('Communication Cost per Round')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 4. Training Time Comparison
    ax = axes[1, 0]
    for i, topo in enumerate(topologies):
        data = results_df[results_df['Topology'] == topo]
        ax.plot(data['Clients'], data['Training_Time'], 
               marker='d', label=topo, linewidth=2, color=colors[i])
    ax.set_xlabel('Number of Clients', fontweight='bold')
    ax.set_ylabel('Training Time (s)', fontweight='bold')
    ax.set_title('Training Time vs Clients')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Efficiency-Accuracy Tradeoff (Fixed clients)
    ax = axes[1, 1]
    fixed_clients = 5  # Choose a representative number
    fixed_data = results_df[results_df['Clients'] == fixed_clients]
    if not fixed_data.empty:
        scatter = ax.scatter(fixed_data['Comm_Cost_MB_Per_Round'], 
                           fixed_data['Ensemble_Accuracy'],
                           s=200, c=range(len(fixed_data)), 
                           cmap='viridis', alpha=0.7, edgecolors='black')
        for idx, row in fixed_data.iterrows():
            ax.annotate(row['Topology'], 
                       (row['Comm_Cost_MB_Per_Round'], row['Ensemble_Accuracy']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax.set_xlabel('Communication Cost (MB/round)', fontweight='bold')
        ax.set_ylabel('Ensemble Accuracy (%)', fontweight='bold')
        ax.set_title(f'Efficiency-Accuracy Tradeoff ({fixed_clients} Clients)')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    
    # 6. Total Communication Cost (MB)
    ax = axes[1, 2]
    width = 0.15
    x = np.arange(len(results_df['Clients'].unique()))
    for i, topo in enumerate(topologies):
        data = results_df[results_df['Topology'] == topo]
        ax.bar(x + i*width, data['Total_Comm_Cost_MB'], 
              width, label=topo, color=colors[i], alpha=0.8)
    ax.set_xlabel('Number of Clients', fontweight='bold')
    ax.set_ylabel('Total Communication Cost (MB)', fontweight='bold')
    ax.set_title('Total Communication Cost (All Rounds)')
    ax.set_xticks(x + width * (len(topologies)-1) / 2)
    ax.set_xticklabels(results_df['Clients'].unique())
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_convergence_curves(all_results, num_clients=5):
    """Plot convergence curves for different topologies"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter results for specific number of clients
    filtered_results = [r for r in all_results if r['num_clients'] == num_clients]
    
    if not filtered_results:
        print(f"No results found for {num_clients} clients")
        return None
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(filtered_results)))
    
    # Accuracy convergence
    ax = axes[0]
    for i, result in enumerate(filtered_results):
        rounds = range(1, len(result['round_accuracies']) + 1)
        ax.plot(rounds, result['round_accuracies'], 
               label=result['topology'], linewidth=2, color=colors[i])
    ax.set_xlabel('Training Round', fontweight='bold')
    ax.set_ylabel('Training Accuracy (%)', fontweight='bold')
    ax.set_title(f'Convergence Curves ({num_clients} Clients)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss convergence
    ax = axes[1]
    for i, result in enumerate(filtered_results):
        rounds = range(1, len(result['round_losses']) + 1)
        ax.plot(rounds, result['round_losses'], 
               label=result['topology'], linewidth=2, color=colors[i])
    ax.set_xlabel('Training Round', fontweight='bold')
    ax.set_ylabel('Training Loss', fontweight='bold')
    ax.set_title(f'Loss Curves ({num_clients} Clients)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_efficiency_heatmap(results_df):
    """Create heatmap showing efficiency metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    topologies = results_df['Topology'].unique()
    clients = sorted(results_df['Clients'].unique())
    
    # Accuracy heatmap
    acc_matrix = np.zeros((len(topologies), len(clients)))
    for i, topo in enumerate(topologies):
        for j, client in enumerate(clients):
            val = results_df[(results_df['Topology'] == topo) & 
                           (results_df['Clients'] == client)]['Ensemble_Accuracy'].values
            if len(val) > 0:
                acc_matrix[i, j] = val[0]
    
    sns.heatmap(acc_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
               xticklabels=clients, yticklabels=topologies,
               ax=axes[0], cbar_kws={'label': 'Accuracy (%)'})
    axes[0].set_title('Ensemble Accuracy Heatmap', fontweight='bold')
    axes[0].set_xlabel('Number of Clients', fontweight='bold')
    axes[0].set_ylabel('Topology', fontweight='bold')
    
    # Communication cost heatmap (in MB)
    comm_matrix = np.zeros((len(topologies), len(clients)))
    for i, topo in enumerate(topologies):
        for j, client in enumerate(clients):
            val = results_df[(results_df['Topology'] == topo) & 
                           (results_df['Clients'] == client)]['Comm_Cost_MB_Per_Round'].values
            if len(val) > 0:
                comm_matrix[i, j] = val[0]
    
    sns.heatmap(comm_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
               xticklabels=clients, yticklabels=topologies,
               ax=axes[1], cbar_kws={'label': 'MB/Round'})
    axes[1].set_title('Communication Cost Heatmap', fontweight='bold')
    axes[1].set_xlabel('Number of Clients', fontweight='bold')
    axes[1].set_ylabel('Topology', fontweight='bold')
    
    plt.tight_layout()
    return fig

# ==================== COMPREHENSIVE EVALUATION ====================

def run_comprehensive_topology_comparison(client_configs=[3, 5, 7], save_results=True):
    """
    Comparison of all topologies
    
    Args:
        client_configs: List of client numbers to test
        save_results: Whether to save results and plots
    """
    print("="*80)
    print(" COMPREHENSIVE VFL-GNN TOPOLOGY COMPARISON")
    print("="*80)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Client configurations: {client_configs}")
    print(f"Training rounds: {T}")
    print("="*80)
    
    all_results = []
    topologies_to_test = ['ring', 'star']
    
    for num_clients in client_configs:
        print(f"\n{'='*80}")
        print(f"📊 TESTING WITH {num_clients} CLIENTS")
        print(f"{'='*80}")
        
        for topology in topologies_to_test:
            # Ring and Star require no extra topology-specific parameters
            topology_kwargs = {}
            
            # Run experiment
            result = run_experiment_for_topology(
                num_clients, topology, verbose=True, **topology_kwargs
            )
            all_results.append(result)
            
            print(f"{topology.upper()} completed")
            time.sleep(1)
    
    # Results DataFrame
    results_df = pd.DataFrame([
        {
            'Topology': r['topology'],
            'Clients': r['num_clients'],
            'Flatten_Dim': r['flatten_dim'],
            'Ensemble_Accuracy': r['ensemble_accuracy'],
            'Ensemble_F1': r['ensemble_f1'],
            'Avg_Individual_Accuracy': r['avg_individual_accuracy'],
            'Avg_Individual_F1': r['avg_individual_f1'],
            'Training_Time': r['training_time'],
            'Num_Messages_Per_Round': r['num_messages_per_round'],
            'MB_Per_Message': r['mb_per_message'],
            'Comm_Cost_MB_Per_Round': r['comm_cost_mb_per_round'],
            'Total_Comm_Cost_MB': r['total_comm_cost_mb']
        } for r in all_results
    ])
    
    # Results table
    print("\n" + "="*120)
    print(" COMPREHENSIVE RESULTS TABLE")
    print("="*120)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*120)
    print(" KEY INSIGHTS")
    print("="*120)
    
    for num_clients in client_configs:
        client_results = results_df[results_df['Clients'] == num_clients]
        print(f"{num_clients} Clients:")
        
        # Best accuracy
        best_acc_idx = client_results['Ensemble_Accuracy'].idxmax()
        best_acc = client_results.loc[best_acc_idx]
        print(f" Best Accuracy: {best_acc['Topology']} ({best_acc['Ensemble_Accuracy']:.2f}%)")
        
        # Most efficient (lowest comm cost with reasonable accuracy)
        client_results_copy = client_results.copy()
        client_results_copy['Efficiency_Score'] = (
            client_results_copy['Ensemble_Accuracy'] / 
            np.log1p(client_results_copy['Comm_Cost_MB_Per_Round'])
        )
        best_eff_idx = client_results_copy['Efficiency_Score'].idxmax()
        best_eff = client_results.loc[best_eff_idx]
        print(f" Most Efficient: {best_eff['Topology']} " +
              f"(Acc: {best_eff['Ensemble_Accuracy']:.2f}%, " +
              f"Cost: {best_eff['Comm_Cost_MB_Per_Round']:.2f} MB/round)")
        
        # Ring vs Star comparison
        ring_result = client_results[client_results['Topology'] == 'ring']
        star_result = client_results[client_results['Topology'] == 'star']
        if not ring_result.empty and not star_result.empty:
            ring_acc = ring_result['Ensemble_Accuracy'].values[0]
            ring_cost = ring_result['Comm_Cost_MB_Per_Round'].values[0]
            star_acc = star_result['Ensemble_Accuracy'].values[0]
            star_cost = star_result['Comm_Cost_MB_Per_Round'].values[0]
            acc_diff = ring_acc - star_acc
            if star_cost > 0:
                cost_diff = (ring_cost - star_cost) / star_cost * 100
            else:
                cost_diff = 0.0
            print(f" Ring vs Star: {acc_diff:+.2f}% accuracy, " +
                  f"{cost_diff:+.1f}% communication cost")
    
    if save_results:
        # Save results table
        results_df.to_csv('topology_comparison_results_corrected.csv', index=False)
        print("Results saved to 'topology_comparison_results_corrected.csv'")
        
        # Create and save visualizations
        print("Generating visualizations...")
        
        # 1. Topology structure visualizations
        num_topos = len(topologies_to_test)
        fig_topos = plt.figure(figsize=(5 * num_topos, 4))
        for i, topo in enumerate(topologies_to_test):
            plt.subplot(1, num_topos, i+1)
            
            G = nx.Graph()
            G.add_nodes_from(range(5))
            for k in range(5):
                neighbors = get_topology_neighbors(k, 5, topo)
                for n in neighbors:
                    if k < n:
                        G.add_edge(k, n)
            
            if topo == 'ring':
                pos = nx.circular_layout(G)
            elif topo == 'star':
                pos = nx.spring_layout(G, center=[0, 0], k=2)
                pos[0] = np.array([0, 0])
            else:
                pos = nx.spring_layout(G, k=1, iterations=50)
            
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                  node_size=500, alpha=0.9)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray')
            plt.title(topo.upper(), fontsize=12, fontweight='bold')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('topology_structures_corrected.png', dpi=300, bbox_inches='tight')
        print(" Saved: topology_structures_corrected.png")
        plt.close()
        
        # 2. Comparison plots
        fig = plot_comparison_results(results_df)
        fig.savefig('topology_comparison_plots_corrected.png', dpi=300, bbox_inches='tight')
        print(" Saved: topology_comparison_plots_corrected.png")
        plt.close()
        
        # 3. Convergence curves (for middle client config)
        mid_idx = len(client_configs) // 2
        fig = plot_convergence_curves(all_results, num_clients=client_configs[mid_idx])
        if fig:
            fig.savefig('topology_convergence_curves_corrected.png', dpi=300, bbox_inches='tight')
            print(" Saved: topology_convergence_curves_corrected.png")
            plt.close()
        
        # 4. Efficiency heatmap
        fig = create_efficiency_heatmap(results_df)
        fig.savefig('topology_efficiency_heatmap_corrected.png', dpi=300, bbox_inches='tight')
        print(" Saved: topology_efficiency_heatmap_corrected.png")
        plt.close()
    
    print("\n" + "="*120)
    print(" TOPOLOGY COMPARISON COMPLETE!")
    print("="*120)
    
    return all_results, results_df

# ==================== MAIN ====================

if __name__ == "__main__":
    # Testing with 3, 5, 7, 9 clients
    results, results_table = run_comprehensive_topology_comparison(
        client_configs=[3, 5, 7, 9],
        save_results=True
    )
    
    print(" All experiments completed successfully!")
    