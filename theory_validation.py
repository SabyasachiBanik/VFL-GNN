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
from scipy.stats import pearsonr, t as t_dist
import os
from typing import Dict, List, Tuple
import warnings
import pickle
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# SEEDING
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#CONFIG
T = 200  # Training rounds
η_initial = 0.01
η_min = 0.001
α = 0.5  # Initial consensus rate
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
warmup_rounds = 5


SEEDS = [43]  # 1 seeds 
K_VALUES = [6, 7, 8, 9, 10]  # All K values

# Checkpoint directory
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# CHECKPOINT MANAGEMENT
def save_experiment_result(result: Dict, K: int, seed: int):
    
    checkpoint_file = f'{CHECKPOINT_DIR}/K{K}_seed{seed}.pkl'
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"   💾 Saved checkpoint: K{K}_seed{seed}.pkl")

def load_experiment_result(K: int, seed: int) -> Dict:
    
    checkpoint_file = f'{CHECKPOINT_DIR}/K{K}_seed{seed}.pkl'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_aggregated_results(all_results: Dict):
    
    checkpoint_file = f'{CHECKPOINT_DIR}/aggregated_results.pkl'
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # JSON
    json_file = f'{CHECKPOINT_DIR}/aggregated_results.json'
    json_safe = {}
    for K, data in all_results.items():
        json_safe[str(K)] = {
            'K': data['K'],
            'n_seeds': data['n_seeds'],
            'accuracy_mean': float(data['accuracy']['mean']),
            'accuracy_std': float(data['accuracy']['std']),
            'f1_mean': float(data['f1']['mean']),
            'consensus_errors_mean': data['consensus_errors']['mean'].tolist()[:10],  # First 10 rounds
            'rounds': data['rounds'][:10]
        }
    
    with open(json_file, 'w') as f:
        json.dump(json_safe, f, indent=2)
    
    print(f"\n💾 CHECKPOINT: Saved aggregated results for K={list(all_results.keys())}")

def load_aggregated_results() -> Dict:
    
    checkpoint_file = f'{CHECKPOINT_DIR}/aggregated_results.pkl'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return {}

def get_completed_experiments() -> Dict[int, List[int]]:
    
    completed = {}
    for K in K_VALUES:
        completed[K] = []
        for seed in SEEDS:
            if load_experiment_result(K, seed) is not None:
                completed[K].append(seed)
    return completed

# DATA LOADING (unchanged using F-MNIST)
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
    """Data augmentation"""
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

class ClientDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def prepare_client_dataloaders(train_splits, train_targets, test_splits, test_targets, batch_size=64):
    """Prepare data loaders for all clients"""
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

# MODEL COMPONENTS (unchanged)
class AttentionGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(AttentionGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert output_dim % num_heads == 0
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
    """Create spatial prior adjacency matrix"""
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

# TRAINING FUNCTIONS (unchanged) 
def get_ring_neighbors(k, K):
    return [(k - 1) % K, (k + 1) % K]

def adaptive_consensus_rate(round_idx, initial_alpha=0.5, min_alpha=0.1, decay_rate=0.02):
    return max(min_alpha, initial_alpha * math.exp(-decay_rate * round_idx))

def compute_consensus_error(W_list):
    """Compute consensus error: ε_t = (1/K) Σ ||W^(k)_t - W̄_t||²"""
    K = len(W_list)
    W_avg = sum(W_list) / K
    
    total_error = 0.0
    for k in range(K):
        diff = torch.norm(W_list[k] - W_avg, p='fro').item()**2
        total_error += diff
    
    consensus_error = total_error / K
    return consensus_error

def train_one_round(client_models, client_optimizers, client_schedulers, client_loaders, 
                    test_loaders, round_idx, consensus=True):
    """Single training round with consensus error tracking"""
    K = len(client_models)
    loss_fn = nn.CrossEntropyLoss()

    # Local training
    for k in range(K):
        model = client_models[k]
        model.train()
        optimizer = client_optimizers[k]
        scheduler = client_schedulers[k]
        loader = client_loaders[k]

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

        scheduler.step(round_idx)

    # consensus error
    W_list = [model.W.clone().detach().cpu() for model in client_models]
    W_avg = sum(W_list) / K
    consensus_error = compute_consensus_error(W_list)
    test_acc = evaluate_ensemble_accuracy_only(client_models, test_loaders)
    
    # Consensus update
    if consensus and round_idx > warmup_rounds:
        current_alpha = adaptive_consensus_rate(round_idx - warmup_rounds, α, 0.1)
        W_updates = [model.W.clone().detach() for model in client_models]
        for k in range(K):
            neighbors = get_ring_neighbors(k, K)
            consensus_W = (1 - current_alpha) * W_updates[k]
            for n in neighbors:
                consensus_W += (current_alpha / len(neighbors)) * W_updates[n]
            client_models[k].W.data = consensus_W
    else:
        current_alpha = 0.0
    
    return {
        'W_avg': W_avg,
        'consensus_error': consensus_error,
        'test_accuracy': test_acc,
        'alpha': current_alpha
    }

def run_training_with_tracking(client_models, client_optimizers, client_schedulers, 
                                train_loaders, test_loaders, num_rounds=T, verbose=False):
    """Full training loop with consensus error tracking"""
    tracking_data = {
        'consensus_errors': [],
        'test_accuracies': [],
        'alpha_history': [],
        'rounds': []
    }
    
    for t in range(1, num_rounds + 1):
        round_data = train_one_round(
            client_models, client_optimizers, client_schedulers, 
            train_loaders, test_loaders, t
        )
        
        tracking_data['consensus_errors'].append(round_data['consensus_error'])
        tracking_data['test_accuracies'].append(round_data['test_accuracy'])
        tracking_data['alpha_history'].append(round_data['alpha'])
        tracking_data['rounds'].append(t)
        
        if verbose and t % 50 == 0:
            print(f"[Round {t:3d}] Acc: {round_data['test_accuracy']:.2f}% | "
                  f"Consensus ε: {round_data['consensus_error']:.3f} | α: {round_data['alpha']:.3f}")
    
    if verbose:
        initial_error = tracking_data['consensus_errors'][0]
        final_error = tracking_data['consensus_errors'][-1]
        reduction = (1 - final_error / initial_error) * 100
        
        print(f"\n✅ Training complete!")
        print(f"   Final accuracy: {tracking_data['test_accuracies'][-1]:.2f}%")
        print(f"   Consensus error: {initial_error:.3f} → {final_error:.3f}")
        print(f"   Error reduction: {reduction:.1f}%")
    
    return tracking_data

@torch.no_grad()
def evaluate_ensemble_accuracy_only(client_models, client_test_loaders):
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
    return 100.0 * correct / total

@torch.no_grad()
def evaluate_ensemble_with_f1(client_models, client_test_loaders):
    """Full evaluation with F1"""
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

# SINGLE EXPERIMENT WITH CHECKPOINT
def run_single_experiment(num_clients: int, seed: int, verbose: bool = False) -> Dict:
    """Run single experiment with checkpoint save"""
    
    existing_result = load_experiment_result(num_clients, seed)
    if existing_result is not None:
        print(f"   ♻️  LOADED from checkpoint (already complete)")
        return existing_result
    
    set_seed(seed)
    
    # Data
    train_splits, train_targets, test_splits, test_targets = load_mnist_vertically_partitioned(
        num_clients, augment=True
    )
    train_loaders, test_loaders = prepare_client_dataloaders(
        train_splits, train_targets, test_splits, test_targets, batch_size
    )

    # Models
    input_shape = train_splits[0][0].shape
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

    # Train
    start_time = time.time()
    tracking_data = run_training_with_tracking(
        client_models, client_optimizers, client_schedulers, 
        train_loaders, test_loaders, num_rounds=T, verbose=verbose
    )
    training_time = time.time() - start_time

    # Final eval
    ensemble_accuracy, ensemble_f1 = evaluate_ensemble_with_f1(client_models, test_loaders)
    
    result = {
        'K': num_clients,
        'seed': seed,
        'ensemble_accuracy': ensemble_accuracy,
        'ensemble_f1': ensemble_f1,
        'training_time': training_time,
        'tracking_data': tracking_data
    }
    
    
    save_experiment_result(result, num_clients, seed)
    
    if verbose:
        print(f"✓ Final: Acc={ensemble_accuracy:.2f}%, F1={ensemble_f1:.2f}%, Time={training_time:.1f}s")
    
    return result

# MULTI-SEED AGGREGATION (unchanged)
def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, Tuple[float, float]]:
    """Compute mean, std, and confidence interval"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1) if n > 1 else 0.0
    
    if n > 1:
        t_crit = t_dist.ppf((1 + confidence) / 2, n - 1)
        margin = t_crit * (std / np.sqrt(n))
    else:
        margin = 0.0
    
    ci_lower = mean - margin
    ci_upper = mean + margin
    
    return mean, std, (ci_lower, ci_upper)

def aggregate_multi_seed_results(results: List[Dict]) -> Dict:
    """Aggregate results across seeds for one K value"""
    K = results[0]['K']
    
    accuracies = [r['ensemble_accuracy'] for r in results]
    f1_scores = [r['ensemble_f1'] for r in results]
    training_times = [r['training_time'] for r in results]
    
    acc_mean, acc_std, acc_ci = compute_confidence_interval(accuracies)
    f1_mean, f1_std, f1_ci = compute_confidence_interval(f1_scores)
    time_mean, time_std, time_ci = compute_confidence_interval(training_times)
    
    consensus_errors_all_seeds = []
    test_accs_all_seeds = []
    
    for result in results:
        consensus_errors_all_seeds.append(result['tracking_data']['consensus_errors'])
        test_accs_all_seeds.append(result['tracking_data']['test_accuracies'])
    
    consensus_errors_all = np.array(consensus_errors_all_seeds)
    test_accs_all = np.array(test_accs_all_seeds)
    
    consensus_mean = np.mean(consensus_errors_all, axis=0)
    consensus_std = np.std(consensus_errors_all, axis=0, ddof=1) if len(results) > 1 else np.zeros_like(consensus_mean)
    
    acc_traj_mean = np.mean(test_accs_all, axis=0)
    acc_traj_std = np.std(test_accs_all, axis=0, ddof=1) if len(results) > 1 else np.zeros_like(acc_traj_mean)
    
    n_seeds = len(results)
    if n_seeds > 1:
        t_crit = t_dist.ppf(0.975, n_seeds - 1)
        consensus_ci = t_crit * consensus_std / np.sqrt(n_seeds)
        acc_traj_ci = t_crit * acc_traj_std / np.sqrt(n_seeds)
    else:
        consensus_ci = np.zeros_like(consensus_mean)
        acc_traj_ci = np.zeros_like(acc_traj_mean)
    
    return {
        'K': K,
        'n_seeds': n_seeds,
        'accuracy': {
            'mean': acc_mean,
            'std': acc_std,
            'ci': acc_ci,
            'all': accuracies
        },
        'f1': {
            'mean': f1_mean,
            'std': f1_std,
            'ci': f1_ci,
            'all': f1_scores
        },
        'training_time': {
            'mean': time_mean,
            'std': time_std,
            'ci': time_ci
        },
        'consensus_errors': {
            'mean': consensus_mean,
            'std': consensus_std,
            'ci': consensus_ci,
            'all_seeds': consensus_errors_all
        },
        'acc_trajectory': {
            'mean': acc_traj_mean,
            'std': acc_traj_std,
            'ci': acc_traj_ci,
            'all_seeds': test_accs_all
        },
        'rounds': results[0]['tracking_data']['rounds']
    }

# MULTI-K MULTI-SEED WITH RESUME
def run_multi_k_multi_seed_experiments(k_values: List[int] = K_VALUES, 
                                       seeds: List[int] = SEEDS,
                                       verbose: bool = True) -> Dict:
    """experiments with checkpoint resume capability"""
    
    # existing results
    all_results = load_aggregated_results()
    completed = get_completed_experiments()
    
    print("\n" + "="*80)
    print("🚀 VFL-GNN MULTI-K MULTI-SEED WITH AUTO-SAVE CHECKPOINTS")
    print("="*80)
    print(f"K values: {k_values}")
    print(f"Seeds: {seeds} ({len(seeds)} seed(s))")
    print(f"Total experiments: {len(k_values) * len(seeds)}")
    print(f"Rounds per experiment: {T}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    
    
    total_completed = sum(len(completed[k]) for k in k_values)
    total_needed = len(k_values) * len(seeds)
    print(f"\n📊 Progress: {total_completed}/{total_needed} experiments completed")
    for K in k_values:
        print(f"   K={K}: {len(completed[K])}/{len(seeds)} seeds complete {completed[K]}")
    print("="*80 + "\n")
    
    for K in k_values:
        print(f"\n{'='*80}")
        print(f"📊 Running experiments for K={K}")
        print(f"{'='*80}")
        
        # Load existing results for this K if available
        k_results = []
        for seed in seeds:
            existing = load_experiment_result(K, seed)
            if existing:
                k_results.append(existing)
        
        # missing experiments
        for seed_idx, seed in enumerate(seeds, 1):
            if seed in completed[K]:
                print(f"\n🌱 Seed {seed_idx}/{len(seeds)} (seed={seed})")
                existing = load_experiment_result(K, seed)
                print(f"   ♻️  SKIPPED (loaded from checkpoint)")
                print(f"   ✓ Acc: {existing['ensemble_accuracy']:.2f}%, F1: {existing['ensemble_f1']:.2f}%")
            else:
                print(f"\n🌱 Seed {seed_idx}/{len(seeds)} (seed={seed}) - RUNNING NOW...")
                
                result = run_single_experiment(K, seed, verbose=False)
                k_results.append(result)
                
                print(f"   ✓ Acc: {result['ensemble_accuracy']:.2f}%, F1: {result['ensemble_f1']:.2f}%, Time: {result['training_time']:.1f}s")
        
        # Aggregated results for this K
        aggregated = aggregate_multi_seed_results(k_results)
        all_results[K] = aggregated
        
        
        save_aggregated_results(all_results)
        
        print(f"\n📈 Aggregated results for K={K}:")
        print(f"   Accuracy: {aggregated['accuracy']['mean']:.2f}% ± {aggregated['accuracy']['std']:.2f}%")
        if aggregated['n_seeds'] > 1:
            print(f"   95% CI: [{aggregated['accuracy']['ci'][0]:.2f}%, {aggregated['accuracy']['ci'][1]:.2f}%]")
        print(f"   F1 Score: {aggregated['f1']['mean']:.2f}% ± {aggregated['f1']['std']:.2f}%")
    
    return all_results

# PLOTTING 

def plot_multi_k_convergence_with_ci(all_results: Dict, save_dir: str = './plots'):
    """Plot consensus error convergence with CI"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    # Plot 1: Consensus Error
    ax = axes[0, 0]
    for idx, (K, data) in enumerate(sorted(all_results.items())):
        rounds = np.array(data['rounds'])
        mean = data['consensus_errors']['mean']
        ci = data['consensus_errors']['ci']
        
        ax.semilogy(rounds, mean, '-', linewidth=2.5, color=colors[idx], 
                   label=f'K={K} (Empirical)', alpha=0.9, zorder=3)
        if data['n_seeds'] > 1:
            ax.fill_between(rounds, mean - ci, mean + ci, 
                            color=colors[idx], alpha=0.2, zorder=2)
        
        # Theory line
        alpha_steady = 0.1
        lambda_2 = np.cos(2 * np.pi / K)
        rho_theory = 1 - (alpha_steady * (1 - lambda_2) / 2)
        
        warmup = warmup_rounds
        if len(mean) > warmup:
            epsilon_0 = mean[warmup]
            rounds_theory = rounds[warmup:]
            theory_pred = epsilon_0 * (rho_theory ** (rounds_theory - warmup))
            ax.semilogy(rounds_theory, theory_pred, '--', linewidth=1.5, 
                       color=colors[idx], alpha=0.6, 
                       label=f'K={K} (Theory: ρ={rho_theory:.3f})', zorder=1)
    
    ax.set_xlabel('Training Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('Consensus Error $\\varepsilon_t$', fontsize=13, fontweight='bold')
    n_seeds_text = f"n={all_results[list(all_results.keys())[0]]['n_seeds']} seed(s)"
    ax.set_title(f'Theorem 4 Validation: Consensus Error Convergence\n(Mean ± 95% CI, {n_seeds_text})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Log-Linear
    ax = axes[0, 1]
    for idx, (K, data) in enumerate(sorted(all_results.items())):
        rounds = np.array(data['rounds'])
        mean = data['consensus_errors']['mean']
        ci = data['consensus_errors']['ci']
        log_mean = np.log(mean + 1e-10)
        
        ax.plot(rounds, log_mean, '-', linewidth=2.5, color=colors[idx], 
               label=f'K={K}', alpha=0.9)
        if data['n_seeds'] > 1:
            log_ci = ci / (mean + 1e-10)
            ax.fill_between(rounds, log_mean - log_ci, log_mean + log_ci, 
                            color=colors[idx], alpha=0.2)
    
    ax.set_xlabel('Training Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('log(Consensus Error)', fontsize=13, fontweight='bold')
    ax.set_title('Log-Linear View', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy
    ax = axes[1, 0]
    for idx, (K, data) in enumerate(sorted(all_results.items())):
        rounds = np.array(data['rounds'])
        mean = data['acc_trajectory']['mean']
        ci = data['acc_trajectory']['ci']
        
        ax.plot(rounds, mean, '-', linewidth=2.5, color=colors[idx], 
               label=f'K={K}', alpha=0.9)
        if data['n_seeds'] > 1:
            ax.fill_between(rounds, mean - ci, mean + ci, 
                            color=colors[idx], alpha=0.2)
    
    ax.set_xlabel('Training Round', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Test Accuracy Trajectory\n(Mean ± 95% CI, {n_seeds_text})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Convergence Rate
    ax = axes[1, 1]
    k_vals = sorted(all_results.keys())
    rho_empirical = []
    rho_theory = []
    
    for K in k_vals:
        data = all_results[K]
        warmup = warmup_rounds
        rounds_fit = np.array(data['rounds'][warmup:])
        errors_fit = data['consensus_errors']['mean'][warmup:]
        
        if len(errors_fit) > 0 and np.all(errors_fit > 0):
            log_errors = np.log(errors_fit)
            coeffs = np.polyfit(rounds_fit - warmup, log_errors, 1)
            rho_emp = np.exp(coeffs[0])
            rho_empirical.append(rho_emp)
        else:
            rho_empirical.append(0.99)
        
        alpha_steady = 0.1
        lambda_2 = np.cos(2 * np.pi / K)
        rho_th = 1 - (alpha_steady * (1 - lambda_2) / 2)
        rho_theory.append(rho_th)
    
    x = np.arange(len(k_vals))
    width = 0.35
    
    ax.bar(x - width/2, rho_theory, width, label='Theory (Theorem 4)',
           color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.bar(x + width/2, rho_empirical, width, label='Empirical (Fitted)',
           color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in k_vals])
    ax.set_ylabel('Convergence Rate ρ', fontsize=13, fontweight='bold')
    ax.set_title('Convergence Rate Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.90, 1.0])
    
    for i, (th, emp) in enumerate(zip(rho_theory, rho_empirical)):
        ax.text(i - width/2, th + 0.003, f'{th:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width/2, emp + 0.003, f'{emp:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'VFL-GNN: Theorem 4 Validation\n{n_seeds_text} per Configuration', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/theorem4_validation_multi_k.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved: {save_dir}/theorem4_validation_multi_k.png")
    plt.close()

def generate_results_table(all_results: Dict, save_dir: str = './plots'):
    """Generate results table"""
    os.makedirs(save_dir, exist_ok=True)
    
    table_data = []
    for K in sorted(all_results.keys()):
        data = all_results[K]
        
        alpha_steady = 0.1
        lambda_2 = np.cos(2 * np.pi / K)
        rho_theory = 1 - (alpha_steady * (1 - lambda_2) / 2)
        
        warmup = warmup_rounds
        rounds_fit = np.array(data['rounds'][warmup:])
        errors_fit = data['consensus_errors']['mean'][warmup:]
        
        if len(errors_fit) > 0 and np.all(errors_fit > 0):
            log_errors = np.log(errors_fit)
            coeffs = np.polyfit(rounds_fit - warmup, log_errors, 1)
            rho_empirical = np.exp(coeffs[0])
        else:
            rho_empirical = 0.99
        
        initial_consensus = data['consensus_errors']['mean'][0]
        final_consensus = data['consensus_errors']['mean'][-1]
        reduction = (1 - final_consensus / initial_consensus) * 100
        
        if data['n_seeds'] > 1:
            table_data.append({
                'K': K,
                'Accuracy (%)': f"{data['accuracy']['mean']:.2f} ± {data['accuracy']['std']:.2f}",
                '95% CI (Acc)': f"[{data['accuracy']['ci'][0]:.2f}, {data['accuracy']['ci'][1]:.2f}]",
                'F1 Score (%)': f"{data['f1']['mean']:.2f} ± {data['f1']['std']:.2f}",
                'Initial ε': f"{initial_consensus:.2f}",
                'Final ε': f"{final_consensus:.3f} ± {data['consensus_errors']['std'][-1]:.3f}",
                'ε Reduction (%)': f"{reduction:.1f}",
                'ρ (Theory)': f"{rho_theory:.4f}",
                'ρ (Empirical)': f"{rho_empirical:.4f}",
                'Training Time (s)': f"{data['training_time']['mean']:.1f} ± {data['training_time']['std']:.1f}"
            })
        else:
            table_data.append({
                'K': K,
                'Accuracy (%)': f"{data['accuracy']['mean']:.2f}",
                '95% CI (Acc)': "N/A",
                'F1 Score (%)': f"{data['f1']['mean']:.2f}",
                'Initial ε': f"{initial_consensus:.2f}",
                'Final ε': f"{final_consensus:.3f}",
                'ε Reduction (%)': f"{reduction:.1f}",
                'ρ (Theory)': f"{rho_theory:.4f}",
                'ρ (Empirical)': f"{rho_empirical:.4f}",
                'Training Time (s)': f"{data['training_time']['mean']:.1f}"
            })
    
    df = pd.DataFrame(table_data)
    
    print("\n" + "="*140)
    print("📊 FINAL RESULTS TABLE - THEOREM 4 VALIDATION")
    print("="*140)
    print(df.to_string(index=False))
    print("="*140)
    
    df.to_csv(f'{save_dir}/theorem4_validation_results.csv', index=False)
    print(f"\n✅ Saved: {save_dir}/theorem4_validation_results.csv")
    
    return df

# DATA EXPORT FOR LATEX 

def export_all_data_for_latex(all_results: Dict, save_dir: str = './latex_data'):
    """
    Export consensus error and accuracy data for LaTeX plotting
    Exports EVERY round, no skipping!
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("📊 EXPORTING DATA FOR LATEX (ALL ROUNDS)")
    print("="*80)
    
    # FORMAT 1: CONSENSUS ERROR 
    print("\n1️⃣ Exporting Consensus Error Data...")
    
    # Individual files per K
    for K in sorted(all_results.keys()):
        data = all_results[K]
        rounds = data['rounds']
        
        # Single seed case
        if data['n_seeds'] == 1:
            consensus_errors = data['consensus_errors']['mean']
            
            filename = f'{save_dir}/consensus_error_K{K}.txt'
            with open(filename, 'w') as f:
                f.write(f"# Consensus Error for K={K} (1 seed)\n")
                f.write(f"# Format: round error\n")
                f.write("round error\n")
                
                for r, err in zip(rounds, consensus_errors):
                    f.write(f"{r} {err:.8f}\n")
            
            print(f"   ✅ Saved: {filename}")
        
        # Multi-seed case
        else:
            mean_errors = data['consensus_errors']['mean']
            std_errors = data['consensus_errors']['std']
            ci_errors = data['consensus_errors']['ci']
            
            filename = f'{save_dir}/consensus_error_K{K}.txt'
            with open(filename, 'w') as f:
                f.write(f"# Consensus Error for K={K} ({data['n_seeds']} seeds)\n")
                f.write(f"# Format: round mean std ci_lower ci_upper\n")
                f.write("round mean std ci_lower ci_upper\n")
                
                for r, mean, std, ci in zip(rounds, mean_errors, std_errors, ci_errors):
                    ci_lower = mean - ci
                    ci_upper = mean + ci
                    f.write(f"{r} {mean:.8f} {std:.8f} {ci_lower:.8f} {ci_upper:.8f}\n")
            
            print(f"   ✅ Saved: {filename}")
    
    # Combined file (all K values)
    combined_consensus_file = f'{save_dir}/consensus_error_all_K.txt'
    with open(combined_consensus_file, 'w') as f:
        f.write(f"# Consensus Error for all K values\n")
        
        if all_results[list(all_results.keys())[0]]['n_seeds'] == 1:
            f.write(f"# Format: K round error\n")
            f.write("K round error\n")
            
            for K in sorted(all_results.keys()):
                data = all_results[K]
                rounds = data['rounds']
                errors = data['consensus_errors']['mean']
                
                for r, err in zip(rounds, errors):
                    f.write(f"{K} {r} {err:.8f}\n")
        else:
            f.write(f"# Format: K round mean std ci_lower ci_upper\n")
            f.write("K round mean std ci_lower ci_upper\n")
            
            for K in sorted(all_results.keys()):
                data = all_results[K]
                rounds = data['rounds']
                means = data['consensus_errors']['mean']
                stds = data['consensus_errors']['std']
                cis = data['consensus_errors']['ci']
                
                for r, mean, std, ci in zip(rounds, means, stds, cis):
                    ci_lower = mean - ci
                    ci_upper = mean + ci
                    f.write(f"{K} {r} {mean:.8f} {std:.8f} {ci_lower:.8f} {ci_upper:.8f}\n")
    
    print(f"   ✅ Saved: {combined_consensus_file}")
    
    # FORMAT 2: TEST ACCURACY
    print("\n2️⃣ Exporting Test Accuracy Data...")
    
    # Individual files per K
    for K in sorted(all_results.keys()):
        data = all_results[K]
        rounds = data['rounds']
        
        # Single seed case
        if data['n_seeds'] == 1:
            accuracies = data['acc_trajectory']['mean']
            
            filename = f'{save_dir}/test_accuracy_K{K}.txt'
            with open(filename, 'w') as f:
                f.write(f"# Test Accuracy for K={K} (1 seed)\n")
                f.write(f"# Format: round accuracy\n")
                f.write("round accuracy\n")
                
                for r, acc in zip(rounds, accuracies):
                    f.write(f"{r} {acc:.8f}\n")
            
            print(f"   ✅ Saved: {filename}")
        
        # Multi-seed case
        else:
            mean_accs = data['acc_trajectory']['mean']
            std_accs = data['acc_trajectory']['std']
            ci_accs = data['acc_trajectory']['ci']
            
            filename = f'{save_dir}/test_accuracy_K{K}.txt'
            with open(filename, 'w') as f:
                f.write(f"# Test Accuracy for K={K} ({data['n_seeds']} seeds)\n")
                f.write(f"# Format: round mean std ci_lower ci_upper\n")
                f.write("round mean std ci_lower ci_upper\n")
                
                for r, mean, std, ci in zip(rounds, mean_accs, std_accs, ci_accs):
                    ci_lower = mean - ci
                    ci_upper = mean + ci
                    f.write(f"{r} {mean:.8f} {std:.8f} {ci_lower:.8f} {ci_upper:.8f}\n")
            
            print(f"   ✅ Saved: {filename}")
    
    # Combined file (all K values)
    combined_accuracy_file = f'{save_dir}/test_accuracy_all_K.txt'
    with open(combined_accuracy_file, 'w') as f:
        f.write(f"# Test Accuracy for all K values\n")
        
        if all_results[list(all_results.keys())[0]]['n_seeds'] == 1:
            f.write(f"# Format: K round accuracy\n")
            f.write("K round accuracy\n")
            
            for K in sorted(all_results.keys()):
                data = all_results[K]
                rounds = data['rounds']
                accs = data['acc_trajectory']['mean']
                
                for r, acc in zip(rounds, accs):
                    f.write(f"{K} {r} {acc:.8f}\n")
        else:
            f.write(f"# Format: K round mean std ci_lower ci_upper\n")
            f.write("K round mean std ci_lower ci_upper\n")
            
            for K in sorted(all_results.keys()):
                data = all_results[K]
                rounds = data['rounds']
                means = data['acc_trajectory']['mean']
                stds = data['acc_trajectory']['std']
                cis = data['acc_trajectory']['ci']
                
                for r, mean, std, ci in zip(rounds, means, stds, cis):
                    ci_lower = mean - ci
                    ci_upper = mean + ci
                    f.write(f"{K} {r} {mean:.8f} {std:.8f} {ci_lower:.8f} {ci_upper:.8f}\n")
    
    print(f"   ✅ Saved: {combined_accuracy_file}")
    
    # FORMAT 3: CSV FILES
    print("\n3️⃣ Exporting CSV Files...")
    
    for K in sorted(all_results.keys()):
        data = all_results[K]
        rounds = data['rounds']
        
        if data['n_seeds'] == 1:
            df = pd.DataFrame({
                'round': rounds,
                'consensus_error': data['consensus_errors']['mean'],
                'test_accuracy': data['acc_trajectory']['mean']
            })
        else:
            df = pd.DataFrame({
                'round': rounds,
                'consensus_error_mean': data['consensus_errors']['mean'],
                'consensus_error_std': data['consensus_errors']['std'],
                'consensus_error_ci': data['consensus_errors']['ci'],
                'test_accuracy_mean': data['acc_trajectory']['mean'],
                'test_accuracy_std': data['acc_trajectory']['std'],
                'test_accuracy_ci': data['acc_trajectory']['ci']
            })
        
        csv_filename = f'{save_dir}/K{K}_complete_data.csv'
        df.to_csv(csv_filename, index=False)
        print(f"   ✅ Saved: {csv_filename}")
    
    print("\n" + "="*80)
    print("✅ ALL DATA EXPORTED FOR LATEX!")
    print("="*80)
    print(f"\n📁 Files saved to: {save_dir}/")
    print(f"   • consensus_error_K*.txt (individual K)")
    print(f"   • consensus_error_all_K.txt (combined)")
    print(f"   • test_accuracy_K*.txt (individual K)")
    print(f"   • test_accuracy_all_K.txt (combined)")
    print(f"   • K*_complete_data.csv (CSV format)")
    print("="*80)


def print_data_to_terminal(all_results: Dict, K_value: int = 5, max_rounds: int = 20):
    """
    Print data to terminal for quick viewing
    Useful for checking values before plotting
    """
    if K_value not in all_results:
        print(f"❌ K={K_value} not found in results!")
        return
    
    data = all_results[K_value]
    rounds = data['rounds'][:max_rounds]
    
    print("\n" + "="*100)
    print(f"📊 DATA FOR K={K_value} (First {max_rounds} rounds)")
    print("="*100)
    
    if data['n_seeds'] == 1:
        print(f"{'Round':<8} {'Consensus Error':<20} {'Test Accuracy (%)':<20}")
        print("-"*100)
        
        cons_errors = data['consensus_errors']['mean'][:max_rounds]
        test_accs = data['acc_trajectory']['mean'][:max_rounds]
        
        for r, err, acc in zip(rounds, cons_errors, test_accs):
            print(f"{r:<8} {err:<20.8f} {acc:<20.8f}")
    
    else:
        print(f"{'Round':<8} {'Cons Err Mean':<20} {'Cons Err CI':<20} {'Acc Mean (%)':<20} {'Acc CI':<20}")
        print("-"*100)
        
        cons_means = data['consensus_errors']['mean'][:max_rounds]
        cons_cis = data['consensus_errors']['ci'][:max_rounds]
        acc_means = data['acc_trajectory']['mean'][:max_rounds]
        acc_cis = data['acc_trajectory']['ci'][:max_rounds]
        
        for r, c_mean, c_ci, a_mean, a_ci in zip(rounds, cons_means, cons_cis, acc_means, acc_cis):
            print(f"{r:<8} {c_mean:<20.8f} ±{c_ci:<19.8f} {a_mean:<20.8f} ±{a_ci:<19.8f}")
    
    print("="*100)

# MAIN
if __name__ == "__main__":
    print(f"\n🔧 Device: {device}")
    if device == 'cuda':
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")
    
    experiment_start = time.time()
    
    # Run with resume capability
    all_results = run_multi_k_multi_seed_experiments(
        k_values=K_VALUES,
        seeds=SEEDS,
        verbose=True
    )
    
    total_time = time.time() - experiment_start
    
    # Plots
    print("\n" + "="*80)
    print("📊 GENERATING PLOTS...")
    print("="*80)
    
    plot_multi_k_convergence_with_ci(all_results)
    results_df = generate_results_table(all_results)
    
    # EXPORT DATA FOR LATEX 
    export_all_data_for_latex(all_results, save_dir='./latex_data')
    
    # Optional: Print sample data to terminal
    print_data_to_terminal(all_results, K_value=5, max_rounds=20)  # First 20 rounds of K=5
    
    print(f"\n⏱️  Total runtime: {total_time/3600:.2f} hours")
    print(f"\n✅ ALL COMPLETE!")
    print(f"\n📁 Results in ./plots/")
    print(f"   • theorem4_validation_multi_k.png")
    print(f"   • theorem4_validation_results.csv")
    print(f"\n📁 LaTeX data in ./latex_data/")
    print(f"   • consensus_error_K*.txt")
    print(f"   • test_accuracy_K*.txt")
    print(f"   • K*_complete_data.csv")
    print(f"\n💾 Checkpoints in {CHECKPOINT_DIR}/")
    print("="*80)
