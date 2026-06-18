# ===========================
# RANDOM-2-NEIGHBOR TOPOLOGY (EACH CLIENT PICKS 2 RANDOM NEIGHBORS PER ROUND)
# Runs: K = [3,5,7,9], seed= 43, 44, 45, 46
# Dataset: FashionMNIST (FMNIST)
# ===========================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
import time
from sklearn.metrics import f1_score
import pandas as pd

# SEED
def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CONFIG 
T = 100
eta_initial = 0.01
eta_min = 0.001
alpha_init = 0.5
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
warmup_rounds = 5

K_VALUES = [3, 5, 7, 9]
SEED = 43

# DATA (ROBIN-ROUND VERTICAL SPLIT) 
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
            img_pil = transforms.functional.affine(
                img_pil, angle=0, translate=[translate_x, translate_y], scale=1, shear=0
            )
        img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0)
        augmented_data.append(img_tensor)
    return torch.cat(augmented_data, dim=0)

def normalize_data(data, mean=0.2860, std=0.3530):
    return (data - mean) / std

def load_fmnist_vertically_partitioned_robin_round(num_clients=2, augment=True):
    base_transform = transforms.ToTensor()
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=base_transform)
    test_dataset  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=base_transform)

    train_data = train_dataset.data.unsqueeze(1).float() / 255.0
    test_data  = test_dataset.data.unsqueeze(1).float() / 255.0
    train_targets = train_dataset.targets
    test_targets  = test_dataset.targets

    def split_images_robin_round(data, num_clients):
      splits = []
      for client_id in range(num_clients):
        splits.append(data[:, :, :, client_id::num_clients]) 

      # ✅ IMPORTANT FIX: widths equal across clients
      min_w = min(s.shape[-1] for s in splits)
      splits = [s[:, :, :, :min_w] for s in splits]

      return splits


    train_splits = split_images_robin_round(train_data, num_clients)
    test_splits  = split_images_robin_round(test_data,  num_clients)


    if augment:
        train_splits = [apply_augmentation(split) for split in train_splits]

    train_splits = [normalize_data(split) for split in train_splits]
    test_splits  = [normalize_data(split) for split in test_splits]

    return train_splits, train_targets, test_splits, test_targets

class ClientDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def prepare_client_dataloaders(train_splits, train_targets, test_splits, test_targets, batch_size=64):
    client_train_loaders, client_test_loaders = [], []
    for i in range(len(train_splits)):
        train_ds = ClientDataset(train_splits[i], train_targets)
        test_ds  = ClientDataset(test_splits[i],  test_targets)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        client_train_loaders.append(train_loader)
        client_test_loaders.append(test_loader)
    return client_train_loaders, client_test_loaders

# MODEL
class AttentionGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super().__init__()
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
        B, N, _ = H.shape
        gnn_out = torch.relu(torch.matmul(torch.matmul(W, H), self.theta))

        Q = self.q_linear(H).view(B, N, self.num_heads, self.head_dim)
        K = self.k_linear(H).view(B, N, self.num_heads, self.head_dim)
        V = self.v_linear(H).view(B, N, self.num_heads, self.head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attn_out = torch.matmul(attn, V).view(B, N, self.output_dim)
        attn_out = self.out_linear(attn_out)

        mix = torch.sigmoid(self.alpha) * gnn_out + (1 - torch.sigmoid(self.alpha)) * attn_out
        if self.input_dim == self.output_dim:
            mix = mix + H
        return self.layer_norm(mix)

class MultiLayerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(AttentionGNNLayer(input_dim, hidden_dims[0], num_heads))
        for i in range(1, len(hidden_dims)):
            self.layers.append(AttentionGNNLayer(hidden_dims[i-1], hidden_dims[i], num_heads))

    def forward(self, H, W):
        for layer in self.layers:
            H = layer(H, W)
        return H

def create_spatial_prior_adjacency(height, width, sigma=1.5):
    num_pixels = height * width
    W = torch.zeros(num_pixels, num_pixels)
    for i in range(num_pixels):
        ri, ci = i // width, i % width
        for j in range(num_pixels):
            rj, cj = j // width, j % width
            dist = math.sqrt((ri-rj)**2 + (ci-cj)**2)
            W[i, j] = math.exp(-dist**2 / (2*sigma**2))
    return W

class EnhancedClientModel(nn.Module):
    def __init__(self, input_shape, gnn_hidden_dims=[64, 32], mlp_hidden=64, num_classes=10, sparsity_reg=1e-4):
        super().__init__()
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
            nn.Linear(mlp_hidden, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        H0 = x.view(B, self.flatten_dim, 1)
        W_norm = self.W / (self.W.norm(dim=1, keepdim=True) + 1e-6)
        H = self.gnn(H0, W_norm)
        return self.classifier(H.view(B, -1))

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
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

# CONSENSUS (As per theory: EACH CLIENT PICKS 2 RANDOM NEIGHBORS)
def adaptive_consensus_rate(round_idx, initial_alpha=0.5, min_alpha=0.1, decay_rate=0.02):
    return max(min_alpha, initial_alpha * math.exp(-decay_rate * round_idx))

def compute_consensus_error(W_list):
    K = len(W_list)
    W_avg = sum(W_list) / K
    err = 0.0
    for k in range(K):
        err += torch.norm(W_list[k] - W_avg, p='fro').item()**2
    return err / K

def random_two_neighbors_per_client(K, rng: random.Random):
    # directed: each client k chooses 2 distinct outgoing neighbors
    neigh = []
    for k in range(K):
        candidates = list(range(K))
        candidates.remove(k)
        # K>=3 always in unsere experiments, so sampling 2 is valid
        j1, j2 = rng.sample(candidates, 2)
        neigh.append((j1, j2))
    return neigh

def apply_random2_consensus(client_models, alpha_t, seed_for_round):
    K = len(client_models)
    rng = random.Random(seed_for_round)

    # snapshot 
    W_snap = [m.W.data.detach().clone() for m in client_models]

    neigh = random_two_neighbors_per_client(K, rng)

    for k in range(K):
        j1, j2 = neigh[k]
        client_models[k].W.data = (1 - alpha_t) * W_snap[k] + (alpha_t / 2.0) * (W_snap[j1] + W_snap[j2])

    return neigh

# EVAL
@torch.no_grad()
def evaluate_ensemble_with_f1(client_models, client_test_loaders):
    K = len(client_models)
    all_probs = []
    all_labels = None
    for k in range(K):
        model = client_models[k]
        model.eval()
        probs_k = []
        labels_k = []
        for x, y in client_test_loaders[k]:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            probs_k.append(probs.cpu())
            labels_k.append(y.cpu())
        all_probs.append(torch.cat(probs_k, dim=0))
        if all_labels is None:
            all_labels = torch.cat(labels_k, dim=0)

    ens_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    ens_pred = ens_probs.argmax(dim=1)
    acc = (ens_pred == all_labels).float().mean().item() * 100.0
    f1  = f1_score(all_labels.numpy(), ens_pred.numpy(), average='macro') * 100.0
    return acc, f1

# TRAINING
def train_one_round_random2(client_models, client_opts, client_scheds, train_loaders, round_idx, base_seed):
    loss_fn = nn.CrossEntropyLoss()

    # Local training
    for k, model in enumerate(client_models):
        model.train()
        opt = client_opts[k]
        sched = client_scheds[k]
        loader = train_loaders[k]

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y) + model.get_sparsity_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        sched.step(round_idx)

    # Consensus error PRE
    W_pre = [m.W.detach().clone().cpu() for m in client_models]
    eps_pre = compute_consensus_error(W_pre)

    # Random-2-neighbor consensus (after warmup)
    if round_idx > warmup_rounds:
        alpha_t = adaptive_consensus_rate(round_idx - warmup_rounds, alpha_init, 0.1)
        neigh = apply_random2_consensus(client_models, alpha_t, seed_for_round=base_seed + 1000*round_idx)
    else:
        alpha_t = 0.0
        neigh = None

    # Consensus error POST
    W_post = [m.W.detach().clone().cpu() for m in client_models]
    eps_post = compute_consensus_error(W_post)

    return eps_pre, eps_post, alpha_t, neigh

def run_single_seed_random2(K, seed=43, verbose=True):
    set_seed(seed)

    train_splits, train_targets, test_splits, test_targets = load_fmnist_vertically_partitioned_robin_round(
        num_clients=K, augment=True
    )
    train_loaders, test_loaders = prepare_client_dataloaders(
        train_splits, train_targets, test_splits, test_targets, batch_size
    )

    input_shape = train_splits[0][0].shape  # (1, 28, width_k)
    client_models = [EnhancedClientModel(input_shape).to(device) for _ in range(K)]

    client_opts = [torch.optim.AdamW(m.parameters(), lr=eta_initial, weight_decay=1e-4) for m in client_models]
    client_scheds = [CosineAnnealingWarmup(opt, warmup_epochs=warmup_rounds, max_epochs=T, eta_min=eta_min)
                     for opt in client_opts]

    eps_post_hist = []

    start = time.time()
    for t in range(1, T+1):
        eps_pre, eps_post, alpha_t, neigh = train_one_round_random2(
            client_models, client_opts, client_scheds, train_loaders, t, base_seed=seed
        )
        eps_post_hist.append(eps_post)

        if verbose and (t % 20 == 0 or t == 1):
            if neigh is None:
                print(f"[K={K} | t={t:03d}] eps_post={eps_post:.3f} | alpha={alpha_t:.3f} | warmup")
            else:
                print(f"[K={K} | t={t:03d}] eps_post={eps_post:.3f} | alpha={alpha_t:.3f} | neigh(client0)={neigh[0]}")

    train_time = time.time() - start
    ens_acc, ens_f1 = evaluate_ensemble_with_f1(client_models, test_loaders)

    eps0 = eps_post_hist[0]
    epsT = eps_post_hist[-1]
    red = (1 - epsT/eps0) * 100 if eps0 > 0 else float('nan')

    return {
        "K": K,
        "seed": seed,
        "ensemble_acc": ens_acc,
        "ensemble_f1": ens_f1,
        "train_time": train_time,
        "eps0": eps0,
        "epsT": epsT,
        "eps_reduction_pct": red,
    }

# -------------------- MAIN --------------------
if __name__ == "__main__":
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    results = []
    for K in K_VALUES:
        print("\n" + "="*80)
        print(f"RUNNING RANDOM-2-NEIGHBORS | K={K} | seed={SEED}")
        print("="*80)
        out = run_single_seed_random2(K=K, seed=SEED, verbose=True)
        results.append(out)

        print(f"✓ K={K} done: Acc={out['ensemble_acc']:.2f} | F1={out['ensemble_f1']:.2f} | "
              f"eps: {out['eps0']:.2f}->{out['epsT']:.2f} ({out['eps_reduction_pct']:.1f}%) | "
              f"time={out['train_time']:.1f}s")

    df = pd.DataFrame(results)
    print("\n" + "="*100)
    print("FINAL SUMMARY (Random-2-Neighbors, one seed)")
    print("="*100)
    print(df.to_string(index=False))
