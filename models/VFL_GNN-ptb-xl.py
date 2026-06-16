"""
PTB-XL VFL-GNN 
Uses full 21,799 sample dataset without memory issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import wfdb
import random
import math
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import os
from scipy import stats

# SEEDING 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CONFIG 
T = 100  # Training rounds
η_initial = 0.001
η_min = 0.0001
α = 0.5  # Consensus rate
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
warmup_rounds = 5
num_clients = 3  # We perform PTB-XL analysis with 3 clients
SEEDS = [42, 123, 456, 789, 2024]  # 5 seeds for averaging
#SEEDS = [42]

# PTB-XL Config
DATA_PATH = './ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
TARGET_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLIENT_LEADS = {
    'client_1': [0, 1, 2],        # I, II, III
    'client_2': [3, 4, 5],        # aVR, aVL, aVF
    'client_3': [6, 7, 8, 9, 10, 11]  # V1-V6
}

print(f"🔧 Device: {device}")
if device == 'cuda':
    print(f"🔧 GPU: {torch.cuda.get_device_name()}")

# DATA LOADING
class PTBXLMetadataLoader:
  
    def __init__(self, data_path, sampling_rate=500, max_samples=None):
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.target_classes = TARGET_CLASSES
        self.max_samples = max_samples
        
    def load_metadata(self):
        print("[1/3] Loading PTB-XL metadata...")
        
        db_path = os.path.join(self.data_path, 'ptbxl_database.csv')
        self.Y = pd.read_csv(db_path, index_col='ecg_id')
        
        scp_path = os.path.join(self.data_path, 'scp_statements.csv')
        self.scp_statements = pd.read_csv(scp_path, index_col=0)
        
        self.Y.scp_codes = self.Y.scp_codes.apply(lambda x: eval(x))
        
        if self.max_samples:
            self.Y = self.Y.head(self.max_samples)
        
        print(f"   ✓ Loaded {len(self.Y)} records (metadata)")
        return self
    
    def create_labels(self):
        """Create 5-class multilabel labels"""
        print("[2/3] Creating labels...")
        
        def aggregate_diagnostic(y_dict, agg_df):
            tmp = []
            for key in y_dict.keys():
                if key in agg_df.index:
                    cls = agg_df.loc[key].diagnostic_class
                    if cls in self.target_classes:
                        tmp.append(cls)
            return list(set(tmp))
        
        self.Y['diagnostic_superclass'] = self.Y.scp_codes.apply(
            lambda x: aggregate_diagnostic(x, self.scp_statements)
        )
        
        # Filename
        if self.sampling_rate == 100:
            self.Y['filename'] = self.Y['filename_lr']
        else:
            self.Y['filename'] = self.Y['filename_hr']
        
        # Label matrix and Valid filepaths
        labels = []
        filepaths = []
        valid_indices = []
        
        for idx, row in self.Y.iterrows():
            classes = row['diagnostic_superclass']
            if len(classes) > 0:  
                label = [1 if cls in classes else 0 for cls in self.target_classes]
                labels.append(label)
                filepaths.append(row['filename'])
                valid_indices.append(idx)
        
        self.labels = np.array(labels)
        self.filepaths = filepaths
        self.valid_indices = valid_indices
        
        print(f"   ✓ Created {len(labels)} labels (5 classes)")
        print(f"   ✓ Class distribution: {self.labels.sum(axis=0)}")
        return self
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        print("[3/3] Creating train/test split...")
        
        n_samples = len(self.labels)
        indices = np.arange(n_samples)
        
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state,
            stratify=self.labels[:, 0]
        )
        
        train_data = {
            'filepaths': [self.filepaths[i] for i in train_idx],
            'labels': self.labels[train_idx]
        }
        
        test_data = {
            'filepaths': [self.filepaths[i] for i in test_idx],
            'labels': self.labels[test_idx]
        }
        
        print(f"   ✓ Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"\n✅ Metadata loaded! ECG signals will stream during training (memory-efficient)")
        
        return train_data, test_data

def load_ptbxl_metadata(max_samples=None):
    """Load only metadata - ECG signals loaded on-demand during training"""
    loader = PTBXLMetadataLoader(DATA_PATH, sampling_rate=500, max_samples=max_samples)
    loader.load_metadata().create_labels()
    return loader.get_train_test_split()

# ==================== STREAMING DATASET ====================
class PTBXLStreamingDataset(Dataset):
    """Loads ECG signals on-demand"""
    
    def __init__(self, filepaths, labels, lead_indices, data_path):
        self.filepaths = filepaths
        self.labels = torch.FloatTensor(labels)
        self.lead_indices = lead_indices
        self.data_path = data_path
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load ECG signal on-demand  
        filepath = os.path.join(self.data_path, self.filepaths[idx])
        try:
            record = wfdb.rdsamp(filepath)
            signal = record[0][:, self.lead_indices]  # (time_steps, num_leads)
            
            # Normalizing per-sample
            signal = (signal - signal.mean(axis=0, keepdims=True)) / (signal.std(axis=0, keepdims=True) + 1e-8)
        except:
            signal = np.zeros((5000, len(self.lead_indices)))
        
        return torch.FloatTensor(signal), self.labels[idx]

def prepare_dataloaders(train_data, test_data, batch_size=32):
    """For 3 clients"""
    train_loaders = []
    test_loaders = []
    
    for i in range(1, 4):
        client_key = f'client_{i}'
        lead_indices = CLIENT_LEADS[client_key]
        
        train_ds = PTBXLStreamingDataset(
            train_data['filepaths'], 
            train_data['labels'], 
            lead_indices,
            DATA_PATH
        )
        test_ds = PTBXLStreamingDataset(
            test_data['filepaths'],
            test_data['labels'],
            lead_indices,
            DATA_PATH
        )
        
        # num_workers=2 enables parallel data loading
        train_loaders.append(DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        ))
        test_loaders.append(DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        ))
    
    return train_loaders, test_loaders

# GNN MODEL (Same as before)
class AttentionGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super(AttentionGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
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
        
        # GNN path
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

class PTBXLClientModel(nn.Module):
    """Client model for PTB-XL ECG data"""
    def __init__(self, time_steps, num_leads, gnn_hidden_dims=[64, 32], mlp_hidden=64, num_classes=5):
        super(PTBXLClientModel, self).__init__()
        self.time_steps = time_steps
        self.num_leads = num_leads
        
        # Adjacency matrix (learnable)
        self.W = nn.Parameter(torch.randn(time_steps, time_steps) * 0.01)
        
        # GNN (same architecture as F-MNIST)
        self.gnn = MultiLayerGNN(input_dim=num_leads, hidden_dims=gnn_hidden_dims, num_heads=4)
        
        # Classifier
        final_gnn_dim = gnn_hidden_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(time_steps * final_gnn_dim, mlp_hidden * 2),
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
        B = x.size(0)  # x: (batch, time_steps, num_leads)
        H0 = x  # Shape: (batch, time_steps, num_leads)
        
        W_norm = self.W / (self.W.norm(dim=1, keepdim=True) + 1e-6)
        H_out = self.gnn(H0, W_norm)
        H_flat = H_out.view(B, -1)
        
        return self.classifier(H_flat)

# TRAINING (Same as before)
def get_ring_neighbors(k, K):
    return [(k - 1) % K, (k + 1) % K]

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

def calculate_pos_weight(labels):
    """Positive class weights for imbalanced data"""
    pos_count = labels.sum(axis=0)
    neg_count = len(labels) - pos_count
    pos_weight = neg_count / (pos_count + 1e-8)
    return torch.FloatTensor(pos_weight).to(device)

def train_one_round(models, optimizers, schedulers, loaders, round_idx, pos_weight=None):
    K = len(models)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    
    for k in range(K):
        model = models[k]
        model.train()
        optimizer = optimizers[k]
        scheduler = schedulers[k]
        
        for batch_x, batch_y in loaders[k]:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step(round_idx)
    
    # Consensus
    if round_idx > warmup_rounds:
        W_updates = [m.W.clone().detach() for m in models]
        for k in range(K):
            neighbors = get_ring_neighbors(k, K)
            consensus_W = (1 - α) * W_updates[k]
            for n in neighbors:
                consensus_W += (α / len(neighbors)) * W_updates[n]
            models[k].W.data = consensus_W

# EVALUATION
@torch.no_grad()
def evaluate_clients(models, loaders):
    """Evaluate individual clients"""
    K = len(models)
    all_preds = [[] for _ in range(K)]
    all_labels = [[] for _ in range(K)]
    all_probs = [[] for _ in range(K)]
    
    for k in range(K):
        models[k].eval()
        for batch_x, batch_y in loaders[k]:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = models[k](batch_x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds[k].append(preds.cpu())
            all_labels[k].append(batch_y.cpu())
            all_probs[k].append(probs.cpu())
    
    # Concatenate
    all_preds = [torch.cat(p, dim=0).numpy() for p in all_preds]
    all_labels = [torch.cat(l, dim=0).numpy() for l in all_labels]
    all_probs = [torch.cat(p, dim=0).numpy() for p in all_probs]
    
    # Calculate metrics
    accuracies = []
    f1_scores = []
    auc_scores = []
    
    for k in range(K):
        acc = ((all_preds[k] == all_labels[k]).sum() / all_labels[k].size) * 100
        f1 = f1_score(all_labels[k], all_preds[k], average='macro') * 100
        try:
            auc = roc_auc_score(all_labels[k], all_probs[k], average='macro') * 100
        except:
            auc = 0.0
        
        accuracies.append(acc)
        f1_scores.append(f1)
        auc_scores.append(auc)
    
    return accuracies, f1_scores, auc_scores, all_preds, all_labels, all_probs

@torch.no_grad()
def evaluate_ensemble(models, loaders):
    """Ensemble evaluation"""
    K = len(models)
    all_probs_list = []
    
    for k in range(K):
        models[k].eval()
        client_probs = []
        for batch_x, batch_y in loaders[k]:
            batch_x = batch_x.to(device)
            logits = models[k](batch_x)
            probs = torch.sigmoid(logits)
            client_probs.append(probs.cpu())
        all_probs_list.append(torch.cat(client_probs, dim=0))
    
    # Labels (same for all clients)
    all_labels = []
    for batch_x, batch_y in loaders[0]:
        all_labels.append(batch_y)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Ensemble (average)
    ensemble_probs = torch.stack(all_probs_list, dim=0).mean(dim=0).numpy()
    ensemble_preds = (ensemble_probs > 0.5).astype(float)
    
    acc = ((ensemble_preds == all_labels).sum() / all_labels.size) * 100
    f1 = f1_score(all_labels, ensemble_preds, average='macro') * 100
    try:
        auc = roc_auc_score(all_labels, ensemble_probs, average='macro') * 100
    except:
        auc = 0.0
    
    return acc, f1, auc

# SINGLE SEED EXPERIMENT
def run_single_seed(seed, train_data, test_data, verbose=False):
    """Run experiment for one seed"""
    set_seed(seed)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RUNNING SEED {seed}")
        print(f"{'='*60}")
    
    # Dataloaders
    train_loaders, test_loaders = prepare_dataloaders(train_data, test_data, batch_size)
    
    # Class weights for imbalanced data
    pos_weight = calculate_pos_weight(train_data['labels'])
    if verbose:
        print(f"Class weights: {pos_weight.cpu().numpy()}")
    
    # Models for each client
    models = []
    for i in range(1, 4):
        num_leads = len(CLIENT_LEADS[f'client_{i}'])
        time_steps = 5000  
        model = PTBXLClientModel(time_steps, num_leads, gnn_hidden_dims=[64, 32], mlp_hidden=64, num_classes=5)
        models.append(model.to(device))
    
    # Optimizers & schedulers
    optimizers = [torch.optim.AdamW(m.parameters(), lr=η_initial, weight_decay=1e-4) for m in models]
    schedulers = [CosineAnnealingWarmup(opt, warmup_rounds, T, η_min) for opt in optimizers]
    
    # Training
    if verbose:
        print(f"\nTraining for {T} rounds (streaming data from disk)...")
    for t in range(1, T + 1):
        train_one_round(models, optimizers, schedulers, train_loaders, t, pos_weight=pos_weight)
        if verbose and t % 20 == 0:
            print(f"  Round {t}/{T}")
    
    # Evaluation
    ind_accs, ind_f1s, ind_aucs, _, _, _ = evaluate_clients(models, test_loaders)
    ens_acc, ens_f1, ens_auc = evaluate_ensemble(models, test_loaders)
    
    results = {
        'seed': seed,
        'individual_accuracies': ind_accs,
        'individual_f1s': ind_f1s,
        'individual_aucs': ind_aucs,
        'avg_individual_accuracy': np.mean(ind_accs),
        'avg_individual_f1': np.mean(ind_f1s),
        'avg_individual_auc': np.mean(ind_aucs),
        'ensemble_accuracy': ens_acc,
        'ensemble_f1': ens_f1,
        'ensemble_auc': ens_auc
    }
    
    if verbose:
        print(f"\nResults (Seed {seed}):")
        print(f"  Individual Avg: {results['avg_individual_accuracy']:.2f}% (Acc), {results['avg_individual_f1']:.2f}% (F1)")
        print(f"  Ensemble: {ens_acc:.2f}% (Acc), {ens_f1:.2f}% (F1)")
    
    return results

#  MULTI-SEED WITH CI
def calculate_confidence_interval(values, confidence=0.95):
    """Calculate 95% confidence interval"""
    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)
    ci = std_err * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, ci

def run_multi_seed_experiment(seeds=SEEDS, max_samples=None):
    """Run experiment for multiple seeds and compute statistics"""
    print(f"\n{'='*70}")
    print(f"VFL-GNN ON PTB-XL - {len(seeds)} SEEDS EVALUATION (STREAMING MODE)")
    print(f"{'='*70}")
    
    # Loading metadata (lightweight - only filenames and labels)
    train_data, test_data = load_ptbxl_metadata(max_samples=max_samples)
    
    print(f"\n🚀 Starting {len(seeds)} seed evaluations...")
    print(f"💡 ECG signals streaming from disk (memory-efficient)")
    
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Running seed {seed}...")
        result = run_single_seed(seed, train_data, test_data, verbose=True)
        all_results.append(result)
    
    # Aggregate results
    metrics = {
        'avg_individual_accuracy': [],
        'avg_individual_f1': [],
        'avg_individual_auc': [],
        'ensemble_accuracy': [],
        'ensemble_f1': [],
        'ensemble_auc': []
    }
    
    for result in all_results:
        for key in metrics.keys():
            metrics[key].append(result[key])
    
    # Mean ± CI
    final_results = {}
    for key, values in metrics.items():
        mean, ci = calculate_confidence_interval(values)
        final_results[key] = {'mean': mean, 'ci': ci, 'values': values}
    
    # Final results
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS ({len(seeds)} seeds, mean ± 95% CI)")
    print(f"{'='*70}")
    
    print(f"\n📊 Individual Client Performance:")
    print(f"   Accuracy: {final_results['avg_individual_accuracy']['mean']:.2f}% ± {final_results['avg_individual_accuracy']['ci']:.2f}%")
    print(f"   F1 Score: {final_results['avg_individual_f1']['mean']:.2f}% ± {final_results['avg_individual_f1']['ci']:.2f}%")
    print(f"   AUC:      {final_results['avg_individual_auc']['mean']:.2f}% ± {final_results['avg_individual_auc']['ci']:.2f}%")
    
    print(f"\n📊 Ensemble Performance:")
    print(f"   Accuracy: {final_results['ensemble_accuracy']['mean']:.2f}% ± {final_results['ensemble_accuracy']['ci']:.2f}%")
    print(f"   F1 Score: {final_results['ensemble_f1']['mean']:.2f}% ± {final_results['ensemble_f1']['ci']:.2f}%")
    print(f"   AUC:      {final_results['ensemble_auc']['mean']:.2f}% ± {final_results['ensemble_auc']['ci']:.2f}%")
    
    # Results table
    results_table = pd.DataFrame({
        'Metric': ['Avg_Individual_Acc', 'Avg_Individual_F1', 'Avg_Individual_AUC', 
                   'Ensemble_Acc', 'Ensemble_F1', 'Ensemble_AUC'],
        'Mean': [f"{final_results[k]['mean']:.2f}%" for k in metrics.keys()],
        'CI': [f"±{final_results[k]['ci']:.2f}%" for k in metrics.keys()],
        'Std': [f"{np.std(final_results[k]['values']):.2f}%" for k in metrics.keys()]
    })
    
    print(f"\n{results_table.to_string(index=False)}")
    
    return final_results, results_table, all_results

# ==================== MAIN ====================
if __name__ == "__main__":
    print("🚀 PTB-XL VFL-GNN Experiment Starting (STREAMING MODE)...")
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Training rounds: {T}")
    print(f"Batch size: {batch_size}")
    
    
    final_results, results_table, all_results = run_multi_seed_experiment(
        seeds=SEEDS,
        max_samples=None  # FULL DATASET with streaming
    )
    
    # Results
    results_table.to_csv('ptbxl_vfl_gnn_results.csv', index=False)
    print(f"\n💾 Results saved to 'ptbxl_vfl_gnn_results.csv'")
    
    print(f"\n✅ Experiment Complete!")