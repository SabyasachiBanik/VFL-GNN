import os, re, gc, time, math, random, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, hamming_loss
from datasets import load_dataset

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

# SEEDING
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CONFIG
T = 80
LR_HEAD = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32

# Consensus
WARMUP_ROUNDS = 3
ALPHA_INIT = 0.5
ALPHA_MIN = 0.1
ALPHA_DECAY = 0.05

USE_W_EMA = True
W_EMA_BETA = 0.7
COMM_DTYPE = torch.float16

NUM_LABELS = 23
MAX_TEXT_LEN = 256

# Graph size
USE_1024_GRAPH = False
IMG_PROJ = 384 if not USE_1024_GRAPH else 512
TXT_PROJ = 384 if not USE_1024_GRAPH else 512
D_TOTAL = IMG_PROJ + TXT_PROJ

GNN_HIDDEN = [64, 32]
MLP_HIDDEN = 512
SPARSITY_REG = 1e-6

CACHE_DIR = "./mmimdb_vflgnn_cache_fusion_asl"
os.makedirs(CACHE_DIR, exist_ok=True)

GENRES = ['drama', 'comedy', 'romance', 'thriller', 'crime', 'action', 'adventure',
          'horror', 'documentary', 'mystery', 'sci-fi', 'fantasy', 'family',
          'biography', 'war', 'history', 'music', 'animation', 'musical',
          'western', 'sport', 'short', 'film-noir']

# UTILS
def split_indices(n, train_ratio=0.7, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

@torch.no_grad()
def find_optimal_thresholds(probs, labels):
    thresholds = torch.zeros(labels.size(1))
    for i in range(labels.size(1)):
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            pred = (probs[:, i] > t).float()
            f1 = f1_score(labels[:, i].numpy(), pred.numpy(), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds

# DATA PARSING
def parse_mmimdb(ds_split):
    plots, labels = [], []
    for ex in tqdm(ds_split, desc="Parsing MM-IMDB"):
        try:
            content = ex["messages"][0]["content"]
            match = re.search(r"Plot:\s*(.+?)\nNote", content, re.DOTALL)
            plot = match.group(1).strip() if match else ""
        except:
            plot = ""
        plots.append(plot)

        mh = torch.zeros(NUM_LABELS, dtype=torch.float32)
        try:
            genre_text = ex["messages"][1]["content"]
            genre_labels = [g.strip().lower() for g in genre_text.split(",")]
            for g in genre_labels:
                if g in GENRES:
                    mh[GENRES.index(g)] = 1.0
        except:
            pass
        labels.append(mh)

    return plots, torch.stack(labels, dim=0)

# PRETRAINED EMBEDDINGS
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoTokenizer, AutoModel

IMG_SIZE = 224
img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def safe_image_tensor(pil_image):
    try:
        if pil_image is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return img_transform(pil_image)
    except:
        return torch.zeros(3, IMG_SIZE, IMG_SIZE)

class FrozenResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.backbone(x).flatten(1)  # (B,2048)

class FrozenDistilBert(nn.Module):
    def __init__(self, name="distilbert-base-uncased"):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # CLS (B,768)

@torch.no_grad()
def extract_or_load_embeddings(ds_split, plots, labels, force=False):
    tag = f"D{D_TOTAL}"
    img_path = os.path.join(CACHE_DIR, f"img_resnet50_{tag}.pt")
    txt_path = os.path.join(CACHE_DIR, f"txt_distilbert_{tag}.pt")
    lbl_path = os.path.join(CACHE_DIR, f"labels_{tag}.pt")

    if (not force) and os.path.exists(img_path) and os.path.exists(txt_path) and os.path.exists(lbl_path):
        print("📦 Loading cached embeddings...")
        return (torch.load(img_path, map_location="cpu"),
                torch.load(txt_path, map_location="cpu"),
                torch.load(lbl_path, map_location="cpu"))

    print("🔧 Extracting embeddings (pretrained, frozen)...")
    img_enc = FrozenResNet50().to(DEVICE).eval()
    txt_enc = FrozenDistilBert().to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    n = len(labels)
    img_feats, txt_feats = [], []

    print("🖼️ Image embeddings...")
    for i in tqdm(range(0, n, BATCH_SIZE), desc="Images"):
        batch_imgs = []
        for j in range(i, min(i + BATCH_SIZE, n)):
            try:
                pil = ds_split[j]["images"][0]
            except:
                pil = None
            batch_imgs.append(safe_image_tensor(pil))
        x = torch.stack(batch_imgs).to(DEVICE)
        feat = img_enc(x).cpu()
        img_feats.append(feat)
        del x, feat
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("📝 Text embeddings...")
    for i in tqdm(range(0, n, BATCH_SIZE), desc="Text"):
        batch_text = plots[i:min(i + BATCH_SIZE, n)]
        batch_tok = tok(batch_text, padding=True, truncation=True,
                        max_length=MAX_TEXT_LEN, return_tensors="pt")
        input_ids = batch_tok["input_ids"].to(DEVICE)
        attn = batch_tok["attention_mask"].to(DEVICE)
        feat = txt_enc(input_ids, attn).cpu()
        txt_feats.append(feat)
        del input_ids, attn, batch_tok, feat
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    img_emb = torch.cat(img_feats, dim=0)
    txt_emb = torch.cat(txt_feats, dim=0)

    torch.save(img_emb, img_path)
    torch.save(txt_emb, txt_path)
    torch.save(labels, lbl_path)

    del img_enc, txt_enc
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"✓ img_emb: {tuple(img_emb.shape)} | txt_emb: {tuple(txt_emb.shape)} | labels: {tuple(labels.shape)}")
    return img_emb, txt_emb, labels

class PartyEmbDataset(Dataset):
    def __init__(self, party_emb, labels, indices):
        self.x = party_emb[indices]
        self.y = labels[indices]
    def __len__(self): return self.y.size(0)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ASYMMETRIC FOCAL LOSS (ASL)
class AsymmetricLoss(nn.Module):
    """
    Strong multi-label loss used in practice.
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip is not None and self.clip > 0:
            xs_neg = torch.clamp(xs_neg + self.clip, max=1.0)

        loss_pos = targets * torch.log(torch.clamp(xs_pos, min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(torch.clamp(xs_neg, min=self.eps))

        # focal modulation
        pt_pos = xs_pos * targets + (1 - targets)
        pt_neg = xs_neg * (1 - targets) + targets

        w_pos = torch.pow(1 - pt_pos, self.gamma_pos)
        w_neg = torch.pow(1 - pt_neg, self.gamma_neg)

        loss = - (w_pos * loss_pos + w_neg * loss_neg)
        return loss.mean()

# VFL-GNN
class AttentionGNNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super().__init__()
        assert output_dim % num_heads == 0
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.q = nn.Linear(input_dim, output_dim)
        self.k = nn.Linear(input_dim, output_dim)
        self.v = nn.Linear(input_dim, output_dim)
        self.o = nn.Linear(output_dim, output_dim)

        self.theta = nn.Parameter(torch.randn(input_dim, output_dim) * 0.02)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(0.1)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, H, W):
        gnn_out = torch.relu(torch.matmul(torch.matmul(W, H), self.theta))

        Q = self.q(H).view(H.size(0), H.size(1), self.num_heads, self.head_dim)
        K = self.k(H).view(H.size(0), H.size(1), self.num_heads, self.head_dim)
        V = self.v(H).view(H.size(0), H.size(1), self.num_heads, self.head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        A = torch.softmax(scores, dim=-1)
        A = self.dropout(A)

        attn_out = torch.matmul(A, V).reshape(H.size(0), H.size(1), self.output_dim)
        attn_out = self.o(attn_out)

        mix = torch.sigmoid(self.alpha) * gnn_out + (1 - torch.sigmoid(self.alpha)) * attn_out
        if H.size(-1) == self.output_dim:
            mix = mix + H
        return self.ln(mix)

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

def init_smooth_prior(D, sigma=12.0):
    idx = torch.arange(D).float()
    dist = idx.unsqueeze(1) - idx.unsqueeze(0)
    return torch.exp(-(dist**2) / (2 * sigma**2))

class VFLGNNPartyModel(nn.Module):
    """
    Party model outputs local logits (23) from its party embedding,
    still uses the shared graph W + GNN to remain your VFL-GNN concept.
    """
    def __init__(self, input_dim, party_start, party_dim, D_total=D_TOTAL):
        super().__init__()
        self.party_start = party_start
        self.party_dim = party_dim
        self.D_total = D_total

        self.proj = nn.Sequential(
            nn.Linear(input_dim, party_dim),
            nn.LayerNorm(party_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        prior = init_smooth_prior(D_total, sigma=12.0)
        self.W = nn.Parameter(prior * 0.05 + 0.01 * torch.randn(D_total, D_total))

        self.gnn = MultiLayerGNN(input_dim=1, hidden_dims=GNN_HIDDEN, num_heads=4)

        final_dim = GNN_HIDDEN[-1]
        self.head = nn.Sequential(
            nn.Linear(D_total * final_dim, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(MLP_HIDDEN, NUM_LABELS)
        )

    def get_sparsity_loss(self):
        return SPARSITY_REG * torch.norm(self.W, p=1)

    def forward_local_logits(self, x_party):
        z = self.proj(x_party)
        B = z.size(0)
        H0 = torch.zeros(B, self.D_total, 1, device=z.device)
        H0[:, self.party_start:self.party_start + self.party_dim, 0] = z

        Wn = self.W / (self.W.norm(dim=1, keepdim=True) + 1e-6)
        H = self.gnn(H0, Wn)
        return self.head(H.reshape(B, -1))

class FusionHead(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_LABELS * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, NUM_LABELS)
        )

    def forward(self, li, lt):
        x = torch.cat([li, lt], dim=1)
        return self.net(x)

# CONSENSUS (RING + EMA)
def get_ring_neighbors(k, K):
    return [(k - 1) % K, (k + 1) % K]

def adaptive_alpha(round_idx):
    return max(ALPHA_MIN, ALPHA_INIT * math.exp(-ALPHA_DECAY * round_idx))

@torch.no_grad()
def ring_consensus_W(models, round_idx):
    K = len(models)
    a = adaptive_alpha(round_idx)

    W_snap = []
    for m in models:
        W_comm = m.W.detach().to(COMM_DTYPE)
        W_back = W_comm.to(torch.float32)
        W_snap.append(W_back)

    for k in range(K):
        neigh = get_ring_neighbors(k, K)
        W_cons = (1 - a) * W_snap[k]
        for n in neigh:
            W_cons += (a / len(neigh)) * W_snap[n]

        if USE_W_EMA:
            mW = models[k].W.data
            models[k].W.data.copy_((1 - W_EMA_BETA) * mW + W_EMA_BETA * W_cons)
        else:
            models[k].W.data.copy_(W_cons)

# TRAIN / EVAL
def train_one_round(models, fusion, opts, fusion_opt, loaders, loss_fn, round_idx):
    # loaders = [img_loader, txt_loader] with same indices order? Not guaranteed.
    # To keep it correct, we train per-party independently + fusion on aligned batches by using SAME sampler order.
    # Simplest: use shuffle=False for train loaders and create identical permutation indices each epoch in dataset.
    # We do a synchronized loader by iterating zip(img_loader, txt_loader).
    models[0].train(); models[1].train(); fusion.train()
    total_loss, nb = 0.0, 0

    for (xi, y1), (xt, y2) in zip(loaders[0], loaders[1]):
        assert torch.equal(y1, y2), "Labels misaligned between loaders; set shuffle=False for train loaders."
        y = y1.to(DEVICE)
        xi = xi.to(DEVICE)
        xt = xt.to(DEVICE)

        opts[0].zero_grad()
        opts[1].zero_grad()
        fusion_opt.zero_grad()

        li = models[0].forward_local_logits(xi)
        lt = models[1].forward_local_logits(xt)
        lf = fusion(li, lt)

        loss = loss_fn(lf, y) + models[0].get_sparsity_loss() + models[1].get_sparsity_loss()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(models[0].parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(models[1].parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)

        opts[0].step()
        opts[1].step()
        fusion_opt.step()

        total_loss += loss.item()
        nb += 1

    if round_idx > WARMUP_ROUNDS:
        ring_consensus_W(models, round_idx - WARMUP_ROUNDS)

    return total_loss / max(1, nb)

@torch.no_grad()
def evaluate(models, fusion, loaders, thresholds=None):
    models[0].eval(); models[1].eval(); fusion.eval()
    probs_all, labels_all = [], []

    for (xi, y1), (xt, y2) in zip(loaders[0], loaders[1]):
        assert torch.equal(y1, y2), "Labels misaligned between loaders; keep shuffle=False in eval loaders."
        y = y1
        xi = xi.to(DEVICE)
        xt = xt.to(DEVICE)

        li = models[0].forward_local_logits(xi)
        lt = models[1].forward_local_logits(xt)
        lf = fusion(li, lt)

        probs = torch.sigmoid(lf).cpu()
        probs_all.append(probs)
        labels_all.append(y)

    probs_all = torch.cat(probs_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)

    if thresholds is None:
        preds = (probs_all > 0.5).float()
    else:
        preds = (probs_all > thresholds.unsqueeze(0)).float()

    f1_macro = f1_score(labels_all.numpy(), preds.numpy(), average="macro", zero_division=0) * 100
    f1_micro = f1_score(labels_all.numpy(), preds.numpy(), average="micro", zero_division=0) * 100
    hamming_acc = (1 - hamming_loss(labels_all.numpy(), preds.numpy())) * 100
    return {"f1_macro": f1_macro, "f1_micro": f1_micro, "hamming_acc": hamming_acc,
            "probs": probs_all, "labels": labels_all}

# MAIN
def run():
    print(f"🔧 Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")

    ds = load_dataset("sxj1215/mmimdb")
    split = ds["train"]

    plots, labels = parse_mmimdb(split)
    n = len(labels)

    train_idx, val_idx, test_idx = split_indices(n, 0.7, 0.1, seed=42)
    print(f"✓ Samples: {n} | Avg labels/sample: {labels.sum(1).mean().item():.2f}")
    print(f"✓ Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    img_emb, txt_emb, labels = extract_or_load_embeddings(split, plots, labels, force=False)

    train_loaders = [
        DataLoader(PartyEmbDataset(img_emb, labels, train_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(PartyEmbDataset(txt_emb, labels, train_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True),
    ]
    val_loaders = [
        DataLoader(PartyEmbDataset(img_emb, labels, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(PartyEmbDataset(txt_emb, labels, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True),
    ]
    test_loaders = [
        DataLoader(PartyEmbDataset(img_emb, labels, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(PartyEmbDataset(txt_emb, labels, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True),
    ]

    models = [
        VFLGNNPartyModel(input_dim=2048, party_start=0, party_dim=IMG_PROJ).to(DEVICE),
        VFLGNNPartyModel(input_dim=768,  party_start=IMG_PROJ, party_dim=TXT_PROJ).to(DEVICE),
    ]
    fusion = FusionHead().to(DEVICE)

    opts = [
        torch.optim.AdamW(models[0].parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY),
        torch.optim.AdamW(models[1].parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY),
    ]
    fusion_opt = torch.optim.AdamW(fusion.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    # Loss (ASL)
    loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

    best_val = -1
    best_state = None
    best_thresh = None
    patience = 0
    PATIENCE_MAX = 12

    print(f"\n🚀 Training VFL-GNN + Fusion + ASL (D_TOTAL={D_TOTAL})")
    t0 = time.time()

    for r in range(1, T + 1):
        loss = train_one_round(models, fusion, opts, fusion_opt, train_loaders, loss_fn, r)

        if r == 1 or r % 2 == 0:
            val_res = evaluate(models, fusion, val_loaders, thresholds=None)
            val_thresh = find_optimal_thresholds(val_res["probs"], val_res["labels"])
            val_res_thr = evaluate(models, fusion, val_loaders, thresholds=val_thresh)

            print(f"[Round {r:03d}/{T}] Loss={loss:.4f} | "
                  f"VAL F1-macro={val_res_thr['f1_macro']:.2f}% | "
                  f"VAL F1-micro={val_res_thr['f1_micro']:.2f}% | "
                  f"VAL Hamming={val_res_thr['hamming_acc']:.2f}%")

            if val_res_thr["f1_macro"] > best_val:
                best_val = val_res_thr["f1_macro"]
                best_state = {
                    "m0": {k: v.detach().cpu().clone() for k, v in models[0].state_dict().items()},
                    "m1": {k: v.detach().cpu().clone() for k, v in models[1].state_dict().items()},
                    "fus": {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()},
                }
                best_thresh = val_thresh.clone()
                patience = 0
                print(f"  ✓ New best VAL macro-F1: {best_val:.2f}%")
            else:
                patience += 1
                if patience >= PATIENCE_MAX:
                    print("⏹️ Early stopping (no VAL improvement).")
                    break

    train_time = time.time() - t0

    if best_state is not None:
        models[0].load_state_dict(best_state["m0"])
        models[1].load_state_dict(best_state["m1"])
        fusion.load_state_dict(best_state["fus"])

    test_res = evaluate(models, fusion, test_loaders, thresholds=best_thresh)

    print("\n" + "=" * 90)
    print("🏁 FINAL TEST RESULTS (VFL-GNN + Fusion + ASL)")
    print("=" * 90)
    print(f"TEST F1-Macro:   {test_res['f1_macro']:.2f}%")
    print(f"TEST F1-Micro:   {test_res['f1_micro']:.2f}%")
    print(f"TEST HammingAcc: {test_res['hamming_acc']:.2f}%")
    print(f"Train time:      {train_time/60:.1f} min")
    print("=" * 90)

if __name__ == "__main__":
    run()

"""
🔧 Device: cuda 🔧 GPU: Tesla T4 Parsing MM-IMDB: 100%|██████████| 15552/15552 [05:54<00:00, 43.86it/s] ✓ Samples: 15552 | Avg labels/sample: 2.48 ✓ Split: train=10886 val=1555 test=3111 🔧 Extracting embeddings (pretrained, frozen)... 🖼️ Image embeddings... Images: 100%|██████████| 486/486 [08:44<00:00, 1.08s/it] 📝 Text embeddings... Text: 100%|██████████| 486/486 [01:52<00:00, 4.31it/s] ✓ img_emb: (15552, 2048) | txt_emb: (15552, 768) | labels: (15552, 23) 🚀 Training VFL-GNN + Fusion + ASL (D_TOTAL=768) [Round 001/80] Loss=0.0533 | VAL F1-macro=36.52% | VAL F1-micro=41.60% | VAL Hamming=79.07% ✓ New best VAL macro-F1: 36.52% 
[Round 002/80] Loss=0.0438 | VAL F1-macro=44.20% | VAL F1-micro=54.49% | VAL Hamming=87.82% ✓ New best VAL macro-F1: 44.20% 
[Round 004/80] Loss=0.0399 | VAL F1-macro=48.94% | VAL F1-micro=58.48% | VAL Hamming=89.33% ✓ New best VAL macro-F1: 48.94% 
[Round 006/80] Loss=0.0381 | VAL F1-macro=50.48% | VAL F1-micro=59.16% | VAL Hamming=89.37% ✓ New best VAL macro-F1: 50.48% 
[Round 008/80] Loss=0.0369 | VAL F1-macro=52.61% | VAL F1-micro=60.40% | VAL Hamming=89.85% ✓ New best VAL macro-F1: 52.61% 
[Round 010/80] Loss=0.0360 | VAL F1-macro=52.55% | VAL F1-micro=60.82% | VAL Hamming=90.54% 
[Round 012/80] Loss=0.0354 | VAL F1-macro=54.04% | VAL F1-micro=61.86% | VAL Hamming=90.73% ✓ New best VAL macro-F1: 54.04% 
[Round 014/80] Loss=0.0349 | VAL F1-macro=54.64% | VAL F1-micro=61.93% | VAL Hamming=90.82% ✓ New best VAL macro-F1: 54.64% 
[Round 016/80] Loss=0.0338 | VAL F1-macro=55.37% | VAL F1-micro=62.24% | VAL Hamming=90.61% ✓ New best VAL macro-F1: 55.37% 
[Round 018/80] Loss=0.0332 | VAL F1-macro=55.07% | VAL F1-micro=62.41% | VAL Hamming=90.92% 
[Round 020/80] Loss=0.0325 | VAL F1-macro=54.61% | VAL F1-micro=62.02% | VAL Hamming=90.88% 
[Round 022/80] Loss=0.0319 | VAL F1-macro=54.08% | VAL F1-micro=61.41% | VAL Hamming=90.92% 
[Round 024/80] Loss=0.0309 | VAL F1-macro=54.44% | VAL F1-micro=61.65% | VAL Hamming=90.40% 
[Round 026/80] Loss=0.0299 | VAL F1-macro=54.42% | VAL F1-micro=61.49% | VAL Hamming=90.60% 
[Round 028/80] Loss=0.0289 | VAL F1-macro=54.64% | VAL F1-micro=61.79% | VAL Hamming=90.92% 
[Round 030/80] Loss=0.0277 | VAL F1-macro=54.52% | VAL F1-micro=61.88% | VAL Hamming=90.75% 
[Round 032/80] Loss=0.0269 | VAL F1-macro=54.56% | VAL F1-micro=61.03% | VAL Hamming=90.29% 
[Round 034/80] Loss=0.0262 | VAL F1-macro=53.61% | VAL F1-micro=60.88% | VAL Hamming=90.79% 
[Round 036/80] Loss=0.0253 | VAL F1-macro=55.21% | VAL F1-micro=61.49% | VAL Hamming=90.63% 
[Round 038/80] Loss=0.0247 | VAL F1-macro=53.81% | VAL F1-micro=61.13% | VAL Hamming=90.66% 
[Round 040/80] Loss=0.0234 | VAL F1-macro=55.86% | VAL F1-micro=62.09% | VAL Hamming=90.71% ✓ New best VAL macro-F1: 55.86% 
[Round 042/80] Loss=0.0224 | VAL F1-macro=54.96% | VAL F1-micro=61.53% | VAL Hamming=90.58% 
[Round 044/80] Loss=0.0215 | VAL F1-macro=55.70% | VAL F1-micro=61.66% | VAL Hamming=90.58% 
[Round 046/80] Loss=0.0204 | VAL F1-macro=55.10% | VAL F1-micro=60.89% | VAL Hamming=90.49% 
[Round 048/80] Loss=0.0200 | VAL F1-macro=55.34% | VAL F1-micro=61.02% | VAL Hamming=90.40% 
[Round 050/80] Loss=0.0189 | VAL F1-macro=55.24% | VAL F1-micro=61.64% | VAL Hamming=90.59% 
[Round 052/80] Loss=0.0180 | VAL F1-macro=53.99% | VAL F1-micro=59.32% | VAL Hamming=90.06% 
[Round 054/80] Loss=0.0175 | VAL F1-macro=54.18% | VAL F1-micro=60.17% | VAL Hamming=89.63% 
[Round 056/80] Loss=0.0164 | VAL F1-macro=54.88% | VAL F1-micro=60.63% | VAL Hamming=90.43% 
[Round 058/80] Loss=0.0156 | VAL F1-macro=54.71% | VAL F1-micro=61.33% | VAL Hamming=90.73% 
[Round 060/80] Loss=0.0146 | VAL F1-macro=53.93% | VAL F1-micro=60.36% | VAL Hamming=90.44% 
[Round 062/80] Loss=0.0138 | VAL F1-macro=54.49% | VAL F1-micro=61.30% | VAL Hamming=90.72% 
[Round 064/80] Loss=0.0131 | VAL F1-macro=53.32% | VAL F1-micro=59.93% | VAL Hamming=90.50% 
⏹️ Early stopping (no VAL improvement). 
========================================================================================== 
🏁 FINAL TEST RESULTS (VFL-GNN + Fusion + ASL) 
========================================================================================== 
TEST F1-Macro: 54.87% TEST F1-Micro: 61.77% TEST HammingAcc: 90.61% Train time: 17.3 min

"""