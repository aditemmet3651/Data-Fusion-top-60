"""
Шаг 5a: DAE Pretraining + TabM Neural Network.
VRAM-safe: данные не держатся в RAM/VRAM одновременно с моделью.
"""

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
import joblib
import gc
import os
import time
import warnings
from config import *

warnings.filterwarnings("ignore")

NN_CHECKPOINT_DIR = os.path.join(ARTIFACTS_PATH, "nn_tabm_checkpoints")
DAE_CHECKPOINT = os.path.join(ARTIFACTS_PATH, "dae_model.pt")
DAE_EMBEDDINGS_TRAIN = os.path.join(ARTIFACTS_PATH, "dae_emb_train.npy")
DAE_EMBEDDINGS_TEST = os.path.join(ARTIFACTS_PATH, "dae_emb_test.npy")
NN_OOF_SAVE = os.path.join(ARTIFACTS_PATH, "nn1_oof.npz")
TRAIN_NPY = os.path.join(ARTIFACTS_PATH, "X_train_nn_full.npy")
TEST_NPY = os.path.join(ARTIFACTS_PATH, "X_test_nn_full.npy")
DAE_DATA_NPY = os.path.join(ARTIFACTS_PATH, "dae_all_data.npy")
DAE_NULL_NPY = os.path.join(ARTIFACTS_PATH, "dae_all_null.npy")

os.makedirs(NN_CHECKPOINT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  Improved Denoising Autoencoder
# ══════════════════════════════════════════════════════════════

class GatedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.gate(x))


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=192):
        super().__init__()
        self.encoder = nn.Sequential(
            GatedLinear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2),
            GatedLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, input_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def apply_masked_noise(clean, non_null_mask, rate=0.20):
    mask = (torch.rand_like(clean) < rate) & (non_null_mask > 0.5)
    corrupted = clean.clone()
    corrupted[mask] = 0.0
    return corrupted, mask


def train_dae(device):
    print("\n  ══ DAE Pretraining (Improved) ══")

    if (os.path.exists(DAE_EMBEDDINGS_TRAIN) and
            os.path.exists(DAE_EMBEDDINGS_TEST)):
        print("  DAE embeddings уже есть, загружаю...")
        train_emb = np.load(DAE_EMBEDDINGS_TRAIN)
        test_emb = np.load(DAE_EMBEDDINGS_TEST)
        print(f"  Train: {train_emb.shape}, Test: {test_emb.shape}")
        return train_emb, test_emb

    # ── Собираем данные и сохраняем на диск ──
    if not os.path.exists(DAE_DATA_NPY):
        print("  Собираю данные для DAE...")
        main_schema = pl.scan_parquet(TRAIN_MAIN).collect_schema()
        main_num_cols = [c for c in main_schema.names()
                         if c != "customer_id" and not c.startswith("cat_")]

        if os.path.exists(SELECTED_FEATURES_700):
            extra_cols = joblib.load(SELECTED_FEATURES_700)
            extra_cols = [c for c in extra_cols
                          if c != "customer_id" and not c.startswith("Column_")]
        else:
            extra_schema = pl.scan_parquet(TRAIN_EXTRA).collect_schema()
            extra_cols = [c for c in extra_schema.names() if c != "customer_id"]

        actual_extra = set(pl.scan_parquet(TRAIN_EXTRA).collect_schema().names())
        extra_cols = [c for c in extra_cols if c in actual_extra]
        extra_cols = extra_cols[:DAE_EXTRA_LIMIT]

        n_dae_features = len(main_num_cols) + len(extra_cols)
        print(f"  DAE features: {len(main_num_cols)} main + "
              f"{len(extra_cols)} extra = {n_dae_features}")

        print("  Загружаю train...")
        tr_main = pl.read_parquet(TRAIN_MAIN, columns=["customer_id"] + main_num_cols)
        tr_extra = pl.read_parquet(TRAIN_EXTRA, columns=["customer_id"] + extra_cols)
        if tr_main["customer_id"].to_list() == tr_extra["customer_id"].to_list():
            tr_data = pl.concat([tr_main.drop("customer_id"),
                                  tr_extra.drop("customer_id")], how="horizontal")
        else:
            tr_data = tr_main.join(tr_extra, on="customer_id", how="left").drop("customer_id")
        del tr_main, tr_extra; gc.collect()
        train_np = tr_data.to_numpy().astype(np.float32)
        n_train = len(train_np)
        del tr_data; gc.collect()

        print("  Загружаю test...")
        te_main = pl.read_parquet(TEST_MAIN, columns=["customer_id"] + main_num_cols)
        te_extra = pl.read_parquet(TEST_EXTRA, columns=["customer_id"] + extra_cols)
        if te_main["customer_id"].to_list() == te_extra["customer_id"].to_list():
            te_data = pl.concat([te_main.drop("customer_id"),
                                  te_extra.drop("customer_id")], how="horizontal")
        else:
            te_data = te_main.join(te_extra, on="customer_id", how="left").drop("customer_id")
        del te_main, te_extra; gc.collect()
        test_np = te_data.to_numpy().astype(np.float32)
        n_test = len(test_np)
        del te_data; gc.collect()

        all_data = np.vstack([train_np, test_np])
        all_null = np.isnan(all_data).astype(np.float32)
        all_data = np.nan_to_num(all_data, nan=0.0)
        del train_np, test_np; gc.collect()

        dae_mean = all_data.mean(axis=0)
        dae_std = all_data.std(axis=0)
        dae_std[dae_std < 1e-8] = 1.0
        all_data = ((all_data - dae_mean) / dae_std).astype(np.float32)

        # Сохраняем на диск и освобождаем RAM
        np.save(DAE_DATA_NPY, all_data)
        np.save(DAE_NULL_NPY, all_null)
        joblib.dump({
            "n_train": n_train, "n_test": n_test,
            "n_dae_features": n_dae_features,
            "dae_mean": dae_mean, "dae_std": dae_std,
        }, os.path.join(ARTIFACTS_PATH, "dae_data_meta.pkl"))

        print(f"  DAE data saved: {all_data.shape}, "
              f"{all_data.nbytes / 1e9:.2f} ГБ")
        del all_data, all_null; gc.collect()
    else:
        print("  DAE данные уже на диске")

    # ── Загружаем метаданные ──
    dae_meta = joblib.load(os.path.join(ARTIFACTS_PATH, "dae_data_meta.pkl"))
    n_train = dae_meta["n_train"]
    n_test = dae_meta["n_test"]
    n_dae_features = dae_meta["n_dae_features"]
    n_total = n_train + n_test

    # ── Обучение DAE (данные через mmap) ──
    if not os.path.exists(DAE_CHECKPOINT):
        print(f"  Обучаю DAE: {n_dae_features} features, "
              f"bottleneck={DAE_BOTTLENECK}...")

        # Очищаем всё перед созданием модели
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model = DenoisingAutoencoder(n_dae_features, DAE_BOTTLENECK)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  DAE параметров: {n_params:,}")

        if device.type == "cuda":
            print(f"  VRAM after model: "
                  f"{torch.cuda.memory_allocated() / 1e6:.0f} MB")

        optimizer = torch.optim.AdamW(model.parameters(), lr=DAE_LR,
                                       weight_decay=DAE_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=DAE_EPOCHS)
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)

        # Используем mmap — не грузим в RAM целиком
        data_mmap = np.load(DAE_DATA_NPY, mmap_mode='r')
        null_mmap = np.load(DAE_NULL_NPY, mmap_mode='r')

        n_batches = (n_total + DAE_BATCH_SIZE - 1) // DAE_BATCH_SIZE

        for epoch in range(DAE_EPOCHS):
            model.train()
            total_loss = 0.0
            perm_indices = np.random.permutation(n_total)

            for b in range(n_batches):
                start = b * DAE_BATCH_SIZE
                end = min(start + DAE_BATCH_SIZE, n_total)
                idx = perm_indices[start:end]
                idx_sorted = np.sort(idx)  # mmap быстрее с sorted indices

                clean_np = data_mmap[idx_sorted].astype(np.float32)
                null_np = null_mmap[idx_sorted].astype(np.float32)

                clean = torch.from_numpy(clean_np).to(device)
                non_null = torch.from_numpy(1.0 - null_np).float().to(device)
                del clean_np, null_np

                with torch.autocast(device_type=device.type,
                                     dtype=torch.float16, enabled=use_amp):
                    corrupted, mask = apply_masked_noise(
                        clean, non_null, DAE_CORRUPTION)
                    reconstructed = model(corrupted)
                    diff_sq = (reconstructed.float() - clean.float()) ** 2
                    weight = torch.ones_like(diff_sq)
                    weight[mask] = 3.0
                    weight = weight * non_null
                    loss = (diff_sq * weight).sum() / weight.sum().clamp(min=1)

                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                total_loss += loss.item()

                del clean, non_null, corrupted, mask, reconstructed, diff_sq, weight

            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Ep {epoch + 1:2d}/{DAE_EPOCHS}  "
                      f"loss={total_loss / n_batches:.6f}")

        del data_mmap, null_mmap, optimizer, scheduler, scaler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        torch.save({"model_state_dict": model.state_dict(),
                     "input_dim": n_dae_features,
                     "bottleneck_dim": DAE_BOTTLENECK}, DAE_CHECKPOINT)
        print(f"  DAE сохранён: {DAE_CHECKPOINT}")
    else:
        print(f"  Загружаю DAE из {DAE_CHECKPOINT}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = DenoisingAutoencoder(n_dae_features, DAE_BOTTLENECK)
        model = model.to(device)
        ckpt = torch.load(DAE_CHECKPOINT, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        del ckpt

    # ── Извлекаем embeddings батчами через mmap ──
    print("  Извлекаю DAE embeddings...")
    model.eval()
    data_mmap = np.load(DAE_DATA_NPY, mmap_mode='r')

    train_emb = np.empty((n_train, DAE_BOTTLENECK), dtype=np.float32)
    test_emb = np.empty((n_test, DAE_BOTTLENECK), dtype=np.float32)

    with torch.no_grad():
        # Train embeddings
        for start in range(0, n_train, DAE_BATCH_SIZE):
            end = min(start + DAE_BATCH_SIZE, n_train)
            batch = torch.from_numpy(
                data_mmap[start:end].astype(np.float32)
            ).to(device)
            emb = model.encode(batch).cpu().numpy()
            train_emb[start:end] = emb
            del batch, emb

        # Test embeddings
        for start in range(0, n_test, DAE_BATCH_SIZE):
            end = min(start + DAE_BATCH_SIZE, n_test)
            batch = torch.from_numpy(
                data_mmap[n_train + start:n_train + end].astype(np.float32)
            ).to(device)
            emb = model.encode(batch).cpu().numpy()
            test_emb[start:end] = emb
            del batch, emb

    del data_mmap

    np.save(DAE_EMBEDDINGS_TRAIN, train_emb)
    np.save(DAE_EMBEDDINGS_TEST, test_emb)
    print(f"  Embeddings: train {train_emb.shape}, test {test_emb.shape}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Cleanup temp files
    for f in [DAE_DATA_NPY, DAE_NULL_NPY]:
        if os.path.exists(f):
            os.remove(f)
            print(f"  Удалён: {f}")

    return train_emb, test_emb


# ══════════════════════════════════════════════════════════════
#  TabM model
# ══════════════════════════════════════════════════════════════

class PiecewiseLinearEncoding(nn.Module):
    def __init__(self, n_features, n_bins=24):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.register_buffer('edges', torch.zeros(n_features, n_bins + 1))
        self.weight = nn.Parameter(torch.empty(n_features, n_bins))
        self.bias = nn.Parameter(torch.zeros(n_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def set_bins(self, X_np):
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        for i in range(self.n_features):
            col = X_np[:, i]
            col_valid = col[~np.isnan(col)] if np.any(np.isnan(col)) else col
            if len(col_valid) < self.n_bins + 1:
                edges = np.linspace(-3.0, 3.0, self.n_bins + 1)
            else:
                edges = np.unique(np.quantile(col_valid, quantiles))
                if len(edges) < self.n_bins + 1:
                    edges = np.linspace(edges[0] - 1e-6, edges[-1] + 1e-6,
                                         self.n_bins + 1)
                else:
                    edges = edges[:self.n_bins + 1]
            self.edges[i] = torch.from_numpy(edges.astype(np.float32))

    def forward(self, x):
        left = self.edges[:, :-1]
        right = self.edges[:, 1:]
        width = (right - left).clamp(min=1e-8)
        ple = ((x.unsqueeze(2) - left) / width).clamp(0, 1)
        return (ple * self.weight).sum(dim=2) + self.bias


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        pos_term = (1 - p) ** self.gamma_pos * torch.log(p.clamp(min=self.eps))
        p_m = (p - self.clip).clamp(min=self.eps) if self.clip > 0 else p
        neg_term = p_m ** self.gamma_neg * torch.log((1 - p_m).clamp(min=self.eps))
        return (-(targets * pos_term) - (1 - targets) * neg_term).mean()


class LinearBE(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.k = k
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.r = nn.Parameter(torch.ones(k, in_dim))
        self.s = nn.Parameter(torch.ones(k, out_dim))
        self.b = nn.Parameter(torch.zeros(k, out_dim))

    def forward(self, x):
        x = x * self.r.unsqueeze(0)
        B, k, D = x.shape
        x = self.linear(x.reshape(B * k, D)).reshape(B, k, -1)
        return x * self.s.unsqueeze(0) + self.b.unsqueeze(0)


class ResidualBlockBE(nn.Module):
    def __init__(self, dim, k, dropout=0.3):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.lin1 = LinearBE(dim, dim, k)
        self.bn2 = nn.BatchNorm1d(dim)
        self.lin2 = LinearBE(dim, dim, k)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()
        nn.init.zeros_(self.lin2.linear.weight)

    def forward(self, x):
        B, k, D = x.shape
        h = self.bn1(x.reshape(B * k, D)).reshape(B, k, D)
        h = self.act(h)
        h = self.lin1(h)
        B, k, D = h.shape
        h = self.bn2(h.reshape(B * k, D)).reshape(B, k, D)
        h = self.act(h)
        h = self.drop(h)
        h = self.lin2(h)
        return x + h


class TabMModel(nn.Module):
    def __init__(self, n_features, n_targets, hidden_dim=256,
                 k=8, n_blocks=3, dropout=0.35, plr_bins=24):
        super().__init__()
        self.k = k
        self.n_targets = n_targets
        self.plr = PiecewiseLinearEncoding(n_features, plr_bins)
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlockBE(hidden_dim, k, dropout=dropout)
            for _ in range(n_blocks)
        ])
        head_dim = hidden_dim // 2
        self.head_bn1 = nn.BatchNorm1d(hidden_dim)
        self.head_drop1 = nn.Dropout(dropout * 0.6)
        self.head_lin1 = LinearBE(hidden_dim, head_dim, k)
        self.head_bn2 = nn.BatchNorm1d(head_dim)
        self.head_drop2 = nn.Dropout(dropout * 0.3)
        self.head_lin2 = LinearBE(head_dim, n_targets, k)
        self.act = nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.input_proj:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.head_lin1.linear.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.head_lin2.linear.weight, nonlinearity='relu')

    def _project(self, x):
        x = self.plr(x)
        x = self.input_proj(x)
        return x.unsqueeze(1).expand(-1, self.k, -1)

    def _head(self, x):
        B, k, D = x.shape
        x = self.head_bn1(x.reshape(B * k, D)).reshape(B, k, D)
        x = self.act(x); x = self.head_drop1(x); x = self.head_lin1(x)
        B, k, D2 = x.shape
        x = self.head_bn2(x.reshape(B * k, D2)).reshape(B, k, D2)
        x = self.act(x); x = self.head_drop2(x)
        return self.head_lin2(x)

    def forward(self, x):
        x = self._project(x)
        for block in self.res_blocks:
            x = block(x)
        logits = self._head(x)
        if self.training:
            return logits
        return torch.sigmoid(logits).mean(dim=1)

    def forward_mixup(self, x, lam, perm, mixup_layer):
        x = self._project(x)
        if mixup_layer == 0:
            x = lam * x + (1 - lam) * x[perm]
        for i, block in enumerate(self.res_blocks):
            x = block(x)
            if mixup_layer == i + 1:
                x = lam * x + (1 - lam) * x[perm]
        return self._head(x)


# ══════════════════════════════════════════════════════════════
#  Training utilities
# ══════════════════════════════════════════════════════════════

def train_epoch_tabm(model, loader, optimizer, criterion, device,
                     scaler, mixup_alpha=0.2):
    model.train()
    total_loss = 0
    n_batches = 0
    k = model.k
    n_layers = len(model.res_blocks) + 1

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mixup_layer = np.random.randint(0, n_layers)
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        lam = max(lam, 1 - lam)
        perm = torch.randperm(x.shape[0], device=device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                             enabled=device.type == "cuda"):
            logits = model.forward_mixup(x, lam, perm, mixup_layer)
            y_k = y.unsqueeze(1).expand(-1, k, -1)
            y_mixed = lam * y_k + (1 - lam) * y[perm].unsqueeze(1).expand(-1, k, -1)
            loss = criterion(logits, y_mixed)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TABM_GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict_tabm(model, loader, device):
    model.eval()
    all_preds = []
    for batch in loader:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        probs = model(x)
        all_preds.append(probs.cpu().numpy())
    return np.vstack(all_preds)


@torch.no_grad()
def predict_numpy_batched(model, X_np, device, batch_size=1024):
    model.eval()
    n = len(X_np)
    all_preds = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = torch.from_numpy(X_np[start:end]).to(device)
        probs = model(x)
        all_preds.append(probs.cpu().numpy())
        del x
    return np.vstack(all_preds)


def compute_macro_auc(y_true, y_pred, n_targets):
    aucs = []
    for t in range(n_targets):
        if len(np.unique(y_true[:, t])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, t], y_pred[:, t]))
        except Exception:
            pass
    return np.mean(aucs) if aucs else 0.0


def fit_quantile_transformer(train_npy_path, tr_idx, n_features):
    qt = QuantileTransformer(
        n_quantiles=min(1000, len(tr_idx)),
        output_distribution="normal",
        subsample=min(100_000, len(tr_idx)),
        random_state=SEED
    )
    X_all = np.load(train_npy_path, mmap_mode='r')
    X_tr = X_all[tr_idx].astype(np.float32)
    qt.fit(X_tr)
    del X_tr
    return qt


def transform_batched(qt, npy_path, indices, batch_size=100_000):
    X_mmap = np.load(npy_path, mmap_mode='r')
    n = len(indices) if indices is not None else X_mmap.shape[0]
    n_features = X_mmap.shape[1]
    result = np.empty((n, n_features), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if indices is not None:
            batch = X_mmap[indices[start:end]].astype(np.float32)
        else:
            batch = X_mmap[start:end].astype(np.float32)
        result[start:end] = np.nan_to_num(
            qt.transform(batch), nan=0.0
        ).astype(np.float32)
        del batch

    del X_mmap
    return result


# ══════════════════════════════════════════════════════════════
#  Data Preparation
# ══════════════════════════════════════════════════════════════

def prepare_and_save_data(train_emb, test_emb):
    print("\n  Готовлю данные для NN...")

    if os.path.exists(TRAIN_NPY) and os.path.exists(TEST_NPY):
        n_features = np.load(TRAIN_NPY, mmap_mode='r').shape[1]
        print(f"  Данные уже на диске: {n_features} features")
        return n_features

    top_feats = joblib.load(SELECTED_FEATURES_300)
    top_feats = [f for f in top_feats
                 if f != "customer_id" and not f.startswith("Column_")]
    actual_cols = set(pl.scan_parquet(TRAIN_EXTRA).collect_schema().names())
    top_feats = [f for f in top_feats if f in actual_cols]
    top_feats = top_feats[:TABM_EXTRA_LIMIT]

    def di(df):
        return df.drop("customer_id") if "customer_id" in df.columns else df

    def ensure_order(df, ref_ids, name):
        if "customer_id" in df.columns:
            if df["customer_id"].to_list() != ref_ids:
                ref_df = pl.DataFrame({"customer_id": ref_ids})
                return ref_df.join(df, on="customer_id", how="left")
        return df

    # === TRAIN ===
    print("  Собираю TRAIN...")
    df_train_main = pl.read_parquet(TRAIN_MAIN)
    train_id_ref = df_train_main["customer_id"].to_list()

    df_train_extra = pl.read_parquet(TRAIN_EXTRA,
                                      columns=["customer_id"] + top_feats)
    meta_train = pl.read_parquet(META_TRAIN)
    aggs_train = pl.read_parquet(GLOBAL_AGGS_TRAIN)

    df_train_extra = ensure_order(df_train_extra, train_id_ref, "train_extra")
    meta_train = ensure_order(meta_train, train_id_ref, "meta_train")
    aggs_train = ensure_order(aggs_train, train_id_ref, "aggs_train")

    parts_train = [di(df_train_main), di(df_train_extra),
                   di(aggs_train), di(meta_train)]

    if os.path.exists(FE_TRAIN):
        fe_tr = pl.read_parquet(FE_TRAIN)
        fe_tr = ensure_order(fe_tr, train_id_ref, "fe_train")
        parts_train.append(di(fe_tr))
        del fe_tr

    if os.path.exists(OOF_TE_TRAIN):
        te_tr = pl.read_parquet(OOF_TE_TRAIN)
        te_tr = ensure_order(te_tr, train_id_ref, "te_train")
        parts_train.append(di(te_tr))
        del te_tr
        print("  ✅ OOF TE включены")

    dae_cols = [f"dae_{i}" for i in range(train_emb.shape[1])]
    parts_train.append(pl.DataFrame(
        {c: train_emb[:, i] for i, c in enumerate(dae_cols)}
    ))

    del df_train_main, df_train_extra, aggs_train, meta_train
    gc.collect()

    X_train_pl = pl.concat(parts_train, how="horizontal")
    del parts_train; gc.collect()

    cat_cols = [c for c in X_train_pl.columns if c.startswith("cat_")]

    df_test_main = pl.read_parquet(TEST_MAIN)
    test_id_ref = df_test_main["customer_id"].to_list()
    test_cats = pl.read_parquet(TEST_MAIN, columns=cat_cols)

    for c in cat_cols:
        combined = pl.concat([X_train_pl.select(c), test_cats.select(c)])
        freq = (combined.group_by(c)
                .agg(pl.len().alias("cnt"))
                .with_columns(
                    (pl.col("cnt") / len(combined))
                    .cast(pl.Float32).alias(c + "_freq")))
        X_train_pl = (X_train_pl.join(freq.select([c, c + "_freq"]),
                                       on=c, how="left")
                      .drop(c).rename({c + "_freq": c}))
        del combined, freq

    del test_cats; gc.collect()

    cast_exprs = [pl.col(c).cast(pl.Float32).fill_null(0.0).alias(c)
                  for c in X_train_pl.columns]
    X_train_pl = X_train_pl.with_columns(cast_exprs)

    feature_names = list(X_train_pl.columns)
    n_features = len(feature_names)

    print(f"  Сохраняю train: {n_features} features...")
    X_train_np = X_train_pl.to_numpy().astype(np.float32)
    np.save(TRAIN_NPY, X_train_np)
    print(f"  Train: {X_train_np.shape}, {X_train_np.nbytes / 1e9:.2f} ГБ")
    del X_train_np, X_train_pl; gc.collect()

    # === TEST ===
    print("  Собираю TEST...")
    df_test_extra = pl.read_parquet(TEST_EXTRA,
                                     columns=["customer_id"] + top_feats)
    meta_test = pl.read_parquet(META_TEST)
    aggs_test = pl.read_parquet(GLOBAL_AGGS_TEST)

    df_test_extra = ensure_order(df_test_extra, test_id_ref, "test_extra")
    meta_test = ensure_order(meta_test, test_id_ref, "meta_test")
    aggs_test = ensure_order(aggs_test, test_id_ref, "aggs_test")

    parts_test = [di(df_test_main), di(df_test_extra),
                  di(aggs_test), di(meta_test)]

    if os.path.exists(FE_TEST):
        fe_te = pl.read_parquet(FE_TEST)
        fe_te = ensure_order(fe_te, test_id_ref, "fe_test")
        parts_test.append(di(fe_te))
        del fe_te

    if os.path.exists(OOF_TE_TEST):
        te_te = pl.read_parquet(OOF_TE_TEST)
        te_te = ensure_order(te_te, test_id_ref, "te_test")
        parts_test.append(di(te_te))
        del te_te

    parts_test.append(pl.DataFrame(
        {c: test_emb[:, i] for i, c in enumerate(dae_cols)}
    ))

    del df_test_main, df_test_extra, aggs_test, meta_test
    gc.collect()

    X_test_pl = pl.concat(parts_test, how="horizontal")
    del parts_test; gc.collect()

    train_main_cats = pl.read_parquet(TRAIN_MAIN, columns=cat_cols)
    test_main_cats = pl.read_parquet(TEST_MAIN, columns=cat_cols)

    for c in cat_cols:
        combined = pl.concat([train_main_cats.select(c),
                               test_main_cats.select(c)])
        freq = (combined.group_by(c)
                .agg(pl.len().alias("cnt"))
                .with_columns(
                    (pl.col("cnt") / len(combined))
                    .cast(pl.Float32).alias(c + "_freq")))
        X_test_pl = (X_test_pl.join(freq.select([c, c + "_freq"]),
                                     on=c, how="left")
                     .drop(c).rename({c + "_freq": c}))
        del combined, freq

    del train_main_cats, test_main_cats; gc.collect()

    X_test_pl = X_test_pl.select(feature_names)
    cast_exprs = [pl.col(c).cast(pl.Float32).fill_null(0.0).alias(c)
                  for c in feature_names]
    X_test_pl = X_test_pl.with_columns(cast_exprs)

    X_test_np = X_test_pl.to_numpy().astype(np.float32)
    np.save(TEST_NPY, X_test_np)
    print(f"  Test: {X_test_np.shape}, {X_test_np.nbytes / 1e9:.2f} ГБ")
    del X_test_np, X_test_pl; gc.collect()

    joblib.dump({
        "feature_names": feature_names,
        "train_ids": train_id_ref,
        "test_ids": test_id_ref,
    }, os.path.join(ARTIFACTS_PATH, "nn_data_meta.pkl"))

    return n_features


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 60)
    print("ШАГ 5: DAE + TabM (Improved)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()

    # Phase 1: DAE
    train_emb, test_emb = train_dae(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Phase 2: Prepare data
    n_features = prepare_and_save_data(train_emb, test_emb)
    del train_emb, test_emb; gc.collect()

    nn_meta = joblib.load(os.path.join(ARTIFACTS_PATH, "nn_data_meta.pkl"))
    train_id_ref = nn_meta["train_ids"]
    test_id_ref = nn_meta["test_ids"]

    target_df = pd.read_parquet(TRAIN_TARGET)
    target_cols = [c for c in target_df.columns if c.startswith("target")]
    target_df = target_df.set_index("customer_id").loc[train_id_ref].reset_index()
    y_train = target_df[target_cols].values.astype(np.float32)
    del target_df; gc.collect()

    n_train = len(y_train)
    n_targets = len(target_cols)

    test_mmap = np.load(TEST_NPY, mmap_mode='r')
    n_test = test_mmap.shape[0]
    del test_mmap

    print(f"\n  n_train={n_train:,}, n_test={n_test:,}, "
          f"n_features={n_features}, n_targets={n_targets}")

    y_strat = (y_train.sum(axis=1) > 0).astype(np.int32)
    skf = StratifiedKFold(n_splits=N_SPLITS_FINAL, shuffle=True,
                           random_state=SEED)
    folds = list(skf.split(np.zeros(n_train), y_strat))

    oof_preds = np.zeros((n_train, n_targets), dtype=np.float32)
    test_preds_sum = np.zeros((n_test, n_targets), dtype=np.float32)
    fold_aucs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        ckpt_path = os.path.join(NN_CHECKPOINT_DIR, f"fold_{fold_idx}.npz")
        if os.path.exists(ckpt_path):
            print(f"\n  Fold {fold_idx + 1}: загружаю чекпоинт")
            d = np.load(ckpt_path)
            oof_preds[va_idx] = d["val_preds"]
            test_preds_sum += d["test_preds"]
            fold_aucs.append(float(d["fold_auc"]))
            print(f"    AUC: {d['fold_auc']:.4f}")
            continue

        print(f"\n  ══ Fold {fold_idx + 1}/{N_SPLITS_FINAL} "
              f"(train={len(tr_idx):,}, val={len(va_idx):,}) ══")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # QT
        print("    Fitting QT...")
        qt = fit_quantile_transformer(TRAIN_NPY, tr_idx, n_features)

        print("    Transforming train fold...")
        X_tr = transform_batched(qt, TRAIN_NPY, tr_idx, batch_size=150_000)
        gc.collect()

        print("    Transforming val fold...")
        X_va = transform_batched(qt, TRAIN_NPY, va_idx, batch_size=150_000)
        gc.collect()

        # НЕ трансформируем test сейчас!

        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]

        # Модель на CPU → set_bins → GPU
        model = TabMModel(
            n_features, n_targets,
            hidden_dim=TABM_HIDDEN_DIM, k=TABM_K_ENSEMBLE,
            n_blocks=TABM_N_RES_BLOCKS, dropout=TABM_DROPOUT,
            plr_bins=TABM_PLR_N_BINS
        )
        model.plr.set_bins(X_tr)

        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = model.to(device)

        if device.type == "cuda":
            print(f"    VRAM after model: "
                  f"{torch.cuda.memory_allocated() / 1e6:.0f} MB")

        if fold_idx == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    Параметров: {n_params:,}")

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=TABM_BATCH_SIZE, shuffle=True, drop_last=True,
            pin_memory=False, num_workers=0
        )
        del X_tr; gc.collect()

        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
            batch_size=TABM_BATCH_SIZE * 2, shuffle=False,
            pin_memory=False, num_workers=0
        )

        criterion = AsymmetricLoss(ASL_GAMMA_NEG, ASL_GAMMA_POS, ASL_CLIP)
        optimizer = torch.optim.AdamW(model.parameters(), lr=TABM_LR,
                                       weight_decay=TABM_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        best_auc = 0
        patience_counter = 0
        top_states = []

        for epoch in range(TABM_EPOCHS):
            t_ep = time.time()
            train_loss = train_epoch_tabm(
                model, train_loader, optimizer, criterion, device,
                scaler, mixup_alpha=TABM_MIXUP_ALPHA
            )
            val_preds_epoch = predict_tabm(model, val_loader, device)
            macro_auc = compute_macro_auc(y_va, val_preds_epoch, n_targets)
            scheduler.step(macro_auc)

            improved = ""
            if macro_auc > best_auc:
                best_auc = macro_auc
                patience_counter = 0
                improved = " *"
            else:
                patience_counter += 1

            state_copy = {kk: v.cpu().clone()
                          for kk, v in model.state_dict().items()}
            top_states.append((macro_auc, state_copy))
            top_states.sort(key=lambda x: x[0], reverse=True)
            if len(top_states) > TABM_TOP_K_SWA:
                top_states.pop()

            ep_time = time.time() - t_ep
            if (epoch + 1) % 5 == 0 or improved:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"    Ep {epoch + 1:2d}/{TABM_EPOCHS}  "
                      f"loss={train_loss:.4f}  val_auc={macro_auc:.4f}  "
                      f"lr={cur_lr:.1e}  {ep_time:.1f}s{improved}")

            if patience_counter >= TABM_PATIENCE:
                print(f"    Early stop at epoch {epoch + 1}")
                break

        # SWA
        avg_state = {}
        for key in top_states[0][1]:
            avg_state[key] = sum(s[key] for _, s in top_states) / len(top_states)
        model.load_state_dict(avg_state)
        model.to(device)

        # Val predictions (SWA)
        val_preds = predict_tabm(model, val_loader, device)
        swa_auc = compute_macro_auc(y_va, val_preds, n_targets)
        print(f"  Fold {fold_idx + 1}: best={best_auc:.4f}, SWA={swa_auc:.4f}")

        oof_preds[va_idx] = val_preds

        # Освобождаем train/val ПЕРЕД test
        del (train_loader, val_loader, optimizer, scheduler, scaler,
             criterion, top_states, avg_state, X_va, y_tr, y_va,
             val_preds_epoch, val_preds)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Test: transform + predict батчами
        print("    Predicting test...")
        X_test_mmap = np.load(TEST_NPY, mmap_mode='r')
        test_batch = 50_000
        test_preds_fold = np.zeros((n_test, n_targets), dtype=np.float32)

        for start in range(0, n_test, test_batch):
            end = min(start + test_batch, n_test)
            chunk_raw = X_test_mmap[start:end].astype(np.float32)
            chunk_qt = np.nan_to_num(
                qt.transform(chunk_raw), nan=0.0
            ).astype(np.float32)
            del chunk_raw

            test_preds_fold[start:end] = predict_numpy_batched(
                model, chunk_qt, device, batch_size=TABM_BATCH_SIZE * 2
            )
            del chunk_qt; gc.collect()

        del X_test_mmap
        test_preds_sum += test_preds_fold
        fold_aucs.append(swa_auc)

        np.savez(ckpt_path,
                 val_idx=va_idx,
                 val_preds=oof_preds[va_idx],
                 test_preds=test_preds_fold,
                 fold_auc=swa_auc)

        del model, test_preds_fold, qt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    test_preds_avg = test_preds_sum / N_SPLITS_FINAL
    nn_oof_auc = compute_macro_auc(y_train, oof_preds, n_targets)

    print(f"\n{'=' * 60}")
    print(f"  TabM OOF Macro AUC: {nn_oof_auc:.4f}")
    print(f"  Per-fold: {[f'{a:.4f}' for a in fold_aucs]}")

    np.savez(NN_OOF_SAVE,
             oof_preds=oof_preds,
             test_preds=test_preds_avg,
             target_cols=target_cols,
             train_ids=train_id_ref,
             test_ids=test_id_ref)
    print(f"  Сохранено: {NN_OOF_SAVE}")

    for f in [TRAIN_NPY, TEST_NPY]:
        if os.path.exists(f):
            os.remove(f)

    elapsed = (time.time() - t0) / 60
    print(f"\n  Готово за {elapsed:.1f} мин!")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()