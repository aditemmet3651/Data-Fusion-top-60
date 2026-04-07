"""
Шаг 5: Обучение нейросети (MLP + PLR + ASL).
Третья модель для ансамбля с LGBM + CatBoost.
Включает FE + OOF TE features (с лимитом по RAM/VRAM).
VRAM-safe: test предсказывается ПОСЛЕ обучения.
Сохраняет OOF и test predictions в artifacts/nn_oof.npz
"""

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
import joblib
import gc
import os
import warnings
from config import *

warnings.filterwarnings("ignore")

# ── Гиперпараметры ──
BATCH_SIZE = 512
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-3
PATIENCE = 8
GRAD_CLIP = 1.0
HIDDEN_DIM = 256
DROPOUT = 0.3
PLR_N_BINS = 24

ASL_GAMMA_NEG = 4
ASL_GAMMA_POS = 1
ASL_CLIP = 0.05

TOP_K_SWA = 3

MAX_TOTAL_FEATURES = 750
MLP_EXTRA_LIMIT = 350
MLP_TE_LIMIT = 80
MLP_FE_LIMIT = 100

NN_CHECKPOINT_DIR = os.path.join(ARTIFACTS_PATH, "nn_checkpoints")
NN_OOF_PATH = os.path.join(ARTIFACTS_PATH, "nn_oof.npz")

os.makedirs(NN_CHECKPOINT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  Модули
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
        for i in range(self.n_features):
            col = X_np[:, i]
            col_valid = col[~np.isnan(col)]
            if len(col_valid) < 2:
                edges = np.linspace(-3.0, 3.0, self.n_bins + 1)
            else:
                quantiles = np.linspace(0, 100, self.n_bins + 1)
                edges = np.unique(np.percentile(col_valid, quantiles))
                if len(edges) < self.n_bins + 1:
                    edges = np.linspace(edges[0], edges[-1], self.n_bins + 1)
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
        pos_term = ((1 - p) ** self.gamma_pos *
                    torch.log(p.clamp(min=self.eps)))
        p_m = (p - self.clip).clamp(min=self.eps) if self.clip > 0 else p
        neg_term = (p_m ** self.gamma_neg *
                    torch.log((1 - p_m).clamp(min=self.eps)))
        return (-(targets * pos_term) - (1 - targets) * neg_term).mean()


class TabMLPModel(nn.Module):
    def __init__(self, n_numerical, n_targets):
        super().__init__()
        self.plr = PiecewiseLinearEncoding(n_numerical, PLR_N_BINS)
        self.net = nn.Sequential(
            nn.Linear(n_numerical, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.BatchNorm1d(HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Dropout(DROPOUT * 0.7),

            nn.Linear(HIDDEN_DIM // 2, n_targets),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.plr(x)
        logits = self.net(x)
        if self.training:
            return logits
        return torch.sigmoid(logits)


# ══════════════════════════════════════════════════════════════
#  Утилиты
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()
    all_preds = []
    for batch in loader:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        probs = model(x)
        all_preds.append(probs.cpu().numpy())
    return np.vstack(all_preds)


@torch.no_grad()
def predict_numpy_batched(model, X_np, device, batch_size=1024):
    """Предсказание из numpy батчами без DataLoader."""
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
        aucs.append(roc_auc_score(y_true[:, t], y_pred[:, t]))
    return np.mean(aucs) if aucs else 0.0


# ══════════════════════════════════════════════════════════════
#  Подготовка данных
# ══════════════════════════════════════════════════════════════

def prepare_data():
    print("  Загружаю данные...")

    top_feats = joblib.load(SELECTED_FEATURES_300)
    top_feats = [f for f in top_feats
                 if f != "customer_id" and not f.startswith("Column_")]
    actual_cols = set(pl.scan_parquet(TRAIN_EXTRA).collect_schema().names())
    top_feats = [f for f in top_feats if f in actual_cols]
    top_feats = top_feats[:MLP_EXTRA_LIMIT]
    print(f"  Extra features: {len(top_feats)}")

    df_train_main = pl.read_parquet(TRAIN_MAIN)
    df_test_main = pl.read_parquet(TEST_MAIN)
    train_id_ref = df_train_main["customer_id"].to_list()
    test_id_ref = df_test_main["customer_id"].to_list()

    df_train_extra = pl.read_parquet(
        TRAIN_EXTRA, columns=["customer_id"] + top_feats)
    df_test_extra = pl.read_parquet(
        TEST_EXTRA, columns=["customer_id"] + top_feats)
    meta_train = pl.read_parquet(META_TRAIN)
    meta_test = pl.read_parquet(META_TEST)
    aggs_train = pl.read_parquet(GLOBAL_AGGS_TRAIN)
    aggs_test = pl.read_parquet(GLOBAL_AGGS_TEST)

    def di(df):
        return df.drop("customer_id") if "customer_id" in df.columns else df

    def ensure_order(df, ref_ids, name):
        if "customer_id" in df.columns:
            if df["customer_id"].to_list() != ref_ids:
                print(f"  ⚠️ {name}: пересортировываю")
                ref_df = pl.DataFrame({"customer_id": ref_ids})
                return ref_df.join(df, on="customer_id", how="left")
        return df

    df_train_extra = ensure_order(df_train_extra, train_id_ref, "train_extra")
    df_test_extra = ensure_order(df_test_extra, test_id_ref, "test_extra")
    meta_train = ensure_order(meta_train, train_id_ref, "meta_train")
    meta_test = ensure_order(meta_test, test_id_ref, "meta_test")
    aggs_train = ensure_order(aggs_train, train_id_ref, "aggs_train")
    aggs_test = ensure_order(aggs_test, test_id_ref, "aggs_test")

    parts_train = [di(df_train_main), di(df_train_extra),
                   di(aggs_train), di(meta_train)]
    parts_test = [di(df_test_main), di(df_test_extra),
                  di(aggs_test), di(meta_test)]

    del (df_train_main, df_test_main, df_train_extra, df_test_extra,
         aggs_train, aggs_test, meta_train, meta_test)
    gc.collect()

    # FE
    if os.path.exists(FE_TRAIN) and os.path.exists(FE_TEST):
        fe_tr = pl.read_parquet(FE_TRAIN)
        fe_te = pl.read_parquet(FE_TEST)
        fe_tr = ensure_order(fe_tr, train_id_ref, "fe_train")
        fe_te = ensure_order(fe_te, test_id_ref, "fe_test")
        fe_tr_noid = di(fe_tr)
        fe_te_noid = di(fe_te)
        if len(fe_tr_noid.columns) > MLP_FE_LIMIT:
            stds = []
            sample = fe_tr_noid[:50_000]
            for c in fe_tr_noid.columns:
                stds.append((c, sample[c].to_numpy().astype(np.float32).std()))
            stds.sort(key=lambda x: -x[1])
            keep_fe = [c for c, _ in stds[:MLP_FE_LIMIT]]
            fe_tr_noid = fe_tr_noid.select(keep_fe)
            fe_te_noid = fe_te_noid.select(keep_fe)
            del sample
            print(f"  ✅ FE: {len(keep_fe)} из {len(stds)}")
        else:
            print(f"  ✅ FE: {len(fe_tr_noid.columns)}")
        parts_train.append(fe_tr_noid)
        parts_test.append(fe_te_noid)
        del fe_tr, fe_te, fe_tr_noid, fe_te_noid; gc.collect()

    # OOF TE
    if os.path.exists(OOF_TE_TRAIN) and os.path.exists(OOF_TE_TEST):
        te_tr = pl.read_parquet(OOF_TE_TRAIN)
        te_te = pl.read_parquet(OOF_TE_TEST)
        te_tr = ensure_order(te_tr, train_id_ref, "oof_te_train")
        te_te = ensure_order(te_te, test_id_ref, "oof_te_test")
        te_tr_noid = di(te_tr)
        te_te_noid = di(te_te)
        if len(te_tr_noid.columns) > MLP_TE_LIMIT:
            stds = []
            sample = te_tr_noid[:50_000]
            for c in te_tr_noid.columns:
                stds.append((c, sample[c].to_numpy().astype(np.float32).std()))
            stds.sort(key=lambda x: -x[1])
            keep_te = [c for c, _ in stds[:MLP_TE_LIMIT]]
            te_tr_noid = te_tr_noid.select(keep_te)
            te_te_noid = te_te_noid.select(keep_te)
            del sample
            print(f"  ✅ OOF TE: {len(keep_te)} из {len(stds)}")
        else:
            print(f"  ✅ OOF TE: {len(te_tr_noid.columns)}")
        parts_train.append(te_tr_noid)
        parts_test.append(te_te_noid)
        del te_tr, te_te, te_tr_noid, te_te_noid; gc.collect()

    X_pl = pl.concat(parts_train, how="horizontal")
    X_test_pl = pl.concat(parts_test, how="horizontal")
    del parts_train, parts_test; gc.collect()

    # Категории → freq encoding
    cat_cols = [c for c in X_pl.columns if c.startswith("cat_")]
    for c in cat_cols:
        combined = pl.concat([X_pl.select(c), X_test_pl.select(c)])
        freq = (combined.group_by(c)
                .agg(pl.len().alias("cnt"))
                .with_columns(
                    (pl.col("cnt") / len(combined))
                    .cast(pl.Float32).alias(c + "_freq")))
        X_pl = (X_pl.join(freq.select([c, c + "_freq"]), on=c, how="left")
                .drop(c).rename({c + "_freq": c}))
        X_test_pl = (X_test_pl.join(freq.select([c, c + "_freq"]),
                                     on=c, how="left")
                     .drop(c).rename({c + "_freq": c}))
        del combined, freq

    cast_exprs = [pl.col(c).cast(pl.Float32).fill_null(0.0).alias(c)
                  for c in X_pl.columns]
    X_pl = X_pl.with_columns(cast_exprs)
    X_test_pl = X_test_pl.with_columns(cast_exprs)
    gc.collect()

    n_features_raw = len(X_pl.columns)
    print(f"  Признаков до лимита: {n_features_raw}")

    if n_features_raw > MAX_TOTAL_FEATURES:
        print(f"  ⚠️ Обрезаю {n_features_raw} → {MAX_TOTAL_FEATURES}...")
        sample = X_pl[:50_000]
        variances = []
        for c in X_pl.columns:
            v = sample[c].to_numpy().astype(np.float32)
            variances.append((c, np.nanvar(v)))
        del sample
        variances.sort(key=lambda x: -x[1])
        keep_features = [c for c, _ in variances[:MAX_TOTAL_FEATURES]]
        X_pl = X_pl.select(keep_features)
        X_test_pl = X_test_pl.select(keep_features)
        gc.collect()

    feature_names = list(X_pl.columns)
    n_features = len(feature_names)
    print(f"  Итого признаков: {n_features}")

    # Test → диск и освобождаем
    test_path = os.path.join(ARTIFACTS_PATH, "X_test_nn.npy")
    X_test_np = X_test_pl.to_numpy().astype(np.float32)
    np.save(test_path, X_test_np)
    n_test = len(X_test_np)
    del X_test_np, X_test_pl; gc.collect()

    # Train → numpy
    X_train = X_pl.to_numpy().astype(np.float32)
    del X_pl; gc.collect()

    # Targets
    target_df = pd.read_parquet(TRAIN_TARGET)
    target_cols = [c for c in target_df.columns if c.startswith("target")]
    target_df = (target_df.set_index("customer_id")
                 .loc[train_id_ref].reset_index())
    y_train = target_df[target_cols].values.astype(np.float32)
    del target_df; gc.collect()

    print(f"  X_train: {X_train.shape}, {X_train.nbytes / 1e9:.2f} ГБ")

    return (X_train, y_train, target_cols, n_features, n_test,
            train_id_ref, test_id_ref, test_path)


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("ШАГ 5: НЕЙРОСЕТЬ (MLP + PLR + ASL) + FE + OOF TE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} ГБ")
        torch.cuda.empty_cache()

    (X_train, y_train, target_cols, n_features, n_test,
     train_id_ref, test_id_ref, test_path) = prepare_data()

    n_train = len(X_train)
    n_targets = len(target_cols)

    y_strat = (y_train.sum(axis=1) > 0).astype(np.int32)
    skf = StratifiedKFold(n_splits=N_SPLITS_FINAL, shuffle=True,
                          random_state=SEED)
    folds = list(skf.split(X_train, y_strat))

    oof_preds = np.zeros((n_train, n_targets), dtype=np.float32)
    test_preds_sum = np.zeros((n_test, n_targets), dtype=np.float32)
    fold_aucs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        ckpt_path = os.path.join(NN_CHECKPOINT_DIR,
                                 f"fold_{fold_idx}.npz")
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

        # ── QT: fit на train fold, transform train+val (НЕ test) ──
        X_tr_raw = X_train[tr_idx]
        X_va_raw = X_train[va_idx]

        n_qt = X_tr_raw.shape[0]
        qt = QuantileTransformer(
            n_quantiles=min(1000, n_qt),
            output_distribution="normal",
            subsample=min(100_000, n_qt),
            random_state=SEED
        )
        qt.fit(X_tr_raw)

        X_tr = np.nan_to_num(qt.transform(X_tr_raw), nan=0.0).astype(np.float32)
        X_va = np.nan_to_num(qt.transform(X_va_raw), nan=0.0).astype(np.float32)
        del X_tr_raw, X_va_raw; gc.collect()

        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]

        # ── DataLoaders (train + val только, NO test) ──
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
            pin_memory=False, num_workers=0
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
            batch_size=BATCH_SIZE * 2, shuffle=False,
            pin_memory=False, num_workers=0
        )

        # ── Модель: create on CPU → set_bins → move to GPU ──
        model = TabMLPModel(n_features, n_targets)
        model.plr.set_bins(X_tr)
        del X_tr; gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = model.to(device)

        if device.type == "cuda":
            allocated_mb = torch.cuda.memory_allocated() / 1e6
            print(f"    VRAM after model.to(): {allocated_mb:.0f} MB")

        if fold_idx == 0:
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Параметров: {n_params:,}")

        criterion = AsymmetricLoss(ASL_GAMMA_NEG, ASL_GAMMA_POS, ASL_CLIP)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )
        scaler = torch.amp.GradScaler()

        best_auc = 0
        patience_counter = 0
        top_states = []

        for epoch in range(EPOCHS):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, scaler
            )
            val_preds_epoch = predict_model(model, val_loader, device)
            macro_auc = compute_macro_auc(y_va, val_preds_epoch, n_targets)
            scheduler.step(macro_auc)

            improved = ""
            if macro_auc > best_auc:
                best_auc = macro_auc
                patience_counter = 0
                improved = " *"
            else:
                patience_counter += 1

            state_copy = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            top_states.append((macro_auc, state_copy))
            top_states.sort(key=lambda x: x[0], reverse=True)
            if len(top_states) > TOP_K_SWA:
                top_states.pop()

            if (epoch + 1) % 5 == 0 or improved:
                lr = optimizer.param_groups[0]['lr']
                print(f"    Ep {epoch + 1:2d}/{EPOCHS}  "
                      f"loss={train_loss:.4f}  "
                      f"val_auc={macro_auc:.4f}  "
                      f"lr={lr:.1e}{improved}")

            if patience_counter >= PATIENCE:
                print(f"    Early stop at epoch {epoch + 1}")
                break

        # ── SWA ──
        avg_state = {}
        for key in top_states[0][1]:
            avg_state[key] = sum(
                s[key] for _, s in top_states
            ) / len(top_states)
        model.load_state_dict(avg_state)
        model.to(device)

        # ── Val predictions (SWA) ──
        val_preds = predict_model(model, val_loader, device)
        swa_auc = compute_macro_auc(y_va, val_preds, n_targets)
        print(f"  Fold {fold_idx + 1}: best={best_auc:.4f}, SWA={swa_auc:.4f}")

        oof_preds[va_idx] = val_preds

        # ── Освобождаем train/val ПЕРЕД test ──
        del (train_loader, val_loader, optimizer, scheduler, scaler,
             criterion, top_states, avg_state, X_va, y_tr, y_va,
             val_preds_epoch, val_preds)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ── Test: загружаем, transform, predict батчами ──
        print("    Predicting test...")
        X_test_mmap = np.load(test_path, mmap_mode='r')
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
                model, chunk_qt, device, batch_size=BATCH_SIZE * 2
            )
            del chunk_qt; gc.collect()

        del X_test_mmap
        test_preds_sum += test_preds_fold
        fold_aucs.append(swa_auc)

        # ── Чекпоинт ──
        np.savez(ckpt_path,
                 val_idx=va_idx,
                 val_preds=oof_preds[va_idx],
                 test_preds=test_preds_fold,
                 fold_auc=swa_auc)

        del model, test_preds_fold, qt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Итоги ──
    test_preds_avg = test_preds_sum / N_SPLITS_FINAL

    print("\n" + "=" * 60)
    nn_oof_auc = compute_macro_auc(y_train, oof_preds, n_targets)
    print(f"  NN OOF Macro AUC: {nn_oof_auc:.4f}")
    print(f"  Per-fold: {[f'{a:.4f}' for a in fold_aucs]}")

    np.savez(NN_OOF_PATH,
             oof_preds=oof_preds,
             test_preds=test_preds_avg,
             target_cols=target_cols,
             train_ids=train_id_ref,
             test_ids=test_id_ref)
    print(f"  Сохранено: {NN_OOF_PATH}")

    if os.path.exists(test_path):
        os.remove(test_path)

    print("\n  Готово!")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()