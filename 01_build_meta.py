"""
Шаг 1: Построение OOF мета-признаков.
  - LightGBM OOF для каждого таргета
  - Cross-target features (group mean/max/std)
  - Сохраняет meta_train.parquet, meta_test.parquet
"""

import numpy as np
import polars as pl
import lightgbm as lgb
import joblib
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from config import *

META_EXTRA_LIMIT = 400


def add_cross_target_features(meta_df, target_cols):
    meta_cols = [f"meta_{t}" for t in target_cols]
    new_cols = {}

    meta_vals = meta_df.select(meta_cols).to_numpy()
    new_cols["meta_all_mean"] = meta_vals.mean(axis=1).astype(np.float32)
    new_cols["meta_all_max"] = meta_vals.max(axis=1).astype(np.float32)
    new_cols["meta_all_std"] = meta_vals.std(axis=1).astype(np.float32)
    new_cols["meta_all_min"] = meta_vals.min(axis=1).astype(np.float32)

    # Quantiles
    new_cols["meta_all_q25"] = np.percentile(meta_vals, 25, axis=1).astype(np.float32)
    new_cols["meta_all_q75"] = np.percentile(meta_vals, 75, axis=1).astype(np.float32)

    for group_id, group_targets in TARGET_GROUPS.items():
        group_meta = [f"meta_{t}" for t in group_targets
                      if f"meta_{t}" in meta_df.columns]
        if len(group_meta) < 2:
            continue

        grp_vals = meta_df.select(group_meta).to_numpy()
        new_cols[f"meta_grp{group_id}_mean"] = grp_vals.mean(axis=1).astype(np.float32)
        new_cols[f"meta_grp{group_id}_max"] = grp_vals.max(axis=1).astype(np.float32)
        new_cols[f"meta_grp{group_id}_std"] = grp_vals.std(axis=1).astype(np.float32)

    # Rank features
    ranks = np.argsort(np.argsort(-meta_vals, axis=1), axis=1).astype(np.float32)
    for idx, mc in enumerate(meta_cols):
        t_name = mc.replace("meta_", "")
        new_cols[f"meta_rank_{t_name}"] = ranks[:, idx]

    return new_cols


def main():
    print("=" * 60)
    print("ШАГ 1: ПОСТРОЕНИЕ OOF МЕТА-ПРИЗНАКОВ")
    print("=" * 60)

    top_feats = joblib.load(SELECTED_FEATURES_700)
    top_feats = [f for f in top_feats
                 if f != "customer_id" and not f.startswith("Column_")]
    actual_cols = set(pl.scan_parquet(TRAIN_EXTRA).collect_schema().names())
    top_feats = [f for f in top_feats if f in actual_cols]
    top_feats = top_feats[:META_EXTRA_LIMIT]
    print(f"  Extra признаков для мета: {len(top_feats)}")

    # --- Train ---
    print("\n  Загружаю train данные...")
    tr_main = pl.read_parquet(TRAIN_MAIN)
    train_ids = tr_main["customer_id"]
    tr_extra = pl.read_parquet(TRAIN_EXTRA, columns=["customer_id"] + top_feats)

    if tr_main["customer_id"].to_list() == tr_extra["customer_id"].to_list():
        print("  ✅ Порядок совпадает")
        X_train_pl = pl.concat([tr_main.drop("customer_id"),
                                 tr_extra.drop("customer_id")], how="horizontal")
    else:
        X_train_pl = tr_main.join(tr_extra, on="customer_id", how="left")
        X_train_pl = X_train_pl.drop("customer_id")
    del tr_main, tr_extra; gc.collect()

    cat_cols_names = set(c for c in X_train_pl.columns if c.startswith("cat_"))
    cast_exprs = []
    for c in X_train_pl.columns:
        if c in cat_cols_names:
            cast_exprs.append(pl.col(c).cast(pl.Int32).fill_null(-1).alias(c))
        else:
            cast_exprs.append(pl.col(c).cast(pl.Float32).fill_null(0.0).alias(c))
    X_train_pl = X_train_pl.with_columns(cast_exprs)
    gc.collect()

    feature_names = list(X_train_pl.columns)
    cat_feature_names = [c for c in feature_names if c in cat_cols_names]

    n_train = len(X_train_pl)
    n_features = len(feature_names)
    estimated_gb = n_train * n_features * 4 / 1e9
    print(f"  Train: {X_train_pl.shape}, ~{estimated_gb:.2f} ГБ")

    if estimated_gb > 3.5:
        extra_in = [c for c in feature_names if c.startswith("num_feature_")]
        keep_extra = extra_in[:300]
        main_features = [c for c in feature_names if c not in extra_in]
        feature_names = main_features + keep_extra
        X_train_pl = X_train_pl.select(feature_names)
        cat_feature_names = [c for c in feature_names if c in cat_cols_names]
        print(f"  Урезано до {len(feature_names)} признаков")

    # --- Test ---
    print("  Загружаю test данные...")
    te_main = pl.read_parquet(TEST_MAIN)
    test_ids = te_main["customer_id"]
    te_extra = pl.read_parquet(TEST_EXTRA, columns=["customer_id"] + top_feats)

    if te_main["customer_id"].to_list() == te_extra["customer_id"].to_list():
        X_test_pl = pl.concat([te_main.drop("customer_id"),
                                te_extra.drop("customer_id")], how="horizontal")
    else:
        X_test_pl = te_main.join(te_extra, on="customer_id", how="left")
        X_test_pl = X_test_pl.drop("customer_id")
    del te_main, te_extra; gc.collect()

    cast_exprs_test = []
    for c in feature_names:
        if c in cat_cols_names:
            cast_exprs_test.append(pl.col(c).cast(pl.Int32).fill_null(-1).alias(c))
        else:
            cast_exprs_test.append(pl.col(c).cast(pl.Float32).fill_null(0.0).alias(c))
    X_test_pl = X_test_pl.select(feature_names).with_columns(cast_exprs_test)
    gc.collect()
    print(f"  Test: {X_test_pl.shape}")

    # --- Таргеты ---
    target_df = pl.read_parquet(TRAIN_TARGET)
    target_cols = [c for c in target_df.columns if c.startswith("target")]
    target_df = (
        pl.DataFrame({"customer_id": train_ids})
        .join(target_df, on="customer_id", how="left")
    )

    # --- Numpy ---
    print("  Конвертирую в numpy...")
    X_train_np = X_train_pl.to_numpy()
    del X_train_pl; gc.collect()
    X_test_np = X_test_pl.to_numpy()
    del X_test_pl; gc.collect()

    print(f"  RAM: train {X_train_np.nbytes / 1e9:.2f} ГБ, "
          f"test {X_test_np.nbytes / 1e9:.2f} ГБ")

    # --- OOF ---
    n_test = len(X_test_np)
    meta_train = np.zeros((n_train, len(target_cols)), dtype=np.float32)
    meta_test = np.zeros((n_test, len(target_cols)), dtype=np.float32)

    target_np = target_df.select(target_cols).to_numpy()
    y_strat = (target_np.sum(axis=1) > 0).astype(np.int32)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_train_np, y_strat))

    aucs_per_target = {}

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, 1):
        print(f"\n  === FOLD {fold_idx}/{N_SPLITS} ===")
        X_tr_f = X_train_np[tr_idx]
        X_va_f = X_train_np[va_idx]

        for i, t in enumerate(target_cols):
            y_tr = target_np[tr_idx, i]
            y_va = target_np[va_idx, i]
            if len(np.unique(y_tr)) < 2:
                continue

            pos_rate = y_tr.mean()
            cur_params = LGBM_META_PARAMS.copy()
            cur_rounds = LGBM_META_ROUNDS

            if pos_rate < 0.005:
                cur_params["min_data_in_leaf"] = 30
                cur_params["num_leaves"] = 31
                cur_params["is_unbalance"] = True

            dtrain = lgb.Dataset(X_tr_f, label=y_tr,
                                 feature_name=feature_names,
                                 categorical_feature=cat_feature_names,
                                 free_raw_data=True)
            dval = lgb.Dataset(X_va_f, label=y_va,
                               feature_name=feature_names,
                               categorical_feature=cat_feature_names,
                               free_raw_data=True, reference=dtrain)

            model = lgb.train(
                cur_params, dtrain, num_boost_round=cur_rounds,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(60, verbose=False),
                           lgb.log_evaluation(0)]
            )

            meta_train[va_idx, i] = model.predict(X_va_f)
            meta_test[:, i] += model.predict(X_test_np) / N_SPLITS
            del dtrain, dval, model

        del X_tr_f, X_va_f; gc.collect()
        print(f"  Fold {fold_idx} завершён")

    # OOF AUC
    print("\n  OOF AUC:")
    for i, t in enumerate(target_cols):
        y_true = target_np[:, i]
        if len(np.unique(y_true)) < 2:
            continue
        try:
            auc = roc_auc_score(y_true, meta_train[:, i])
            aucs_per_target[t] = auc
        except Exception:
            pass

    mean_auc = np.mean(list(aucs_per_target.values()))
    print(f"  Средний OOF AUC: {mean_auc:.4f}")

    del X_train_np, X_test_np, target_np; gc.collect()

    # --- Cross-target ---
    print("\n  Cross-target features...")
    meta_cols = [f"meta_{t}" for t in target_cols]
    df_meta_train = pl.DataFrame(
        {col: meta_train[:, i] for i, col in enumerate(meta_cols)}
    )
    df_meta_test = pl.DataFrame(
        {col: meta_test[:, i] for i, col in enumerate(meta_cols)}
    )

    cross_train = add_cross_target_features(df_meta_train, target_cols)
    cross_test = add_cross_target_features(df_meta_test, target_cols)

    for k, v in cross_train.items():
        df_meta_train = df_meta_train.with_columns(pl.Series(k, v))
    for k, v in cross_test.items():
        df_meta_test = df_meta_test.with_columns(pl.Series(k, v))

    df_meta_train = df_meta_train.with_columns(pl.Series("customer_id", train_ids))
    df_meta_train.write_parquet(META_TRAIN)
    df_meta_test = df_meta_test.with_columns(pl.Series("customer_id", test_ids))
    df_meta_test.write_parquet(META_TEST)

    print(f"  Сохранено: {META_TRAIN} ({df_meta_train.shape})")
    print(f"  Сохранено: {META_TEST} ({df_meta_test.shape})")


if __name__ == "__main__":
    main()