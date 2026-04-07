"""
Шаг 4: Финальное обучение v3 — RAM-safe.
Ключевое изменение: per-target extra фичи подклеиваются IN-PLACE,
base массивы уменьшены, test обрабатывается батчами.
"""

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib
import gc
import os
import warnings
from config import *

warnings.filterwarnings("ignore")

# Пути для test на диске (RAM-safe)
BASE_TEST_NPY = os.path.join(ARTIFACTS_PATH, "base_test_04.npy")
BASE_TRAIN_NPY = os.path.join(ARTIFACTS_PATH, "base_train_04.npy")


def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        try:
            ckpt = joblib.load(CHECKPOINT)
            print(f"  Найден чекпоинт: {len(ckpt.get('done_targets', []))} таргетов")
            return ckpt
        except Exception:
            print("  Чекпоинт повреждён, начинаю заново")
    return {"done_targets": [], "oof_aucs": {}}


def save_checkpoint(ckpt):
    joblib.dump(ckpt, CHECKPOINT)


def drop_id(df):
    return df.drop("customer_id") if "customer_id" in df.columns else df


def make_local_aggs(df_pl):
    cols = df_pl.columns
    return df_pl.select([
        pl.sum_horizontal(cols).alias("local_extra_sum"),
        pl.mean_horizontal(cols).alias("local_extra_mean"),
        pl.sum_horizontal([
            (pl.col(c) != 0).cast(pl.Int8) for c in cols
        ]).alias("local_extra_nonzero"),
    ])


def rank_average(pred1, pred2, w1, w2):
    if w1 <= 0:
        return rankdata(pred2) / len(pred2)
    if w2 <= 0:
        return rankdata(pred1) / len(pred1)
    r1 = rankdata(pred1) / len(pred1)
    r2 = rankdata(pred2) / len(pred2)
    return r1 * w1 + r2 * w2


def ensure_order(df, ref_ids, name):
    if "customer_id" in df.columns:
        if df["customer_id"].to_list() != ref_ids:
            print(f"  ⚠️ {name}: пересортировываю")
            ref_df = pl.DataFrame({"customer_id": ref_ids})
            return ref_df.join(df, on="customer_id", how="left")
    return df


def build_and_save_base(train_id_ref, test_id_ref):
    """Собирает base фичи и сохраняет на диск. Возвращает feature_names, cat_cols."""

    if os.path.exists(BASE_TRAIN_NPY) and os.path.exists(BASE_TEST_NPY):
        meta = joblib.load(os.path.join(ARTIFACTS_PATH, "base_meta_04.pkl"))
        print(f"  Base данные на диске: {meta['n_features']} features")
        return meta["feature_names"], meta["cat_cols"]

    print("  Собираю base данные...")
    df_train_main = pl.read_parquet(TRAIN_MAIN)
    df_test_main = pl.read_parquet(TEST_MAIN)

    # Глобальные extra — top-100
    global_extra = joblib.load(SELECTED_FEATURES_300)
    global_extra = [f for f in global_extra
                    if f != "customer_id" and not f.startswith("Column_")]
    actual_cols = set(pl.scan_parquet(TRAIN_EXTRA).collect_schema().names())
    global_extra = [f for f in global_extra if f in actual_cols][:100]

    df_train_extra = pl.read_parquet(TRAIN_EXTRA,
                                      columns=["customer_id"] + global_extra)
    df_test_extra = pl.read_parquet(TEST_EXTRA,
                                     columns=["customer_id"] + global_extra)

    meta_train = pl.read_parquet(META_TRAIN)
    meta_test = pl.read_parquet(META_TEST)
    aggs_train = pl.read_parquet(GLOBAL_AGGS_TRAIN)
    aggs_test = pl.read_parquet(GLOBAL_AGGS_TEST)

    # Выравниваем
    df_train_extra = ensure_order(df_train_extra, train_id_ref, "train_extra")
    meta_train = ensure_order(meta_train, train_id_ref, "meta_train")
    aggs_train = ensure_order(aggs_train, train_id_ref, "aggs_train")
    df_test_extra = ensure_order(df_test_extra, test_id_ref, "test_extra")
    meta_test = ensure_order(meta_test, test_id_ref, "meta_test")
    aggs_test = ensure_order(aggs_test, test_id_ref, "aggs_test")

    # Локальные агрегации
    local_agg_train = make_local_aggs(drop_id(df_train_extra))
    local_agg_test = make_local_aggs(drop_id(df_test_extra))

    # Склеиваем части
    parts_train = [
        drop_id(df_train_main), drop_id(df_train_extra), local_agg_train,
        drop_id(aggs_train), drop_id(meta_train),
    ]
    parts_test = [
        drop_id(df_test_main), drop_id(df_test_extra), local_agg_test,
        drop_id(aggs_test), drop_id(meta_test),
    ]

    # FE
    if os.path.exists(FE_TRAIN):
        fe_tr = ensure_order(pl.read_parquet(FE_TRAIN), train_id_ref, "fe_train")
        fe_te = ensure_order(pl.read_parquet(FE_TEST), test_id_ref, "fe_test")
        parts_train.append(drop_id(fe_tr))
        parts_test.append(drop_id(fe_te))
        del fe_tr, fe_te

    # OOF TE
    if os.path.exists(OOF_TE_TRAIN):
        te_tr = ensure_order(pl.read_parquet(OOF_TE_TRAIN), train_id_ref, "te_train")
        te_te = ensure_order(pl.read_parquet(OOF_TE_TEST), test_id_ref, "te_test")
        parts_train.append(drop_id(te_tr))
        parts_test.append(drop_id(te_te))
        del te_tr, te_te

    X_train_pl = pl.concat(parts_train, how="horizontal")
    X_test_pl = pl.concat(parts_test, how="horizontal")

    del (df_train_main, df_test_main, df_train_extra, df_test_extra,
         local_agg_train, local_agg_test, aggs_train, aggs_test,
         meta_train, meta_test, parts_train, parts_test)
    gc.collect()

    # Cast + TE для категорий
    cat_cols = [c for c in X_train_pl.columns if c.startswith("cat_")]
    non_cat_cols = [c for c in X_train_pl.columns if c not in cat_cols]

    # Target Encoding
    target_df = pd.read_parquet(TRAIN_TARGET)
    target_cols_all = [c for c in target_df.columns if c.startswith("target")]
    target_df = target_df.set_index("customer_id").loc[train_id_ref].reset_index()
    y_global = target_df[target_cols_all].mean(axis=1).values

    print("  Target Encoding...")
    for c in cat_cols:
        train_cat = X_train_pl[c].to_pandas()
        test_cat = X_test_pl[c].to_pandas()
        global_mean = y_global.mean()
        stats = pd.DataFrame({"cat": train_cat, "y": y_global})
        agg = stats.groupby("cat")["y"].agg(["mean", "count"])
        agg["smoothed"] = (
            (agg["count"] * agg["mean"] + 50 * global_mean) /
            (agg["count"] + 50)
        ).astype(np.float32)
        mapping = agg["smoothed"].to_dict()
        tr_enc = train_cat.map(mapping).fillna(global_mean).astype(np.float32)
        te_enc = test_cat.map(mapping).fillna(global_mean).astype(np.float32)
        X_train_pl = X_train_pl.with_columns(pl.Series(c, tr_enc.values).cast(pl.Float32))
        X_test_pl = X_test_pl.with_columns(pl.Series(c, te_enc.values).cast(pl.Float32))
        del train_cat, test_cat, tr_enc, te_enc

    del target_df, y_global; gc.collect()

    # Cast всё в float32
    cast_exprs = [pl.col(c).cast(pl.Float32).fill_null(0.0).alias(c)
                  for c in X_train_pl.columns]
    X_train_pl = X_train_pl.with_columns(cast_exprs)
    X_test_pl = X_test_pl.with_columns(cast_exprs)

    feature_names = list(X_train_pl.columns)
    n_features = len(feature_names)
    print(f"  Base features: {n_features}")

    # Сохраняем train
    print(f"  Сохраняю train на диск...")
    X_train_np = X_train_pl.to_numpy().astype(np.float32)
    np.save(BASE_TRAIN_NPY, X_train_np)
    print(f"  Train: {X_train_np.shape}, {X_train_np.nbytes / 1e9:.2f} ГБ")
    del X_train_np, X_train_pl; gc.collect()

    # Сохраняем test
    print(f"  Сохраняю test на диск...")
    X_test_np = X_test_pl.to_numpy().astype(np.float32)
    np.save(BASE_TEST_NPY, X_test_np)
    print(f"  Test: {X_test_np.shape}, {X_test_np.nbytes / 1e9:.2f} ГБ")
    del X_test_np, X_test_pl; gc.collect()

    # Метаданные
    joblib.dump({
        "feature_names": feature_names,
        "cat_cols": cat_cols,
        "n_features": n_features,
    }, os.path.join(ARTIFACTS_PATH, "base_meta_04.pkl"))

    return feature_names, cat_cols


def get_per_target_cols(target_name, base_col_set):
    """Возвращает список per-target extra колонок (исключая уже в base)."""
    if not os.path.exists(PER_TARGET_FEATURES):
        return []
    per_target = joblib.load(PER_TARGET_FEATURES)
    if target_name not in per_target:
        return []
    pt_cols = per_target[target_name]
    pt_cols = [c for c in pt_cols if c not in base_col_set and c != "customer_id"]
    actual_cols = set(pl.scan_parquet(TRAIN_EXTRA).collect_schema().names())
    pt_cols = [c for c in pt_cols if c in actual_cols]
    return pt_cols


def load_fold_data(tr_idx, va_idx, pt_cols, n_base):
    """Загружает train fold с per-target features. RAM-safe."""
    # Base из mmap
    base_mmap = np.load(BASE_TRAIN_NPY, mmap_mode='r')

    if not pt_cols:
        X_tr = base_mmap[tr_idx].astype(np.float32).copy()
        X_va = base_mmap[va_idx].astype(np.float32).copy()
        del base_mmap
        return X_tr, X_va

    # Base + per-target
    X_tr_base = base_mmap[tr_idx].astype(np.float32)
    X_va_base = base_mmap[va_idx].astype(np.float32)
    del base_mmap

    # Per-target extra
    pt_data = pl.read_parquet(TRAIN_EXTRA, columns=pt_cols)
    pt_np = pt_data.to_numpy().astype(np.float32)
    pt_np = np.nan_to_num(pt_np, nan=0.0)
    del pt_data; gc.collect()

    pt_tr = pt_np[tr_idx]
    pt_va = pt_np[va_idx]
    del pt_np; gc.collect()

    X_tr = np.concatenate([X_tr_base, pt_tr], axis=1)
    X_va = np.concatenate([X_va_base, pt_va], axis=1)
    del X_tr_base, X_va_base, pt_tr, pt_va; gc.collect()

    return X_tr, X_va


def predict_test_batched(model_lgbm, model_cat, pt_cols, n_base,
                          feature_names_full, n_test, batch_size=50_000):
    """Предсказание на test батчами — не загружаем весь test в RAM."""
    pred_lgbm = np.zeros(n_test, dtype=np.float64)
    pred_cat = np.zeros(n_test, dtype=np.float64)

    base_mmap = np.load(BASE_TEST_NPY, mmap_mode='r')

    if pt_cols:
        pt_data = pl.read_parquet(TEST_EXTRA, columns=pt_cols)
        pt_np = pt_data.to_numpy().astype(np.float32)
        pt_np = np.nan_to_num(pt_np, nan=0.0)
        del pt_data; gc.collect()
    else:
        pt_np = None

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        X_batch = base_mmap[start:end].astype(np.float32)

        if pt_np is not None:
            X_batch = np.concatenate([X_batch, pt_np[start:end]], axis=1)

        if model_lgbm is not None:
            pred_lgbm[start:end] = model_lgbm.predict(X_batch)
        if model_cat is not None:
            pred_cat[start:end] = model_cat.predict_proba(X_batch)[:, 1]

        del X_batch; gc.collect()

    del base_mmap
    if pt_np is not None:
        del pt_np; gc.collect()

    return pred_lgbm, pred_cat


def main():
    print("=" * 60)
    print("ШАГ 4: ФИНАЛЬНОЕ ОБУЧЕНИЕ v3 (RAM-safe)")
    print("=" * 60)

    # IDs
    train_ids_pl = pl.read_parquet(TRAIN_MAIN, columns=["customer_id"])
    test_ids_pl = pl.read_parquet(TEST_MAIN, columns=["customer_id"])
    train_id_ref = train_ids_pl["customer_id"].to_list()
    test_id_ref = test_ids_pl["customer_id"].to_list()
    del train_ids_pl, test_ids_pl; gc.collect()

    n_train = len(train_id_ref)
    n_test = len(test_id_ref)

    # Build base
    base_feature_names, cat_cols = build_and_save_base(train_id_ref, test_id_ref)
    n_base = len(base_feature_names)
    base_col_set = set(base_feature_names)

    # Targets
    target_df = pd.read_parquet(TRAIN_TARGET)
    target_cols = [c for c in target_df.columns if c.startswith("target")]
    target_df = target_df.set_index("customer_id").loc[train_id_ref].reset_index()

    test_ids = pd.Series(test_id_ref, name="customer_id")

    # Folds
    y_all = target_df[target_cols].values
    y_strat = (y_all.sum(axis=1) > 0).astype(np.int32)
    skf = StratifiedKFold(n_splits=N_SPLITS_FINAL, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(n_train), y_strat))

    # OOF arrays
    oof_lgbm_all = np.zeros((n_train, len(target_cols)), dtype=np.float32)
    oof_cat_all = np.zeros((n_train, len(target_cols)), dtype=np.float32)
    test_lgbm_all = np.zeros((n_test, len(target_cols)), dtype=np.float32)
    test_cat_all = np.zeros((n_test, len(target_cols)), dtype=np.float32)

    ckpt = load_checkpoint()
    done_targets = set(ckpt.get("done_targets", []))
    oof_aucs = ckpt.get("oof_aucs", {})
    eps = 1e-7

    if os.path.exists(OOF_LGBM_PATH) and done_targets:
        oof_lgbm_all = np.load(OOF_LGBM_PATH)
        oof_cat_all = np.load(OOF_CAT_PATH)
        test_lgbm_all = np.load(TEST_LGBM_PATH)
        test_cat_all = np.load(TEST_CAT_PATH)
        print(f"  Загружены OOF: {len(done_targets)} таргетов")

    for i, col in enumerate(target_cols):
        if col in done_targets:
            continue

        y = target_df[col].values.astype(np.float32)
        if len(np.unique(y)) < 2:
            done_targets.add(col)
            continue

        pos_rate = y.mean()
        is_rare = pos_rate < RARE_THRESHOLD
        is_hard = col in HARD_TARGETS

        # Per-target extra
        pt_cols = get_per_target_cols(col, base_col_set)

        if is_rare:
            lgbm_params = LGBM_RARE_PARAMS.copy()
            cat_params = CAT_RARE_PARAMS.copy()
            lgbm_rounds = LGBM_RARE_ROUNDS
            lgbm_es = LGBM_RARE_EARLY_STOPPING
            tag = "RARE"
        elif is_hard:
            lgbm_params = LGBM_HARD_PARAMS.copy()
            cat_params = CAT_FINAL_PARAMS.copy()
            lgbm_rounds = LGBM_HARD_ROUNDS
            lgbm_es = LGBM_HARD_EARLY_STOPPING
            tag = "HARD"
        else:
            lgbm_params = LGBM_FINAL_PARAMS.copy()
            cat_params = CAT_FINAL_PARAMS.copy()
            lgbm_rounds = LGBM_FINAL_ROUNDS
            lgbm_es = LGBM_EARLY_STOPPING
            tag = "normal"

        # Feature names для этого таргета
        feature_names_full = base_feature_names + pt_cols

        print(f"\n  [{i + 1:>2}/{len(target_cols)}] {col:>18s} "
              f"(pos={pos_rate:.4f}, {tag}, +{len(pt_cols)} pt)")

        oof_lgbm = np.zeros(n_train, dtype=np.float64)
        oof_cat = np.zeros(n_train, dtype=np.float64)
        test_lgbm_acc = np.zeros(n_test, dtype=np.float64)
        test_cat_acc = np.zeros(n_test, dtype=np.float64)

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            y_tr = y[tr_idx]
            y_va = y[va_idx]

            # Загружаем fold данные (RAM-safe)
            X_tr, X_va = load_fold_data(tr_idx, va_idx, pt_cols, n_base)

            # === LightGBM ===
            dtrain = lgb.Dataset(X_tr, label=y_tr,
                                 feature_name=feature_names_full,
                                 free_raw_data=True)
            dval = lgb.Dataset(X_va, label=y_va,
                               feature_name=feature_names_full,
                               free_raw_data=True, reference=dtrain)

            model_lgbm = lgb.train(
                lgbm_params, dtrain, num_boost_round=lgbm_rounds,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(lgbm_es, verbose=False),
                           lgb.log_evaluation(0)]
            )

            oof_lgbm[va_idx] = model_lgbm.predict(X_va)
            del dtrain, dval; gc.collect()

            # === CatBoost ===
            cat_params_fold = cat_params.copy()
            es_rounds = cat_params_fold.pop("early_stopping_rounds", 80)

            model_cat = CatBoostClassifier(**cat_params_fold)
            model_cat.fit(X_tr, y_tr,
                          eval_set=(X_va, y_va),
                          early_stopping_rounds=es_rounds,
                          verbose=False)

            oof_cat[va_idx] = model_cat.predict_proba(X_va)[:, 1]

            del X_tr, X_va; gc.collect()

            # Test predictions батчами
            p_lgbm, p_cat = predict_test_batched(
                model_lgbm, model_cat, pt_cols, n_base,
                feature_names_full, n_test, batch_size=50_000
            )
            test_lgbm_acc += p_lgbm / N_SPLITS_FINAL
            test_cat_acc += p_cat / N_SPLITS_FINAL

            del model_lgbm, model_cat; gc.collect()

        # Сохраняем OOF
        oof_lgbm_all[:, i] = oof_lgbm.astype(np.float32)
        oof_cat_all[:, i] = oof_cat.astype(np.float32)
        test_lgbm_all[:, i] = test_lgbm_acc.astype(np.float32)
        test_cat_all[:, i] = test_cat_acc.astype(np.float32)

        # OOF AUC
        try:
            auc_lgbm = roc_auc_score(y, oof_lgbm)
            auc_cat = roc_auc_score(y, oof_cat)

            best_w, best_auc = 0.5, 0.0
            lgbm_gap = auc_cat - auc_lgbm

            if lgbm_gap > 0.05:
                best_w = 0.0
                best_auc = auc_cat
            else:
                for w in np.arange(0.0, 1.05, 0.05):
                    blend_oof = rank_average(oof_lgbm, oof_cat, w, 1 - w)
                    auc_b = roc_auc_score(y, blend_oof)
                    if auc_b > best_auc:
                        best_auc = auc_b
                        best_w = w

            oof_aucs[col] = {
                "lgbm": auc_lgbm, "cat": auc_cat,
                "blend": best_auc, "w_lgbm": best_w
            }

            cat_only = " CAT-ONLY" if best_w == 0.0 else ""
            print(f"    LGBM: {auc_lgbm:.4f}, Cat: {auc_cat:.4f}, "
                  f"Blend: {best_auc:.4f} (w={best_w:.2f}){cat_only}")

        except Exception as e:
            print(f"    ⚠️ OOF AUC failed: {e}")
            oof_aucs[col] = {"lgbm": 0.5, "cat": 0.5,
                             "blend": 0.5, "w_lgbm": 0.5}

        done_targets.add(col)

        # Чекпоинт
        np.save(OOF_LGBM_PATH, oof_lgbm_all)
        np.save(OOF_CAT_PATH, oof_cat_all)
        np.save(TEST_LGBM_PATH, test_lgbm_all)
        np.save(TEST_CAT_PATH, test_cat_all)
        save_checkpoint({"done_targets": list(done_targets),
                         "oof_aucs": oof_aucs})

        del oof_lgbm, oof_cat, test_lgbm_acc, test_cat_acc
        gc.collect()

    # --- Summary ---
    print("\n" + "=" * 60)
    blend_aucs = [v["blend"] for v in oof_aucs.values()]
    if blend_aucs:
        print(f"  Mean blend AUC: {np.mean(blend_aucs):.4f}")
        print(f"  Min: {np.min(blend_aucs):.4f}, Max: {np.max(blend_aucs):.4f}")

    # --- Сабмит ---
    print("\n  Формирую сабмит...")
    predictions_dict = {}
    for j, col in enumerate(target_cols):
        pred_col = col.replace("target", "predict")
        info = oof_aucs.get(col, {"w_lgbm": 0.5})
        w = info.get("w_lgbm", 0.5)
        final_probs = rank_average(
            test_lgbm_all[:, j], test_cat_all[:, j], w, 1 - w
        )
        logits = np.log(
            np.clip(final_probs, eps, 1 - eps) /
            (1 - np.clip(final_probs, eps, 1 - eps))
        )
        predictions_dict[pred_col] = logits.astype(np.float64)

    final_preds = {"customer_id": test_ids}
    final_preds.update(predictions_dict)
    submission = pd.DataFrame(final_preds)
    submission["customer_id"] = submission["customer_id"].astype(np.int32)

    if os.path.exists(SAMPLE_SUBMIT):
        sample = pd.read_parquet(SAMPLE_SUBMIT)
        submission = sample[["customer_id"]].merge(
            submission, on="customer_id", how="left"
        )
        submission["customer_id"] = submission["customer_id"].astype(np.int32)

    submission = submission.fillna(-10.0)
    pred_cols_sub = [c for c in submission.columns if c.startswith("predict")]
    submission[pred_cols_sub] = submission[pred_cols_sub].astype(np.float64)
    submission.to_parquet(SUBMIT_MAIN, index=False)

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print(f"\n  Сабмит: {SUBMIT_MAIN} ({submission.shape})")
    print("  ✅ Готово!")


if __name__ == "__main__":
    main()