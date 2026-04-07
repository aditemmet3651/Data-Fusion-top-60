"""
Шаг 0: Отбор признаков.
  1. Глобальный отбор (как раньше): top-700, top-350
  2. НОВОЕ: Per-target отбор — top-150 extra features для каждого таргета
     Это даёт каждому таргету свой оптимальный набор фич.
"""

import polars as pl
import numpy as np
import lightgbm as lgb
import joblib
import gc
from collections import defaultdict
from config import *


def get_turbo_data():
    print("  Вычисляю turbo-индексы...")
    target_df = pl.read_parquet(TRAIN_TARGET)
    target_cols = [c for c in target_df.columns if c.startswith("target_")]
    n = len(target_df)

    counts = target_df.select(target_cols).sum()
    rare_targets = [c for c in target_cols if counts[c][0] < n * 0.01]
    print(f"  Редких таргетов (<1%): {len(rare_targets)} из {len(target_cols)}")

    target_np = target_df.select(rare_targets).to_numpy()
    is_rare = target_np.sum(axis=1) > 0

    rare_idx = np.where(is_rare)[0]
    common_idx = np.where(~is_rare)[0]

    np.random.seed(SEED)
    sampled_common = np.random.choice(
        common_idx,
        size=int(len(common_idx) * TURBO_SAMPLE_RATIO),
        replace=False
    )

    final_idx = np.sort(np.concatenate([rare_idx, sampled_common]))
    print(f"  Turbo: {len(final_idx):,} строк ({len(final_idx) / n:.1%})")

    target_turbo = target_df[final_idx.tolist()]

    del target_np
    gc.collect()
    return final_idx, target_turbo, target_cols


def variance_filter(train_path, turbo_idx, all_cols, min_variance=1e-6):
    print("\n  [1/3] Variance filter...")
    batch_size = 500
    good_cols = []

    for start in range(0, len(all_cols), batch_size):
        batch_cols = all_cols[start:start + batch_size]
        chunk = pl.read_parquet(train_path, columns=batch_cols)
        arr = chunk[turbo_idx.tolist()].to_numpy().astype(np.float32)

        for j, col_name in enumerate(batch_cols):
            col_data = arr[:, j]
            valid = col_data[~np.isnan(col_data)]
            if len(valid) < 100:
                continue
            var = np.var(valid)
            nonzero_rate = (valid != 0).mean()
            if var > min_variance and nonzero_rate > 0.001:
                good_cols.append(col_name)

        del chunk, arr
        gc.collect()

    print(f"    Прошло variance filter: {len(good_cols)} из {len(all_cols)}")
    return good_cols


def correlation_scores(train_path, turbo_idx, cols, target_turbo, target_cols):
    """Считаем abs(correlation) каждого extra-столбца с каждым таргетом.
    Возвращает два словаря:
      - global_scores: {col: sum_abs_corr} (сумма по всем таргетам)
      - per_target_scores: {target: {col: abs_corr}} (по каждому таргету)
    """
    print("\n  [2/3] Correlation scores...")
    batch_size = 500
    global_scores = {}
    per_target_scores = {t: {} for t in target_cols}

    y_all = target_turbo.select(target_cols).to_numpy().astype(np.float32)
    y_centered = y_all - y_all.mean(axis=0, keepdims=True)
    y_std = y_all.std(axis=0, keepdims=True)
    y_std[y_std < 1e-8] = 1.0
    y_norm = y_centered / y_std

    for start in range(0, len(cols), batch_size):
        batch_cols = cols[start:start + batch_size]
        batch_end = min(start + batch_size, len(cols))
        if (start // batch_size) % 3 == 0:
            print(f"    Батч {start // batch_size + 1}: "
                  f"столбцы {start}-{batch_end}...")

        chunk = pl.read_parquet(train_path, columns=batch_cols)
        arr = chunk[turbo_idx.tolist()].to_numpy().astype(np.float32)

        for j, col_name in enumerate(batch_cols):
            col_data = arr[:, j]
            col_data = np.nan_to_num(col_data, nan=0.0)

            col_centered = col_data - col_data.mean()
            col_std = col_data.std()
            if col_std < 1e-8:
                global_scores[col_name] = 0.0
                for t in target_cols:
                    per_target_scores[t][col_name] = 0.0
                continue

            col_norm = col_centered / col_std
            n = len(col_norm)
            abs_corrs = np.abs(col_norm @ y_norm / n)

            global_scores[col_name] = float(abs_corrs.sum())
            for ti, t in enumerate(target_cols):
                per_target_scores[t][col_name] = float(abs_corrs[ti])

        del chunk, arr
        gc.collect()

    del y_all, y_centered, y_norm
    gc.collect()
    return global_scores, per_target_scores


def lgbm_importance(train_path, turbo_idx, cols, target_turbo, target_cols,
                    n_rounds=150):
    """LightGBM gain по КАЖДОМУ таргету.
    Возвращает:
      - global_importance: {col: sum_gain}
      - per_target_importance: {target: {col: gain}}
    """
    print(f"\n  [3/3] LightGBM importance (все {len(target_cols)} таргетов)...")
    batch_size = 500
    global_importance = defaultdict(float)
    per_target_importance = {t: defaultdict(float) for t in target_cols}

    # Таргеты с достаточным числом позитивов
    target_counts = target_turbo.select(target_cols).sum()
    usable_targets = [c for c in target_cols
                      if target_counts[c][0] >= 50]
    print(f"    Таргетов с >= 50 позитивов: {len(usable_targets)}")

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_jobs": 4,
        "verbose": -1,
        "seed": SEED,
    }

    for batch_start in range(0, len(cols), batch_size):
        batch_cols = cols[batch_start:batch_start + batch_size]
        batch_end = min(batch_start + batch_size, len(cols))
        if (batch_start // batch_size) % 2 == 0:
            print(f"\n    Батч {batch_start // batch_size + 1}: "
                  f"столбцы {batch_start}-{batch_end}...")

        chunk = pl.read_parquet(train_path, columns=batch_cols)
        X_batch = chunk[turbo_idx.tolist()].to_numpy().astype(np.float32)
        X_batch = np.nan_to_num(X_batch, nan=0.0)

        del chunk
        gc.collect()

        for target_name in usable_targets:
            y = target_turbo[target_name].to_numpy().astype(np.float32)
            if len(np.unique(y)) < 2:
                continue

            pos_rate = y.mean()
            cur_params = params.copy()
            if pos_rate < 0.01:
                cur_params["scale_pos_weight"] = min(1.0 / pos_rate, 50)

            dtrain = lgb.Dataset(
                X_batch, label=y,
                feature_name=batch_cols,
                free_raw_data=True
            )
            model = lgb.train(cur_params, dtrain, num_boost_round=n_rounds)

            imp = dict(zip(
                model.feature_name(),
                model.feature_importance(importance_type="gain")
            ))

            for feat, score in imp.items():
                global_importance[feat] += score
                per_target_importance[target_name][feat] += score

            del dtrain, model

        del X_batch
        gc.collect()

    return dict(global_importance), {t: dict(v) for t, v in per_target_importance.items()}


def main():
    print("=" * 60)
    print("ШАГ 0: ОТБОР ПРИЗНАКОВ (с per-target)")
    print("=" * 60)

    turbo_idx, target_turbo, target_cols = get_turbo_data()

    extra_schema = pl.scan_parquet(TRAIN_EXTRA).collect_schema()
    all_extra_cols = [c for c in extra_schema.names() if c != "customer_id"]
    print(f"  Всего extra столбцов: {len(all_extra_cols)}")

    # Проверка порядка
    extra_ids = pl.read_parquet(TRAIN_EXTRA, columns=["customer_id"])
    target_all = pl.read_parquet(TRAIN_TARGET)
    target_ids_at_turbo = target_all["customer_id"][turbo_idx.tolist()]
    extra_ids_at_turbo = extra_ids["customer_id"][turbo_idx.tolist()]
    assert target_ids_at_turbo.to_list() == extra_ids_at_turbo.to_list()
    print("  ✅ Порядок customer_id совпадает")
    del extra_ids, target_all, target_ids_at_turbo, extra_ids_at_turbo
    gc.collect()

    # 1. Variance filter
    good_cols = variance_filter(TRAIN_EXTRA, turbo_idx, all_extra_cols)

    # 2. Correlation scores (global + per-target)
    corr_global, corr_per_target = correlation_scores(
        TRAIN_EXTRA, turbo_idx, good_cols, target_turbo, target_cols
    )

    # 3. LightGBM importance (global + per-target)
    lgbm_global, lgbm_per_target = lgbm_importance(
        TRAIN_EXTRA, turbo_idx, good_cols, target_turbo, target_cols
    )

    # === Глобальный отбор (как раньше) ===
    print("\n  Комбинирую глобальные скоры...")

    max_corr = max(corr_global.values()) if corr_global else 1.0
    max_lgbm = max(lgbm_global.values()) if lgbm_global else 1.0
    if max_corr == 0: max_corr = 1.0
    if max_lgbm == 0: max_lgbm = 1.0

    combined = {}
    for col in good_cols:
        c_score = corr_global.get(col, 0.0) / max_corr
        l_score = lgbm_global.get(col, 0.0) / max_lgbm
        combined[col] = 0.4 * c_score + 0.6 * l_score

    sorted_feats = sorted(combined.items(), key=lambda x: -x[1])
    top_700 = [f for f, _ in sorted_feats[:TOP_FEATURES_META]]
    top_300 = [f for f, _ in sorted_feats[:TOP_FEATURES_FINAL]]

    joblib.dump(top_700, SELECTED_FEATURES_700)
    joblib.dump(top_300, SELECTED_FEATURES_300)
    print(f"  Глобальный: {len(top_700)} / {len(top_300)} признаков сохранено")

    # === Per-target отбор (НОВОЕ) ===
    print("\n  Per-target feature selection...")
    per_target_feats = {}

    for t in target_cols:
        t_corr = corr_per_target.get(t, {})
        t_lgbm = lgbm_per_target.get(t, {})

        max_c = max(t_corr.values()) if t_corr and max(t_corr.values()) > 0 else 1.0
        max_l = max(t_lgbm.values()) if t_lgbm and max(t_lgbm.values()) > 0 else 1.0

        t_combined = {}
        for col in good_cols:
            cs = t_corr.get(col, 0.0) / max_c
            ls = t_lgbm.get(col, 0.0) / max_l
            t_combined[col] = 0.35 * cs + 0.65 * ls

        t_sorted = sorted(t_combined.items(), key=lambda x: -x[1])
        per_target_feats[t] = [f for f, _ in t_sorted[:PER_TARGET_TOP_K]]

    joblib.dump(per_target_feats, PER_TARGET_FEATURES)
    print(f"  Per-target: {len(per_target_feats)} таргетов × "
          f"{PER_TARGET_TOP_K} признаков")

    # Статистика overlap
    all_pt_feats = set()
    for feats in per_target_feats.values():
        all_pt_feats.update(feats)
    print(f"  Уникальных extra features across targets: {len(all_pt_feats)}")

    overlap_with_global = len(all_pt_feats & set(top_300))
    print(f"  Пересечение с global top-350: {overlap_with_global}")

    print("\n  ✅ Feature selection завершён!")


if __name__ == "__main__":
    main()