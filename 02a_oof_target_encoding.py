"""
Шаг 2a: OOF Target Encoding.
Для каждого таргета создаём target-encoded версии категориальных признаков,
вычисленные через OOF чтобы избежать лика.
  - Smoothing = 50
  - 5-fold OOF для train
  - Full train statistics для test
"""

import numpy as np
import polars as pl
import pandas as pd
import gc
from sklearn.model_selection import StratifiedKFold
from config import *


def oof_target_encode_one(train_cat, test_cat, y, folds, smoothing=50):
    """OOF target encoding для одного категориального столбца и одного таргета."""
    global_mean = y.mean()
    n = len(train_cat)
    oof_encoded = np.full(n, global_mean, dtype=np.float32)

    for tr_idx, va_idx in folds:
        # Считаем stats только по train fold
        df_tr = pd.DataFrame({"cat": train_cat[tr_idx], "y": y[tr_idx]})
        agg = df_tr.groupby("cat")["y"].agg(["mean", "count"])
        agg["smoothed"] = (
            (agg["count"] * agg["mean"] + smoothing * global_mean) /
            (agg["count"] + smoothing)
        ).astype(np.float32)
        mapping = agg["smoothed"].to_dict()

        va_cats = train_cat[va_idx]
        oof_encoded[va_idx] = np.array(
            [mapping.get(c, global_mean) for c in va_cats],
            dtype=np.float32
        )

    # Full train stats для test
    df_full = pd.DataFrame({"cat": train_cat, "y": y})
    agg = df_full.groupby("cat")["y"].agg(["mean", "count"])
    agg["smoothed"] = (
        (agg["count"] * agg["mean"] + smoothing * global_mean) /
        (agg["count"] + smoothing)
    ).astype(np.float32)
    mapping_full = agg["smoothed"].to_dict()

    test_encoded = np.array(
        [mapping_full.get(c, global_mean) for c in test_cat],
        dtype=np.float32
    )

    return oof_encoded, test_encoded


def main():
    print("=" * 60)
    print("ШАГ 2a: OOF TARGET ENCODING")
    print("=" * 60)

    # Загружаем категориальные колонки
    train_main = pl.read_parquet(TRAIN_MAIN)
    test_main = pl.read_parquet(TEST_MAIN)

    train_ids = train_main["customer_id"].to_list()
    test_ids = test_main["customer_id"].to_list()

    cat_cols = [c for c in train_main.columns if c.startswith("cat_")]
    print(f"  Категориальных признаков: {len(cat_cols)}")

    # Конвертируем в pandas для скорости
    train_cats = train_main.select(cat_cols).to_pandas()
    test_cats = test_main.select(cat_cols).to_pandas()
    del train_main, test_main; gc.collect()

    # Таргеты
    target_df = pd.read_parquet(TRAIN_TARGET)
    target_df = target_df.set_index("customer_id").loc[train_ids].reset_index()
    target_cols = [c for c in target_df.columns if c.startswith("target")]

    # Отбираем таргеты для encoding — не слишком редкие
    selected_targets = []
    for t in target_cols:
        pos_rate = target_df[t].mean()
        if pos_rate >= 0.002:  # хотя бы 1500 позитивов
            selected_targets.append(t)
    print(f"  Таргетов для encoding: {len(selected_targets)} из {len(target_cols)}")

    # Отбираем категории — не слишком много уникальных
    selected_cats = []
    for c in cat_cols:
        n_unique = train_cats[c].nunique()
        if 2 <= n_unique <= 500:
            selected_cats.append(c)
    print(f"  Категорий для encoding: {len(selected_cats)} из {len(cat_cols)}")

    # Folds
    y_all = target_df[target_cols].values
    y_strat = (y_all.sum(axis=1) > 0).astype(np.int32)
    skf = StratifiedKFold(n_splits=N_SPLITS_FINAL, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(train_cats)), y_strat))

    # OOF TE
    train_te_dict = {}
    test_te_dict = {}
    n_created = 0

    for ti, target in enumerate(selected_targets):
        y = target_df[target].values.astype(np.float32)

        for ci, cat_col in enumerate(selected_cats):
            col_name = f"te_{cat_col}_{target}"

            train_cat_arr = train_cats[cat_col].fillna("__NA__").values
            test_cat_arr = test_cats[cat_col].fillna("__NA__").values

            oof_enc, test_enc = oof_target_encode_one(
                train_cat_arr, test_cat_arr, y, folds, smoothing=50
            )

            train_te_dict[col_name] = oof_enc
            test_te_dict[col_name] = test_enc
            n_created += 1

        if (ti + 1) % 10 == 0:
            print(f"    Обработано {ti + 1}/{len(selected_targets)} таргетов, "
                  f"создано {n_created} признаков")

    print(f"\n  Всего OOF TE признаков: {n_created}")

    # Ограничение количества — оставляем только самые вариативные
    MAX_TE_FEATURES = 300
    if n_created > MAX_TE_FEATURES:
        print(f"  Отбираем top-{MAX_TE_FEATURES} по std...")
        stds = {k: np.std(v) for k, v in train_te_dict.items()}
        top_keys = sorted(stds, key=stds.get, reverse=True)[:MAX_TE_FEATURES]
        train_te_dict = {k: train_te_dict[k] for k in top_keys}
        test_te_dict = {k: test_te_dict[k] for k in top_keys}
        print(f"  Оставлено {len(train_te_dict)} признаков")

    # Сохраняем
    for k, v in train_te_dict.items():
        train_te_dict[k] = pl.Series(k, v)
    for k, v in test_te_dict.items():
        test_te_dict[k] = pl.Series(k, v)

    df_train_te = pl.DataFrame(train_te_dict)
    df_train_te = df_train_te.with_columns(
        pl.Series("customer_id", train_ids)
    )
    df_train_te.write_parquet(OOF_TE_TRAIN)

    df_test_te = pl.DataFrame(test_te_dict)
    df_test_te = df_test_te.with_columns(
        pl.Series("customer_id", test_ids)
    )
    df_test_te.write_parquet(OOF_TE_TEST)

    print(f"  Сохранено: {OOF_TE_TRAIN} ({df_train_te.shape})")
    print(f"  Сохранено: {OOF_TE_TEST} ({df_test_te.shape})")
    print("  ✅ OOF Target Encoding завершён!")


if __name__ == "__main__":
    main()