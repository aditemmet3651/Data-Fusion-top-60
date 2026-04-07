"""
Шаг 2.5: Feature Engineering (оптимизированный)
Изменения:
  - Сокращены попарные взаимодействия (top-8 вместо top-15)
  - Укрупнены NaN-паттерны (группы по 500 вместо 200)
  - Убраны subgroup aggs min/max (низкий impact, много фичей)
  - Добавлены log/sqrt трансформации для top числовых
"""

import numpy as np
import polars as pl
import joblib
import gc
import os
from sklearn.decomposition import IncrementalPCA
from config import *

FE_TRAIN_PATH = os.path.join(ARTIFACTS_PATH, "fe_train.parquet")
FE_TEST_PATH = os.path.join(ARTIFACTS_PATH, "fe_test.parquet")


def frequency_encoding(train_pl, test_pl, cat_cols):
    """Частотное кодирование категорий."""
    print("  [1/7] Частотное кодирование...")
    new_train_cols = {}
    new_test_cols = {}

    for c in cat_cols:
        all_vals = pl.concat([
            train_pl.select(pl.col(c).cast(pl.Utf8)),
            test_pl.select(pl.col(c).cast(pl.Utf8)),
        ])
        freq = all_vals.group_by(c).agg(pl.len().alias("cnt"))
        total = len(all_vals)

        new_name = f"freq_{c}"

        train_freq = (
            train_pl.select(pl.col(c).cast(pl.Utf8))
            .join(freq, on=c, how="left")
            .select((pl.col("cnt") / total).cast(pl.Float32).alias(new_name))
        )
        test_freq = (
            test_pl.select(pl.col(c).cast(pl.Utf8))
            .join(freq, on=c, how="left")
            .select((pl.col("cnt") / total).cast(pl.Float32).alias(new_name))
        )

        new_train_cols[new_name] = train_freq[new_name]
        new_test_cols[new_name] = test_freq[new_name]

    print(f"    Создано {len(new_train_cols)} частотных признаков")
    return new_train_cols, new_test_cols


def group_stats(train_pl, test_pl, cat_cols, num_cols):
    """Статистики числовых признаков внутри категориальных групп."""
    print("  [2/7] Групповые статистики...")
    top_cats = cat_cols[:5]
    top_nums = num_cols[:10]

    new_train_cols = {}
    new_test_cols = {}

    for cat in top_cats:
        stats = (
            train_pl
            .select([pl.col(cat).cast(pl.Utf8).alias(cat)] +
                    [pl.col(n) for n in top_nums])
            .group_by(cat)
            .agg(
                [pl.col(n).mean().alias(f"{n}_mean") for n in top_nums] +
                [pl.col(n).std().alias(f"{n}_std") for n in top_nums]
            )
        )

        for n in top_nums:
            zscore_name = f"zscore_{cat}_{n}"
            ratio_name = f"ratio_{cat}_{n}"

            for src_pl, dst_dict in [(train_pl, new_train_cols), (test_pl, new_test_cols)]:
                joined = (
                    src_pl
                    .select([pl.col(cat).cast(pl.Utf8).alias(cat), pl.col(n)])
                    .join(stats, on=cat, how="left")
                )
                result = joined.select([
                    (
                        (pl.col(n) - pl.col(f"{n}_mean").fill_null(0.0)) /
                        pl.when(pl.col(f"{n}_std").fill_null(1.0).abs() < 0.001)
                        .then(1.0)
                        .otherwise(pl.col(f"{n}_std").fill_null(1.0))
                    ).cast(pl.Float32).fill_null(0.0).fill_nan(0.0).alias(zscore_name),
                    (
                        pl.col(n) /
                        pl.when(pl.col(f"{n}_mean").fill_null(1.0).abs() < 0.001)
                        .then(1.0)
                        .otherwise(pl.col(f"{n}_mean").fill_null(1.0))
                    ).cast(pl.Float32).fill_null(1.0).fill_nan(1.0).clip(-100, 100).alias(ratio_name),
                ])
                dst_dict[zscore_name] = result[zscore_name]
                dst_dict[ratio_name] = result[ratio_name]

    print(f"    Создано {len(new_train_cols)} групповых признаков")
    return new_train_cols, new_test_cols


def pairwise_interactions(train_pl, test_pl, num_cols):
    """Попарные взаимодействия top-8 числовых main признаков (было 15)."""
    print("  [3/7] Попарные взаимодействия (top-8)...")
    top = num_cols[:8]  # Сокращено с 15 до 8
    new_train_cols = {}
    new_test_cols = {}

    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            mult_name = f"mult_{a}_{b}"
            div_name = f"div_{a}_{b}"

            for src_pl, dst_dict in [(train_pl, new_train_cols), (test_pl, new_test_cols)]:
                result = src_pl.select([
                    (pl.col(a) * pl.col(b))
                    .cast(pl.Float32).fill_null(0.0).fill_nan(0.0)
                    .alias(mult_name),
                    (
                        pl.col(a) /
                        pl.when(pl.col(b).abs() < 0.001)
                        .then(pl.when(pl.col(b) >= 0).then(0.001).otherwise(-0.001))
                        .otherwise(pl.col(b))
                    ).cast(pl.Float32).fill_null(0.0).fill_nan(0.0)
                    .clip(-1000, 1000).alias(div_name),
                ])
                dst_dict[mult_name] = result[mult_name]
                dst_dict[div_name] = result[div_name]

    print(f"    Создано {len(new_train_cols)} признаков взаимодействий")
    return new_train_cols, new_test_cols


def numeric_transforms(train_pl, test_pl, num_cols):
    """Log и sqrt трансформации для top числовых признаков."""
    print("  [4/7] Числовые трансформации (log, sqrt)...")
    top = num_cols[:20]
    new_train_cols = {}
    new_test_cols = {}

    for c in top:
        log_name = f"log1p_{c}"
        sqrt_name = f"sqrt_{c}"

        for src_pl, dst_dict in [(train_pl, new_train_cols), (test_pl, new_test_cols)]:
            result = src_pl.select([
                pl.col(c).abs().log1p()
                .cast(pl.Float32).fill_null(0.0).fill_nan(0.0)
                .alias(log_name),
                pl.col(c).abs().sqrt()
                .cast(pl.Float32).fill_null(0.0).fill_nan(0.0)
                .alias(sqrt_name),
            ])
            dst_dict[log_name] = result[log_name]
            dst_dict[sqrt_name] = result[sqrt_name]

    print(f"    Создано {len(new_train_cols)} трансформированных признаков")
    return new_train_cols, new_test_cols


def incremental_pca_features(train_path, test_path, n_components=20):
    """IncrementalPCA по top extra features."""
    print(f"  [5/7] IncrementalPCA ({n_components} компонент)...")

    if os.path.exists(SELECTED_FEATURES_700):
        pca_cols = joblib.load(SELECTED_FEATURES_700)
        pca_cols = [c for c in pca_cols if c != "customer_id"]
    else:
        schema = pl.scan_parquet(train_path).collect_schema()
        pca_cols = [c for c in schema.names() if c != "customer_id"]

    pca_cols = pca_cols[:PCA_TOP_COLS]
    n_cols = len(pca_cols)
    print(f"    Используем {n_cols} столбцов для PCA")

    if n_cols < n_components:
        return {}, {}

    ipca = IncrementalPCA(n_components=n_components)
    col_chunk = 300
    row_batch = 100_000

    n_train = pl.scan_parquet(train_path).select(pl.len()).collect().item()
    n_test = pl.scan_parquet(test_path).select(pl.len()).collect().item()

    # Fit
    for row_start in range(0, n_train, row_batch):
        row_end = min(row_start + row_batch, n_train)
        row_chunks = []
        for col_start in range(0, n_cols, col_chunk):
            batch_cols = pca_cols[col_start:col_start + col_chunk]
            chunk = pl.read_parquet(train_path, columns=batch_cols)
            row_chunks.append(chunk[row_start:row_end].to_numpy().astype(np.float32))
            del chunk; gc.collect()
        row_data = np.nan_to_num(np.concatenate(row_chunks, axis=1), nan=0.0)
        ipca.partial_fit(row_data)
        del row_chunks, row_data; gc.collect()

    print(f"    Explained variance: {ipca.explained_variance_ratio_.sum():.3f}")

    # Transform
    def transform_dataset(path, n_rows):
        result = np.zeros((n_rows, n_components), dtype=np.float32)
        for row_start in range(0, n_rows, row_batch):
            row_end = min(row_start + row_batch, n_rows)
            row_chunks = []
            for col_start in range(0, n_cols, col_chunk):
                batch_cols = pca_cols[col_start:col_start + col_chunk]
                chunk = pl.read_parquet(path, columns=batch_cols)
                row_chunks.append(chunk[row_start:row_end].to_numpy().astype(np.float32))
                del chunk; gc.collect()
            row_data = np.nan_to_num(np.concatenate(row_chunks, axis=1), nan=0.0)
            result[row_start:row_end] = ipca.transform(row_data).astype(np.float32)
            del row_chunks, row_data; gc.collect()
        return result

    print("    Transform train...")
    pca_train = transform_dataset(train_path, n_train)
    print("    Transform test...")
    pca_test = transform_dataset(test_path, n_test)

    pca_names = [f"pca_extra_{i}" for i in range(n_components)]
    train_dict = {name: pca_train[:, i] for i, name in enumerate(pca_names)}
    test_dict = {name: pca_test[:, i] for i, name in enumerate(pca_names)}

    del pca_train, pca_test; gc.collect()
    return train_dict, test_dict


def nan_pattern_features(train_path, test_path):
    """NaN-паттерны с укрупнёнными группами (500 вместо 200)."""
    print("  [6/7] NaN-паттерны (укрупнённые)...")

    schema = pl.scan_parquet(train_path).collect_schema()
    all_cols = [c for c in schema.names() if c != "customer_id"]
    group_size = 500  # Увеличено с 200

    train_dict = {}
    test_dict = {}

    for start in range(0, len(all_cols), group_size):
        batch_cols = all_cols[start:start + group_size]
        g = start // group_size

        for path, d in [(train_path, train_dict), (test_path, test_dict)]:
            chunk = pl.read_parquet(path, columns=batch_cols)
            null_count = chunk.select(
                pl.sum_horizontal([
                    pl.col(c).is_null().cast(pl.Int8) for c in batch_cols
                ]).alias("nc")
            )["nc"].to_numpy().astype(np.float32)

            arr = chunk.to_numpy().astype(np.float32)
            nonzero_count = ((arr != 0) & ~np.isnan(arr)).sum(axis=1).astype(np.float32)

            d[f"nullcnt_{g}"] = null_count
            d[f"nonzero_{g}"] = nonzero_count

            del chunk, arr; gc.collect()

    print(f"    Создано {len(train_dict)} NaN-паттерн признаков")
    return train_dict, test_dict


def extra_subgroup_aggs(train_path, test_path):
    """Агрегации по подгруппам — только sum, mean, std (без min/max)."""
    print("  [7/7] Агрегации по подгруппам extra...")

    schema = pl.scan_parquet(train_path).collect_schema()
    all_cols = [c for c in schema.names() if c != "customer_id"]
    group_size = 500  # Увеличено

    train_dict = {}
    test_dict = {}

    for start in range(0, len(all_cols), group_size):
        batch_cols = all_cols[start:start + group_size]
        g = start // group_size

        for path, d in [(train_path, train_dict), (test_path, test_dict)]:
            chunk = pl.read_parquet(path, columns=batch_cols)
            arr = np.nan_to_num(chunk.to_numpy().astype(np.float32), nan=0.0)

            d[f"subgroup_sum_{g}"] = arr.sum(axis=1).astype(np.float32)
            d[f"subgroup_mean_{g}"] = arr.mean(axis=1).astype(np.float32)
            d[f"subgroup_std_{g}"] = arr.std(axis=1).astype(np.float32)

            del chunk, arr; gc.collect()

    print(f"    Создано {len(train_dict)} подгрупповых признаков")
    return train_dict, test_dict


def validate_dict(d, name):
    bad_keys = []
    for k, v in d.items():
        if not isinstance(v, (pl.Series, np.ndarray, list)):
            bad_keys.append((k, type(v).__name__))
    if bad_keys:
        raise TypeError(f"{name} содержит объекты типа {bad_keys[0][1]}")
    print(f"  ✅ {name}: все {len(d)} значений валидны")


def main():
    print("=" * 60)
    print("ШАГ 2.5: FEATURE ENGINEERING (оптимизированный)")
    print("=" * 60)

    print("\n  Загружаю main features...")
    train_pl = pl.read_parquet(TRAIN_MAIN)
    test_pl = pl.read_parquet(TEST_MAIN)

    cat_cols = [c for c in train_pl.columns if c.startswith("cat_")]
    num_cols = [c for c in train_pl.columns
                if c not in cat_cols and c != "customer_id"]

    print(f"  Категориальных: {len(cat_cols)}, Числовых: {len(num_cols)}")

    all_train = {}
    all_test = {}

    # 1. Частотное кодирование
    tr, te = frequency_encoding(train_pl, test_pl, cat_cols)
    all_train.update(tr); all_test.update(te)
    gc.collect()

    # 2. Групповые статистики
    tr, te = group_stats(train_pl, test_pl, cat_cols, num_cols)
    all_train.update(tr); all_test.update(te)
    gc.collect()

    # 3. Попарные взаимодействия (сокращённые)
    tr, te = pairwise_interactions(train_pl, test_pl, num_cols)
    all_train.update(tr); all_test.update(te)

    # 4. Числовые трансформации (НОВОЕ)
    tr, te = numeric_transforms(train_pl, test_pl, num_cols)
    all_train.update(tr); all_test.update(te)

    train_ids = train_pl["customer_id"]
    test_ids = test_pl["customer_id"]
    del train_pl, test_pl; gc.collect()

    # 5. PCA
    tr, te = incremental_pca_features(TRAIN_EXTRA, TEST_EXTRA, n_components=PCA_N_COMPONENTS)
    all_train.update(tr); all_test.update(te)
    gc.collect()

    # 6. NaN паттерны
    tr, te = nan_pattern_features(TRAIN_EXTRA, TEST_EXTRA)
    all_train.update(tr); all_test.update(te)
    gc.collect()

    # 7. Подгрупповые агрегации
    tr, te = extra_subgroup_aggs(TRAIN_EXTRA, TEST_EXTRA)
    all_train.update(tr); all_test.update(te)
    gc.collect()

    print(f"\n  Всего новых признаков: {len(all_train)}")
    validate_dict(all_train, "all_train")
    validate_dict(all_test, "all_test")

    for k, v in all_train.items():
        if isinstance(v, np.ndarray):
            all_train[k] = pl.Series(k, v)
    for k, v in all_test.items():
        if isinstance(v, np.ndarray):
            all_test[k] = pl.Series(k, v)

    df_train = pl.DataFrame(all_train)
    df_train = df_train.with_columns(pl.Series("customer_id", train_ids))
    df_train.write_parquet(FE_TRAIN_PATH)
    print(f"  Сохранено: {FE_TRAIN_PATH} ({df_train.shape})")
    del df_train; gc.collect()

    df_test = pl.DataFrame(all_test)
    df_test = df_test.with_columns(pl.Series("customer_id", test_ids))
    df_test.write_parquet(FE_TEST_PATH)
    print(f"  Сохранено: {FE_TEST_PATH} ({df_test.shape})")

    print("\n  Feature Engineering завершён!")


if __name__ == "__main__":
    main()