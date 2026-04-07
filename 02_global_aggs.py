"""
Шаг 2: Глобальные агрегации по extra features.
Считает sum, mean, count_nonzero по ВСЕМ extra столбцам для каждого клиента.
Использует Polars lazy execution — почти не расходует RAM.
Исправлено: Boolean -> Int8 для корректного sum_horizontal.
"""

import polars as pl
from config import *


def compute_aggs(input_path, output_path, name):
    print(f"  Считаю агрегации для {name}...")
    lazy = pl.scan_parquet(input_path)
    cols = [c for c in lazy.collect_schema().names() if c != "customer_id"]

    result = lazy.select([
        pl.col("customer_id"),
        pl.sum_horizontal(cols).alias("global_extra_sum"),
        pl.mean_horizontal(cols).alias("global_extra_mean"),
        pl.sum_horizontal([
            pl.col(c).is_not_null().cast(pl.Int8) for c in cols
        ]).alias("global_extra_nonnull"),
        pl.sum_horizontal([
            (pl.col(c) != 0).cast(pl.Int8) for c in cols
        ]).alias("global_extra_nonzero"),
    ]).collect()

    result.write_parquet(output_path)
    print(f"  Сохранено: {output_path} ({result.shape})")


def main():
    print("=" * 60)
    print("ШАГ 2: ГЛОБАЛЬНЫЕ АГРЕГАЦИИ")
    print("=" * 60)

    compute_aggs(TRAIN_EXTRA, GLOBAL_AGGS_TRAIN, "Train")
    compute_aggs(TEST_EXTRA, GLOBAL_AGGS_TEST, "Test")

    print("\n  Готово!")


if __name__ == "__main__":
    main()