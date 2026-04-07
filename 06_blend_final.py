"""
Шаг 6: 4-Model Blend с OOF-оптимизацией.
Модели:
  1. LightGBM  (из 04)
  2. CatBoost  (из 04)
  3. MLP       (из 05 — PLR+ASL)
  4. TabM      (из 05a — DAE+TabM)

Per-target grid search оптимальных весов на OOF.
Генерирует: OOF-optimized бленд + fixed-weight варианты.
"""

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import joblib
import gc
import os
from config import *

BLEND_SUBMIT = os.path.join(ARTIFACTS_PATH, "submission_blend_4model.parquet")


def rank_col(x):
    return rankdata(x).astype(np.float64) / len(x)


def to_logits(probs):
    eps = 1e-7
    p = np.clip(probs, eps, 1 - eps)
    return np.log(p / (1 - p))


def make_submit(test_ids, pred_cols, predictions, filename):
    sub = pd.DataFrame({"customer_id": test_ids})
    sub["customer_id"] = sub["customer_id"].astype(np.int32)
    for t, col in enumerate(pred_cols):
        sub[col] = to_logits(predictions[:, t]).astype(np.float64)
    if os.path.exists(SAMPLE_SUBMIT):
        sample = pd.read_parquet(SAMPLE_SUBMIT)
        sub = sample[["customer_id"]].merge(sub, on="customer_id", how="left")
        sub["customer_id"] = sub["customer_id"].astype(np.int32)
    sub = sub.fillna(-10.0)
    sub.to_parquet(filename, index=False)
    print(f"  ✅ {filename} ({sub.shape})")


def align_nn_predictions(oof, test_preds, nn_target_cols, nn_train_ids,
                          nn_test_ids, target_cols, train_id_ref,
                          test_id_ref, n_train, n_test, name):
    """Выравнивает NN предсказания по ID и колонкам."""
    # Выравниваем по train ID
    if nn_train_ids != train_id_ref:
        print(f"  ⚠️ {name}: пересортировываю train IDs")
        id_to_idx = {cid: i for i, cid in enumerate(nn_train_ids)}
        reorder = [id_to_idx.get(cid, 0) for cid in train_id_ref]
        oof = oof[reorder]

    # Выравниваем по test ID
    if nn_test_ids != test_id_ref:
        print(f"  ⚠️ {name}: пересортировываю test IDs")
        id_to_idx = {cid: i for i, cid in enumerate(nn_test_ids)}
        reorder = [id_to_idx.get(cid, 0) for cid in test_id_ref]
        test_preds = test_preds[reorder]

    # Выравниваем колонки
    col_map = {c: i for i, c in enumerate(nn_target_cols)}
    oof_aligned = np.full((n_train, len(target_cols)), 0.5, dtype=np.float64)
    test_aligned = np.full((n_test, len(target_cols)), 0.5, dtype=np.float64)

    for j, col in enumerate(target_cols):
        if col in col_map:
            idx = col_map[col]
            oof_aligned[:, j] = oof[:, idx]
            test_aligned[:, j] = test_preds[:, idx]

    return oof_aligned, test_aligned


def grid_search_4weights(oof_list, y_t):
    """Двухэтапный grid search по 4 весам."""
    ranks = [rank_col(o) for o in oof_list]
    best_auc = 0.0
    best_w = (0.25, 0.25, 0.25, 0.25)

    # Грубый grid (step=0.1)
    for w1 in np.arange(0.0, 1.01, 0.1):
        for w2 in np.arange(0.0, 1.01 - w1, 0.1):
            for w3 in np.arange(0.0, 1.01 - w1 - w2, 0.1):
                w4 = round(1.0 - w1 - w2 - w3, 2)
                if w4 < -0.01:
                    continue
                w4 = max(0.0, w4)

                blend = w1 * ranks[0] + w2 * ranks[1] + w3 * ranks[2] + w4 * ranks[3]
                try:
                    auc = roc_auc_score(y_t, blend)
                    if auc > best_auc:
                        best_auc = auc
                        best_w = (w1, w2, w3, w4)
                except Exception:
                    pass

    # Тонкий grid (step=0.02) вокруг лучшего
    w1_c, w2_c, w3_c, w4_c = best_w
    for d1 in np.arange(-0.1, 0.11, 0.02):
        for d2 in np.arange(-0.1, 0.11, 0.02):
            for d3 in np.arange(-0.1, 0.11, 0.02):
                w1 = round(w1_c + d1, 2)
                w2 = round(w2_c + d2, 2)
                w3 = round(w3_c + d3, 2)
                w4 = round(1.0 - w1 - w2 - w3, 2)

                if w1 < 0 or w2 < 0 or w3 < 0 or w4 < 0:
                    continue
                if w1 > 1 or w2 > 1 or w3 > 1 or w4 > 1:
                    continue

                blend = w1 * ranks[0] + w2 * ranks[1] + w3 * ranks[2] + w4 * ranks[3]
                try:
                    auc = roc_auc_score(y_t, blend)
                    if auc > best_auc:
                        best_auc = auc
                        best_w = (w1, w2, w3, w4)
                except Exception:
                    pass

    return best_w, best_auc


def grid_search_3weights(oof_list, y_t):
    """Grid search по 3 весам."""
    ranks = [rank_col(o) for o in oof_list]
    best_auc = 0.0
    best_w = (0.33, 0.33, 0.34)

    for w1 in np.arange(0.0, 1.01, 0.05):
        for w2 in np.arange(0.0, 1.01 - w1, 0.05):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0:
                continue
            blend = w1 * ranks[0] + w2 * ranks[1] + w3 * ranks[2]
            try:
                auc = roc_auc_score(y_t, blend)
                if auc > best_auc:
                    best_auc = auc
                    best_w = (w1, w2, w3)
            except Exception:
                pass

    return best_w, best_auc


def grid_search_2weights(oof_list, y_t):
    """Grid search по 2 весам."""
    ranks = [rank_col(o) for o in oof_list]
    best_auc = 0.0
    best_w = (0.5, 0.5)

    for w in np.arange(0.0, 1.01, 0.02):
        blend = w * ranks[0] + (1 - w) * ranks[1]
        try:
            auc = roc_auc_score(y_t, blend)
            if auc > best_auc:
                best_auc = auc
                best_w = (w, 1 - w)
        except Exception:
            pass

    return best_w, best_auc


def main():
    print("=" * 60)
    print("ШАГ 6: 4-MODEL BLEND С OOF-ОПТИМИЗАЦИЕЙ")
    print("=" * 60)

    # ── IDs ──
    train_ids_pl = pl.read_parquet(TRAIN_MAIN, columns=["customer_id"])
    test_ids_pl = pl.read_parquet(TEST_MAIN, columns=["customer_id"])
    train_id_ref = train_ids_pl["customer_id"].to_list()
    test_id_ref = test_ids_pl["customer_id"].to_list()
    del train_ids_pl, test_ids_pl; gc.collect()

    n_train = len(train_id_ref)
    n_test = len(test_id_ref)

    # ── Targets ──
    target_df = pd.read_parquet(TRAIN_TARGET)
    target_cols = [c for c in target_df.columns if c.startswith("target")]
    target_df = target_df.set_index("customer_id").loc[train_id_ref].reset_index()
    y_train = target_df[target_cols].values.astype(np.float32)
    del target_df; gc.collect()

    n_targets = len(target_cols)
    pred_cols = [c.replace("target", "predict") for c in target_cols]

    # ── Загрузка моделей ──
    model_names = []
    oof_sources = []
    test_sources = []

    # 1 & 2: LGBM + CatBoost
    if (os.path.exists(OOF_LGBM_PATH) and os.path.exists(OOF_CAT_PATH) and
            os.path.exists(TEST_LGBM_PATH) and os.path.exists(TEST_CAT_PATH)):
        oof_lgbm = np.load(OOF_LGBM_PATH).astype(np.float64)
        oof_cat = np.load(OOF_CAT_PATH).astype(np.float64)
        test_lgbm = np.load(TEST_LGBM_PATH).astype(np.float64)
        test_cat = np.load(TEST_CAT_PATH).astype(np.float64)
        model_names.extend(["LGBM", "CatBoost"])
        oof_sources.extend([oof_lgbm, oof_cat])
        test_sources.extend([test_lgbm, test_cat])
        print("  ✅ LGBM + CatBoost загружены")
    else:
        print("  ❌ GBDT OOF не найден!")
        return

    # 3: MLP
    mlp_path = os.path.join(ARTIFACTS_PATH, "nn_oof.npz")
    if os.path.exists(mlp_path):
        mlp_data = np.load(mlp_path, allow_pickle=True)
        oof_mlp, test_mlp = align_nn_predictions(
            mlp_data["oof_preds"].astype(np.float64),
            mlp_data["test_preds"].astype(np.float64),
            list(mlp_data["target_cols"]),
            list(mlp_data["train_ids"]),
            list(mlp_data["test_ids"]),
            target_cols, train_id_ref, test_id_ref,
            n_train, n_test, "MLP"
        )
        model_names.append("MLP")
        oof_sources.append(oof_mlp)
        test_sources.append(test_mlp)
        print("  ✅ MLP загружен")
        del mlp_data
    else:
        print("  ⚠️ MLP не найден")

    # 4: TabM
    tabm_path = os.path.join(ARTIFACTS_PATH, "nn1_oof.npz")
    if os.path.exists(tabm_path):
        tabm_data = np.load(tabm_path, allow_pickle=True)
        oof_tabm, test_tabm = align_nn_predictions(
            tabm_data["oof_preds"].astype(np.float64),
            tabm_data["test_preds"].astype(np.float64),
            list(tabm_data["target_cols"]),
            list(tabm_data["train_ids"]),
            list(tabm_data["test_ids"]),
            target_cols, train_id_ref, test_id_ref,
            n_train, n_test, "TabM"
        )
        model_names.append("TabM")
        oof_sources.append(oof_tabm)
        test_sources.append(test_tabm)
        print("  ✅ TabM загружен")
        del tabm_data
    else:
        print("  ⚠️ TabM не найден")

    n_models = len(model_names)
    print(f"\n  Моделей: {n_models} ({', '.join(model_names)})")

    if n_models < 2:
        print("  ❌ Нужно минимум 2 модели!")
        return

    # ── Solo AUCs ──
    print("\n  Solo OOF AUCs:")
    for name, oof in zip(model_names, oof_sources):
        aucs = []
        for t in range(n_targets):
            if len(np.unique(y_train[:, t])) < 2:
                continue
            try:
                aucs.append(roc_auc_score(y_train[:, t], oof[:, t]))
            except Exception:
                pass
        print(f"    {name:>10s}: {np.mean(aucs):.4f}")

    # ── Per-target optimization ──
    print(f"\n  Per-target optimization ({n_models} моделей)...")

    header = f"  {'Target':<20s}"
    for name in model_names:
        header += f" {name[:4]:>6s}"
    for name in model_names:
        header += f" w_{name[:2]:>3s}"
    header += f" {'Blend':>7s}"
    print(header)
    print("  " + "-" * len(header))

    blend_test = np.zeros((n_test, n_targets), dtype=np.float64)
    per_target_info = {}

    for t in range(n_targets):
        y_t = y_train[:, t]
        col_name = target_cols[t]

        if len(np.unique(y_t)) < 2:
            blend_test[:, t] = np.mean(
                [rank_col(s[:, t]) for s in test_sources], axis=0
            )
            per_target_info[col_name] = {
                "weights": {n: 1.0 / n_models for n in model_names},
                "oof_auc": 0.5
            }
            continue

        oof_list = [s[:, t] for s in oof_sources]

        if n_models == 4:
            best_w, best_auc = grid_search_4weights(oof_list, y_t)
        elif n_models == 3:
            best_w, best_auc = grid_search_3weights(oof_list, y_t)
        else:
            best_w, best_auc = grid_search_2weights(oof_list, y_t)

        # Apply to test
        test_ranks = [rank_col(s[:, t]) for s in test_sources]
        blend_test[:, t] = sum(w * r for w, r in zip(best_w, test_ranks))

        weights_dict = {n: w for n, w in zip(model_names, best_w)}
        per_target_info[col_name] = {
            "weights": weights_dict,
            "oof_auc": best_auc
        }

        # Compute solo AUCs for logging
        solo_aucs = []
        for s in oof_sources:
            try:
                solo_aucs.append(roc_auc_score(y_t, s[:, t]))
            except Exception:
                solo_aucs.append(0.5)

        line = f"  {col_name:<20s}"
        for a in solo_aucs:
            line += f" {a:>6.4f}"
        for w in best_w:
            line += f" {w:>5.2f}"
        line += f" {best_auc:>7.4f}"

        # Flag if MLP weight is 0
        if n_models >= 3 and "MLP" in model_names:
            mlp_idx = model_names.index("MLP")
            if best_w[mlp_idx] < 0.01:
                line += " (no MLP)"

        print(line)

    # ── Summary ──
    aucs = [v["oof_auc"] for v in per_target_info.values() if v["oof_auc"] > 0.5]
    mean_blend = np.mean(aucs) if aucs else 0.0

    print(f"\n  {'=' * 50}")
    print(f"  Blend OOF Macro AUC: {mean_blend:.4f}")
    print(f"  Min: {np.min(aucs):.4f}, Max: {np.max(aucs):.4f}")

    # Average weights
    avg_w = {n: 0.0 for n in model_names}
    n_valid = 0
    for info in per_target_info.values():
        if info["oof_auc"] > 0.5:
            for n, w in info["weights"].items():
                avg_w[n] += w
            n_valid += 1

    print(f"\n  Средние веса ({n_valid} таргетов):")
    for name in model_names:
        print(f"    {name:>10s}: {avg_w[name] / max(n_valid, 1):.3f}")

    # How often each model has weight > 0
    usage = {n: 0 for n in model_names}
    for info in per_target_info.values():
        for n, w in info["weights"].items():
            if w > 0.01:
                usage[n] += 1
    print(f"\n  Использование (w > 0.01):")
    for name in model_names:
        print(f"    {name:>10s}: {usage[name]}/{n_valid} таргетов")

    # Worst/best
    sorted_targets = sorted(per_target_info.items(),
                            key=lambda x: x[1]["oof_auc"])
    print("\n  Worst 5:")
    for name, info in sorted_targets[:5]:
        w_str = " ".join(f"{n[:3]}={w:.2f}" for n, w in info["weights"].items())
        print(f"    {name:>18s}: {info['oof_auc']:.4f} ({w_str})")

    print("\n  Best 5:")
    for name, info in sorted_targets[-5:]:
        w_str = " ".join(f"{n[:3]}={w:.2f}" for n, w in info["weights"].items())
        print(f"    {name:>18s}: {info['oof_auc']:.4f} ({w_str})")

    # ── Сабмиты ──
    print("\n  Создаю сабмиты...")

    # Main: OOF-optimized blend
    make_submit(test_id_ref, pred_cols, blend_test, BLEND_SUBMIT)

    # Также 3-model blend без MLP для сравнения
    if "MLP" in model_names:
        mlp_idx = model_names.index("MLP")
        names_no_mlp = [n for i, n in enumerate(model_names) if i != mlp_idx]
        oof_no_mlp = [s for i, s in enumerate(oof_sources) if i != mlp_idx]
        test_no_mlp = [s for i, s in enumerate(test_sources) if i != mlp_idx]

        blend_test_3 = np.zeros((n_test, n_targets), dtype=np.float64)
        aucs_3 = []
        for t in range(n_targets):
            y_t = y_train[:, t]
            if len(np.unique(y_t)) < 2:
                blend_test_3[:, t] = np.mean(
                    [rank_col(s[:, t]) for s in test_no_mlp], axis=0
                )
                continue
            oof_list_3 = [s[:, t] for s in oof_no_mlp]
            best_w_3, best_auc_3 = grid_search_3weights(oof_list_3, y_t)
            test_ranks_3 = [rank_col(s[:, t]) for s in test_no_mlp]
            blend_test_3[:, t] = sum(w * r for w, r in zip(best_w_3, test_ranks_3))
            aucs_3.append(best_auc_3)

        mean_3 = np.mean(aucs_3) if aucs_3 else 0.0
        print(f"\n  3-model (без MLP) OOF AUC: {mean_3:.4f}")
        make_submit(test_id_ref, pred_cols, blend_test_3,
                    os.path.join(ARTIFACTS_PATH, "submission_blend_3model.parquet"))

        print(f"\n  4-model vs 3-model: {mean_blend:.4f} vs {mean_3:.4f} "
              f"({'4-model лучше' if mean_blend > mean_3 else '3-model лучше'})")

    # Save blend info
    joblib.dump({
        "per_target_info": per_target_info,
        "model_names": model_names,
        "mean_blend_auc": mean_blend,
        "n_models": n_models,
    }, os.path.join(ARTIFACTS_PATH, "blend_info.pkl"))

    print(f"\n  🏆 Рекомендуемый сабмит: {BLEND_SUBMIT}")
    print(f"  Ожидаемый LB: ~{mean_blend + 0.004:.4f}")
    print("  ✅ Готово!")


if __name__ == "__main__":
    main()