
# 🏆 Data Fusion Contest 2026 — Задача 2 «Киберполка»

## Efficient Multi-Model Solution | Private 0.845 · Top-60

> **Решение задачи multi-label классификации 41 банковского продукта для 1 000 000 клиентов на полностью анонимных и обфусцированных данных. Top-60 на private leaderboard при минимальных аппаратных требованиях (6 ГБ VRAM, 10 ГБ RAM).**

---

## 📋 Описание задачи

**Data Fusion Contest 2026** — ежегодное ML-соревнование, организуемое крупным российским банком.

**Задача «Киберполка»**: предсказать вероятности открытия каждого из 41 финансового продукта (счета, карты, услуги) для 250 000 клиентов банка. Бизнес использует эти вероятности для гибкой настройки рекомендаций.

### Особенности данных

| Параметр | Значение |
|----------|----------|
| **Клиентов** | 1 000 000 (750K train / 250K test) |
| **Таргетов** | 41 бинарный (multi-label) |
| **Основных признаков** | 199 (67 категориальных + 132 числовых) |
| **Дополнительных признаков** | 2 241 (числовые) |
| **Анонимизация** | Полная — названия продуктов и признаков не раскрыты |
| **Метрика** | Macro Averaged ROC-AUC |
| **Public/Private split** | 30% / 70% |

### Ключевые вызовы

- **Экстремальный дисбаланс классов**: positive rate от 0.01% (`target_2_8`, 83 примера) до 31.5% (`target_10_1`), медиана 0.79%
- **Массивные пропуски**: медиана 29.7% в основных и 52.1% в дополнительных признаках
- **Сильные выбросы**: 105 из 132 числовых признаков содержат экстремальные значения
- **Высокая размерность**: 2 441 признак при необходимости обучать 41 модель
- **Слабые корреляции**: средняя |корреляция| признаков с таргетами всего 0.017
- **Мультиколлинеарность**: 9 пар признаков с |корр| > 0.95

### Структура таргетов

41 таргет организован в **10 продуктовых групп**. Большинство таргетов крайне разреженные:

```
Распределение positive rate:
     <1%:  24 таргета   ████████████████████████
    1-5%:  10 таргетов  ██████████
   5-10%:   4 таргета   ████
  10-20%:   1 таргет    █
  20-50%:   2 таргета   ██

79.6% клиентов владеют только 1 продуктом
```

---

## 📊 Результаты

| Метрика | Значение |
|---------|----------|
| **Private Score** | **0.845** |
| **Leaderboard Position** | **Top 60** |
| **RAM Usage** | **10 GB** |
| **VRAM Usage** | **6 GB** |
| **Training Time** | ~25 часов (одна GPU) |

### OOF Performance по моделям

| Модель | OOF Macro AUC | LB Score |
|--------|---------------|----------|
| LightGBM + CatBoost (Step 04) | 0.8380 | **0.8430** |
| TabM + DAE (Step 05a) | 0.8329 | **0.8428** |
| MLP + PLR + ASL (Step 05) | 0.8310 | **0.8392** |
| **4-Model Blend (Step 06)** | **0.8450** | **0.845** |

---

## 🏗️ Архитектура решения

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE OVERVIEW                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │  Step 0  │──  │  Step 1  │─── │   Steps 2/2a/3   │   │
│  │ Feature  │    │   Meta   │    │  Global Aggs +   │   │
│  │Selection │    │ Features │    │  OOF TE + FE     │   │
│  └──────────┘    └──────────┘    └──────────────────┘   │
│       │                                    │            │
│       ▼                                    ▼            │
│  ┌─────────────────────────────────────────────────┐    │
│  │              FEATURE MATRIX                     │    │
│  │  Main(199) + Extra(350) + Meta(~100) + FE(~300) │    │
│  │  + OOF TE(300) + DAE Embeddings(192)            │    │
│  └─────────────────────────────────────────────────┘    │
│       │              │               │                  │
│       ▼              ▼               ▼                  │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐              │
│  │ Step 04 │   │ Step 05  │   │ Step 05a │              │
│  │ LightGBM│   │   MLP    │   │  DAE +   │              │
│  │+CatBoost│   │ PLR+ASL  │   │  TabM    │              │
│  └────┬────┘   └────┬─────┘   └────┬─────┘              │
│       │             │              │                    │
│       ▼             ▼              ▼                    │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Step 06: BLEND                     │    │
│  │  Per-target OOF-optimized rank averaging        │    │
│  │  4 models × 41 targets = 164 weight combos      │    │
│  └─────────────────────────────────────────────────┘    │
│                        │                                │
│                        ▼                                │
│               submission.parquet                        │
└─────────────────────────────────────────────────────────┘
```

### Ключевые инновации

1. **Per-Target Feature Selection** — каждый из 41 таргетов получает индивидуальный набор top-150 extra features вместо общего, что даёт +0.0035 AUC
2. **Adaptive Hyperparameters** — отдельные конфигурации GBDT для Normal / Rare / Hard таргетов
3. **DAE Pretraining** — Denoising Autoencoder с Gated Linear units извлекает 192-мерные эмбеддинги из всех признаков (train + test)
4. **TabM с Batch Ensemble** — k=12 sub-networks через параметризацию r·W·s+b вместо k отдельных моделей
5. **RAM-Safe Design** — mmap для массивов, батчевая обработка test, on-disk checkpointing

---

## 📁 Структура проекта

```
├── config.py                      # Все гиперпараметры и пути
├── 00_feature_selection.py        # Per-target + global feature selection
├── 01_build_meta.py               # OOF meta-features (LightGBM)
├── 02_global_aggs.py              # Глобальные агрегации по extra
├── 02a_oof_target_encoding.py     # OOF Target Encoding
├── 03_feature_engineering.py      # PCA, interactions, NaN patterns
├── 04_train_boosting.py           # LightGBM + CatBoost (per-target)
├── 05_train_nn_mlp.py             # MLP: PLR + ASL + SWA
├── 05a_train_nn_tabm.py           # DAE Pretraining + TabM
├── 06_blend_final.py              # 4-model OOF-optimized blend
├── requirements.txt               # Зависимости
└── README.md
```

---

## 🛠️ Установка и запуск

### Требования

- Python 3.11+
- CUDA 11.8+ (для GPU-ускорения NN и CatBoost)
- **10 GB RAM** (16 GB рекомендуется)
- **6 GB VRAM** (8 GB рекомендуется)

### 1. Установка

```bash
pip install -r requirements.txt
```

### 2. Данные

Положите файлы данных в корневую папку (или измените `DATA_PATH` в `config.py`):

```
./train_main_features.parquet
./test_main_features.parquet
./train_extra_features.parquet
./test_extra_features.parquet
./train_target.parquet
./sample_submit.parquet
```

### 3. Запуск

**Пошагово:**

```bash
python 00_feature_selection.py
python 01_build_meta.py
python 02_global_aggs.py
python 02a_oof_target_encoding.py
python 03_feature_engineering.py
python 04_train_boosting.py
python 05_train_nn_mlp.py
python 05a_train_nn_tabm.py
python 06_blend_final.py
```

### 4. Результат

```
artifacts/submission_blend_4model.parquet
```

---

## 🔬 Подробное описание шагов

### Step 0 — Feature Selection

**Проблема**: 2 241 extra-признак, но многие из них неинформативны, а обучение на всех — слишком дорого.

**Решение**:

1. **Turbo Sampling** — для ускорения вычислений: 100% строк с редкими таргетами + 30% остальных
2. **Variance filter** — отсев признаков с нулевой/около-нулевой дисперсией
3. **Dual scoring** — для каждого признака считаются:
   - Абсолютная корреляция с каждым таргетом
   - LightGBM feature importance (gain) по каждому таргету
4. **Global top-700/350** — взвешенная комбинация (0.4·corr + 0.6·lgbm)
5. **Per-target top-150** — индивидуальный набор для каждого таргета (0.35·corr + 0.65·lgbm)

```
Уникальных extra features across targets: ~600
Пересечение с global top-350: ~210
→ Per-target подход захватывает признаки, важные для конкретных продуктов
```

### Step 1 — Meta Features

OOF-предсказания LightGBM для каждого из 41 таргета + cross-target агрегации:

- `meta_all_mean/max/std/min/q25/q75` — статистики по всем meta-предсказаниям
- `meta_grpN_mean/max/std` — статистики внутри продуктовых групп
- `meta_rank_*` — ранги предсказаний внутри клиента

### Step 2 — Global Aggregations

Polars lazy execution для вычисления `sum`, `mean`, `count_nonzero` по всем 2 241 extra-столбцам практически без расхода RAM.

### Step 2a — OOF Target Encoding

OOF TE для каждой пары (категория, таргет) с smoothing=50:

```
smoothed = (count × mean + 50 × global_mean) / (count + 50)
```

- 5-fold OOF для train (без лика)
- Full train statistics для test
- Отбор top-300 по std

### Step 3 — Feature Engineering

| Блок | Признаков | Описание |
|------|-----------|----------|
| Частотное кодирование | 67 | Доля каждой категории в train+test |
| Групповые Z-scores | ~100 | `(x - group_mean) / group_std` для top категорий |
| Попарные взаимодействия | 56 | `a×b` и `a/b` для top-8 числовых |
| Log/sqrt трансформации | 40 | `log1p(|x|)`, `sqrt(|x|)` для top-20 числовых |
| Incremental PCA | 30 | По top-700 extra features |
| NaN-паттерны | ~20 | Число null/nonzero в группах по 500 столбцов |
| Subgroup агрегации | ~30 | sum/mean/std по подгруппам extra |

### Step 4 — GBDT Training (основная модель)

**Архитектура**: для каждого из 41 таргета обучаются LightGBM + CatBoost с 5-fold CV.

**Adaptive regime**:

| Режим | Условие | LightGBM | CatBoost |
|-------|---------|----------|----------|
| **Normal** | pos_rate ≥ 0.5% | lr=0.03, leaves=128, 700 rounds | lr=0.05, depth=7, 1200 iter |
| **Rare** | pos_rate < 0.5% | lr=0.03, leaves=31, 500 rounds | lr=0.03, depth=6, 1500 iter |
| **Hard** | Ручной список | lr=0.02, leaves=64, 1200 rounds | Standard CatBoost |

**RAM-safe design**:
- Base features сохраняются как `.npy` и читаются через `mmap_mode='r'`
- Per-target extra подгружаются динамически для каждого таргета
- Test predictions батчами по 50K строк
- Checkpointing после каждого таргета (возобновление при сбое)

**Blending**: Per-target grid search веса LGBM vs CatBoost через rank averaging.

### Step 5 — MLP (PLR + ASL)

```
Input → PiecewiseLinearEncoding(24 bins) → Linear(256) → BN → SiLU → Dropout(0.3)
    → Linear(256) → BN → SiLU → Dropout(0.3)
    → Linear(128) → BN → SiLU → Dropout(0.21)
    → Linear(41)
```

- **PLR**: Piecewise Linear Encoding — непараметрическое кодирование числовых признаков через квантильные бины
- **ASL**: Asymmetric Loss (γ_neg=4, γ_pos=1, clip=0.05) — фокус на позитивном классе при дисбалансе
- **SWA**: Stochastic Weight Averaging из top-3 эпох
- **QuantileTransformer**: fit per fold, transform train+val+test

### Step 5a — DAE + TabM (лучшая NN)

**Phase 1: Denoising Autoencoder**

```
Encoder: GatedLinear(D→512) → BN → SiLU → Drop(0.2)
       → GatedLinear(512→256) → BN → SiLU → Drop(0.15)
       → Linear(256→192) → BN → SiLU

Decoder: Symmetric architecture (192→D)
```

- Обучается на train+test (unsupervised)
- Masked corruption: 20% non-null значений зануляются
- Weighted reconstruction: вес 3× для замаскированных позиций
- Результат: 192-мерные эмбеддинги для каждого клиента

**Phase 2: TabM**

```
Input → PLR(24 bins) → Linear(D→256) → BN → SiLU
    → Expand to k=12 copies (Batch Ensemble)
    → ResBlock_BE(256) × 3 [each: BN→SiLU→LinearBE→BN→SiLU→Drop→LinearBE + skip]
    → Head: BN→SiLU→Drop→LinearBE(256→128)→BN→SiLU→Drop→LinearBE(128→41)
    → Mean over k=12 → Sigmoid
```

- **Batch Ensemble**: `y = Linear(x ⊙ r) ⊙ s + b` с k=12 параметрами (r, s, b)
- **Manifold Mixup**: λ ~ Beta(0.2, 0.2), случайный слой для mixup
- **SWA**: top-5 checkpoints

### Step 6 — 4-Model Blend

Per-target двухэтапный grid search:

1. **Coarse grid** (step=0.1): перебор всех комбинаций 4 весов
2. **Fine grid** (step=0.02): уточнение вокруг лучшей точки

Все предсказания конвертируются в ранги перед смешиванием (rank averaging).

**Средние веса по 41 таргету:**

| Модель | Средний вес | Использование |
|--------|-------------|---------------|
| LightGBM | 0.31 | 41/41 таргетов |
| CatBoost | 0.29 | 41/41 таргетов |
| TabM | 0.26 | 38/41 таргетов |
| MLP | 0.14 | 29/41 таргетов |

---

## ⚙️ Конфигурация

Все параметры сосредоточены в `config.py`:

```python
# Feature Selection
TOP_FEATURES_META = 700       # Extra features для meta-модели
TOP_FEATURES_FINAL = 350      # Extra features для финальной модели
PER_TARGET_TOP_K = 150        # Per-target extra features

# Cross-Validation
N_SPLITS_FINAL = 5            # Stratified K-Fold
SEED = 42

# Rare/Hard targets
RARE_THRESHOLD = 0.005        # pos_rate < 0.5%
HARD_TARGETS = {"target_3_1", "target_9_3", "target_9_6",
                "target_5_2", "target_6_1"}

# PCA
PCA_N_COMPONENTS = 30
PCA_TOP_COLS = 700

# DAE
DAE_BOTTLENECK = 192
DAE_EPOCHS = 25
DAE_CORRUPTION = 0.20

# TabM
TABM_K_ENSEMBLE = 12
TABM_HIDDEN_DIM = 256
TABM_N_RES_BLOCKS = 3
TABM_MIXUP_ALPHA = 0.2
```

