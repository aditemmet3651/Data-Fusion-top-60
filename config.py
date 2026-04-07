import os

# ============================================================
# ПУТИ К ДАННЫМ
# ============================================================
DATA_PATH = "."
ARTIFACTS_PATH = "./artifacts"
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

# ============================================================
# ИМЕНА ФАЙЛОВ ДАННЫХ
# ============================================================
TRAIN_MAIN = os.path.join(DATA_PATH, "train_main_features.parquet")
TEST_MAIN = os.path.join(DATA_PATH, "test_main_features.parquet")
TRAIN_EXTRA = os.path.join(DATA_PATH, "train_extra_features.parquet")
TEST_EXTRA = os.path.join(DATA_PATH, "test_extra_features.parquet")
TRAIN_TARGET = os.path.join(DATA_PATH, "train_target.parquet")
SAMPLE_SUBMIT = os.path.join(DATA_PATH, "sample_submit.parquet")

# ============================================================
# АРТЕФАКТЫ
# ============================================================
SELECTED_FEATURES_700 = os.path.join(ARTIFACTS_PATH, "selected_features_700.pkl")
SELECTED_FEATURES_300 = os.path.join(ARTIFACTS_PATH, "selected_features_300.pkl")
PER_TARGET_FEATURES = os.path.join(ARTIFACTS_PATH, "per_target_features.pkl")
META_TRAIN = os.path.join(ARTIFACTS_PATH, "meta_train.parquet")
META_TEST = os.path.join(ARTIFACTS_PATH, "meta_test.parquet")
GLOBAL_AGGS_TRAIN = os.path.join(ARTIFACTS_PATH, "global_aggs_train.parquet")
GLOBAL_AGGS_TEST = os.path.join(ARTIFACTS_PATH, "global_aggs_test.parquet")
CHECKPOINT = os.path.join(ARTIFACTS_PATH, "checkpoint.pkl")
FE_TRAIN = os.path.join(ARTIFACTS_PATH, "fe_train.parquet")
FE_TEST = os.path.join(ARTIFACTS_PATH, "fe_test.parquet")
NN_OOF_PATH = os.path.join(ARTIFACTS_PATH, "nn_oof.npz")
OOF_LGBM_PATH = os.path.join(ARTIFACTS_PATH, "oof_lgbm.npy")
OOF_CAT_PATH = os.path.join(ARTIFACTS_PATH, "oof_cat.npy")
TEST_LGBM_PATH = os.path.join(ARTIFACTS_PATH, "test_lgbm.npy")
TEST_CAT_PATH = os.path.join(ARTIFACTS_PATH, "test_cat.npy")
OOF_TE_TRAIN = os.path.join(ARTIFACTS_PATH, "oof_te_train.parquet")
OOF_TE_TEST = os.path.join(ARTIFACTS_PATH, "oof_te_test.parquet")
SUBMIT_MAIN = os.path.join(ARTIFACTS_PATH, "submission_main.parquet")
SUBMIT_FINAL = "submission_final.parquet"

# ============================================================
# ПАРАМЕТРЫ
# ============================================================
SEED = 42
N_SPLITS = 5
N_SPLITS_FINAL = 5
TOP_FEATURES_META = 700
TOP_FEATURES_FINAL = 350
TURBO_SAMPLE_RATIO = 0.3
PER_TARGET_TOP_K = 150

# ============================================================
# PCA
# ============================================================
PCA_N_COMPONENTS = 30
PCA_TOP_COLS = 700

# ============================================================
# Группы таргетов
# ============================================================
TARGET_GROUPS = {
    1: ["target_1_1", "target_1_2", "target_1_3", "target_1_4", "target_1_5"],
    2: ["target_2_1", "target_2_2", "target_2_3", "target_2_4",
        "target_2_5", "target_2_6", "target_2_7", "target_2_8"],
    3: ["target_3_1", "target_3_2", "target_3_3", "target_3_4", "target_3_5"],
    4: ["target_4_1"],
    5: ["target_5_1", "target_5_2"],
    6: ["target_6_1", "target_6_2", "target_6_3", "target_6_4", "target_6_5"],
    7: ["target_7_1", "target_7_2", "target_7_3"],
    8: ["target_8_1", "target_8_2", "target_8_3"],
    9: ["target_9_1", "target_9_2", "target_9_3", "target_9_4",
        "target_9_5", "target_9_6", "target_9_7", "target_9_8"],
    10: ["target_10_1"],
}

ALL_TARGETS = []
for g in sorted(TARGET_GROUPS.keys()):
    ALL_TARGETS.extend(TARGET_GROUPS[g])

# ============================================================
# RARE / HARD
# ============================================================
RARE_THRESHOLD = 0.005

HARD_TARGETS = {
    "target_3_1", "target_9_3", "target_9_6",
    "target_5_2", "target_6_1",
}

# ============================================================
# LightGBM — CPU с максимальным параллелизмом
# (GPU не работает из-за кириллических путей Windows)
# ============================================================
LGBM_META_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=63,
    max_depth=7,
    min_data_in_leaf=80,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=1.0,
    n_jobs=2,
    verbose=-1,
    seed=SEED,
)
LGBM_META_ROUNDS = 300

LGBM_FINAL_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=128,
    min_data_in_leaf=150,
    feature_fraction=0.6,
    bagging_fraction=0.75,
    bagging_freq=5,
    lambda_l1=0.1,
    lambda_l2=1.0,
    max_bin=255,
    n_jobs=2,
    verbose=-1,
    seed=SEED,
)
LGBM_FINAL_ROUNDS = 700
LGBM_EARLY_STOPPING = 80

LGBM_RARE_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.03,
    num_leaves=31,
    min_data_in_leaf=20,
    feature_fraction=0.5,
    bagging_fraction=0.7,
    bagging_freq=5,
    lambda_l1=0.5,
    lambda_l2=5.0,
    max_bin=255,
    n_jobs=2,
    verbose=-1,
    seed=SEED,
)
LGBM_RARE_ROUNDS = 500
LGBM_RARE_EARLY_STOPPING = 120

LGBM_HARD_PARAMS = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.02,
    num_leaves=64,
    min_data_in_leaf=100,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l1=0.05,
    lambda_l2=0.5,
    max_bin=255,
    n_jobs=2,
    verbose=-1,
    seed=SEED,
)
LGBM_HARD_ROUNDS = 1200
LGBM_HARD_EARLY_STOPPING = 150

# ============================================================
# CatBoost финал — NORMAL (GPU)
# ============================================================
CAT_FINAL_PARAMS = dict(
    iterations=1200,
    learning_rate=0.05,
    depth=7,
    l2_leaf_reg=3.0,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=False,
    allow_writing_files=False,
    thread_count=6,
    random_seed=SEED,
    task_type="GPU",
    devices="0",
    gpu_ram_part=0.8,
    early_stopping_rounds=100,
)

# ============================================================
# CatBoost финал — RARE (GPU)
# ============================================================
CAT_RARE_PARAMS = dict(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=False,
    allow_writing_files=False,
    thread_count=6,
    random_seed=SEED,
    task_type="GPU",
    devices="0",
    gpu_ram_part=0.8,
    early_stopping_rounds=150,
)

# ============================================================
# DAE
# ============================================================
DAE_BOTTLENECK = 192
DAE_EPOCHS = 25
DAE_LR = 8e-4
DAE_BATCH_SIZE = 2048
DAE_CORRUPTION = 0.20
DAE_WEIGHT_DECAY = 1e-5
DAE_EXTRA_LIMIT = 400

# ============================================================
# TabM
# ============================================================
TABM_BATCH_SIZE = 512
TABM_EPOCHS = 50
TABM_LR = 5e-4
TABM_WEIGHT_DECAY = 1e-3
TABM_PATIENCE = 10
TABM_GRAD_CLIP = 1.0
TABM_HIDDEN_DIM = 256
TABM_K_ENSEMBLE = 12
TABM_N_RES_BLOCKS = 3
TABM_DROPOUT = 0.35
TABM_PLR_N_BINS = 24
TABM_MIXUP_ALPHA = 0.2
TABM_TOP_K_SWA = 5
TABM_EXTRA_LIMIT = 350

# ASL
ASL_GAMMA_NEG = 4
ASL_GAMMA_POS = 1
ASL_CLIP = 0.05