"""
Hold-out Explainability Module (SHAP & LIME)

- Uses the fixed hold-out split defined in main pipeline
- Avoids any retraining bias or data leakage
- Saves global & local explainability outputs for publication
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

from sklearn.preprocessing import StandardScaler
from neurocombat_sklearn import CombatModel

from src.load_data import load_and_clean
from src.models import get_models_and_params
from src.explainability import run_explainability


# LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - HOLDOUT_EXPLAINABILITY - %(message)s"
)
logger = logging.getLogger()


# PATHS

DATA_FILE = Path("data/Radiomic_Features_All.xlsx")
RESULTS_DIR = Path("results")
EXPL_DIR = RESULTS_DIR / "explainability" / "holdout"

OUTER_FILE = RESULTS_DIR / "outer_cv_results.csv"
FEATURES_FILE = RESULTS_DIR / "selected_features.csv"
HOLDOUT_INDICES_FILE = RESULTS_DIR / "holdout_indices.npy"

EXPL_DIR.mkdir(parents=True, exist_ok=True)


# 1. LOAD DATA

logger.info("Loading dataset...")
X, y, center = load_and_clean(DATA_FILE)


# 2. LOAD FIXED HOLD-OUT INDICES

logger.info("Loading fixed hold-out indices from main...")
holdout_indices = np.load(HOLDOUT_INDICES_FILE)

train_indices = np.setdiff1d(np.arange(len(y)), holdout_indices)

X_train = X.iloc[train_indices].copy()
y_train = y[train_indices]
center_train = center.iloc[train_indices].copy()

X_holdout = X.iloc[holdout_indices].copy()
y_holdout = y[holdout_indices]
center_holdout = center.iloc[holdout_indices].copy()

logger.info(f"Hold-out samples: {len(X_holdout)}")


# 3. LOAD BEST MODEL CONFIGURATION

df_outer = pd.read_csv(OUTER_FILE)
best_row = df_outer.sort_values("Test_F1", ascending=False).iloc[0]

best_fs = best_row["FS_method"]
best_clf = best_row["Classifier"]
best_k = best_row["Top_k"]

logger.info(
    f"Best model: {best_fs} + {best_clf} (Top-{best_k})"
)

df_features = pd.read_csv(FEATURES_FILE)
feats_row = df_features[
    (df_features["FS_method"] == best_fs) &
    (df_features["Classifier"] == best_clf) &
    (df_features["Best_k"] == best_k)
].iloc[0]

selected_features = feats_row["Selected_Features"].split(",")

# 4. COMBAT + SCALING (TRAIN → HOLD-OUT)

logger.info("Applying ComBat harmonization and scaling...")

center_codes_train = pd.factorize(center_train)[0].reshape(-1, 1)
center_codes_holdout = pd.factorize(center_holdout)[0].reshape(-1, 1)

combat = CombatModel()
X_train_h = combat.fit_transform(X_train, center_codes_train)
X_holdout_h = combat.transform(X_holdout, center_codes_holdout)

scaler = StandardScaler()
X_train_s = pd.DataFrame(
    scaler.fit_transform(X_train_h),
    columns=X.columns
)
X_holdout_s = pd.DataFrame(
    scaler.transform(X_holdout_h),
    columns=X.columns
)

X_train_sel = X_train_s[selected_features]
X_holdout_sel = X_holdout_s[selected_features]


# 5. LOAD & TRAIN BEST MODEL

models, _ = get_models_and_params()
model = models[best_clf]
model.fit(X_train_sel, y_train)


# 6. RUN SHAP & LIME (HOLD-OUT ONLY)

logger.info("Running SHAP & LIME on HOLD-OUT set...")

run_explainability(
    model=model,
    X_train=X_train_sel,
    X_test=X_holdout_sel,
    y_train=y_train,
    y_test=y_holdout,
    feature_names=X_holdout_sel.columns.tolist(),
    output_dir=EXPL_DIR,
    fold_name="holdout"
)

logger.info("Explainability analysis completed successfully.")
logger.info(f"Results saved to: {EXPL_DIR}")
