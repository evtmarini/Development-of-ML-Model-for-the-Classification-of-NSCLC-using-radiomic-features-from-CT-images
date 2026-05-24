
"""
Hold-out Evaluation Module

Performs final model validation on an independent hold-out test set.

Pipeline:
- Uses fixed hold-out split defined in the main pipeline
- Selects best model and feature subset from aggregated outer CV results
- Applies ComBat harmonization and feature scaling
- Computes evaluation metrics (F1, Accuracy, Precision, Recall, AUC)

Designed for unbiased final performance estimation in radiomics studies.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from neurocombat_sklearn import CombatModel

from load_data import load_and_clean
from models import get_models_and_params

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - HOLDOUT_PIPELINE - %(message)s"
)

logger = logging.getLogger()

# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path("Data/Radiomic_Features_All.xlsx")

RESULTS_DIR = Path("Results")

OUTER_FILE = RESULTS_DIR / "Outer_cv_results" / "average_leaderboard.csv"

FEATURES_FILE = RESULTS_DIR / "Selected Features" / "selected_features_new.csv"

HOLDOUT_INDICES_FILE = RESULTS_DIR / "holdout_indices.npy"

HOLDOUT_RESULTS_DIR = RESULTS_DIR / "Holdout_results"
HOLDOUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = HOLDOUT_RESULTS_DIR / "Holdout_results.xlsx"

# ============================================================
# 1. LOAD DATA
# ============================================================

logger.info("Loading dataset...")

X, y, center = load_and_clean(DATA_FILE)

# ============================================================
# 2. LOAD HOLD-OUT SPLIT
# ============================================================

logger.info("Loading hold-out indices...")

holdout_idx = np.load(HOLDOUT_INDICES_FILE)

train_idx = np.setdiff1d(np.arange(len(X)), holdout_idx)

X_train = X.iloc[train_idx].reset_index(drop=True)
y_train = y[train_idx]
center_train = center.iloc[train_idx].reset_index(drop=True)

X_holdout = X.iloc[holdout_idx].reset_index(drop=True)
y_holdout = y[holdout_idx]
center_holdout = center.iloc[holdout_idx].reset_index(drop=True)

logger.info(f"Hold-out samples: {len(X_holdout)}")

# ============================================================
# 3. LOAD BEST MODEL
# ============================================================

logger.info("Loading best model from outer CV leaderboard...")

df_outer = pd.read_csv(OUTER_FILE)

if "F1_mean" not in df_outer.columns:
    raise ValueError("Missing 'F1_mean' column in leaderboard.")

best_row = df_outer.sort_values("F1_mean", ascending=False).iloc[0]

best_fs = best_row["FS_method"]
best_clf = best_row["Classifier"]
best_k = best_row["Top_k"]

logger.info(f"Best configuration: {best_clf} + {best_fs} (Top-{best_k})")

# ============================================================
# 4. LOAD FEATURES
# ============================================================

logger.info("Loading selected features...")

df_features = pd.read_csv(FEATURES_FILE)

# if file contains only feature_name column
if "feature_name" in df_features.columns:
    selected_features = df_features["feature_name"].tolist()

# otherwise use full metadata format
else:
    mask = (
        (df_features["FS_method"] == best_fs) &
        (df_features["Classifier"] == best_clf) &
        (df_features["Best_k"] == best_k)
    )

    if mask.sum() == 0:
        raise ValueError("No matching feature set found.")

    selected_features = (
        df_features.loc[mask, "Selected_Features"]
        .iloc[0]
        .split(",")
    )

selected_features = [f.strip() for f in selected_features]

missing_feats = [f for f in selected_features if f not in X.columns]

if missing_feats:
    raise ValueError(f"Missing features detected: {missing_feats[:5]}")

logger.info(f"Loaded {len(selected_features)} selected features.")

# ============================================================
# 5. COMBAT + SCALING
# ============================================================

logger.info("Applying ComBat harmonization + scaling...")

centers_all = pd.concat([center_train, center_holdout]).astype("category")

codes = centers_all.cat.codes.values

center_codes_train = codes[:len(center_train)].reshape(-1, 1)
center_codes_holdout = codes[len(center_train):].reshape(-1, 1)

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

# ============================================================
# 6. TRAIN MODEL
# ============================================================

logger.info("Training final model...")

models, _ = get_models_and_params()

if best_clf not in models:
    raise ValueError(f"Model '{best_clf}' not found.")

model = models[best_clf]

model.fit(X_train_sel, y_train)

# ============================================================
# 7. EVALUATION
# ============================================================

logger.info("Evaluating on hold-out set...")

y_pred = model.predict(X_holdout_sel)

f1 = f1_score(y_holdout, y_pred, average="weighted")
acc = accuracy_score(y_holdout, y_pred)
prec = precision_score(
    y_holdout,
    y_pred,
    average="weighted",
    zero_division=0
)

rec = recall_score(
    y_holdout,
    y_pred,
    average="weighted",
    zero_division=0
)

# ============================================================
# AUC
# ============================================================

auc = None

if hasattr(model, "predict_proba"):

    try:
        le = LabelEncoder()

        y_holdout_enc = le.fit_transform(y_holdout)

        y_proba = model.predict_proba(X_holdout_sel)

        if len(np.unique(y_holdout_enc)) == 2:
            auc = roc_auc_score(y_holdout_enc, y_proba[:, 1])
        else:
            auc = roc_auc_score(
                y_holdout_enc,
                y_proba,
                multi_class="ovr",
                average="weighted"
            )

    except Exception as e:
        logger.warning(f"AUC computation failed: {e}")

# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "FS_method": best_fs,
        "Classifier": best_clf,
        "Top_k": best_k,
        "F1": round(f1, 3),
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "AUC": round(auc, 3) if auc is not None else None
    }
])

results.to_excel(OUTPUT_CSV, index=False)

logger.info(f"Hold-out results saved to: {OUTPUT_CSV}")

print("\n===== HOLD-OUT RESULTS =====")
print(results)
