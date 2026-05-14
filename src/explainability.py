"""
SHAP Reporting Module

Generates model interpretability outputs for both outer cross-validation 
and final hold-out evaluation.

Key functionalities:
- Loads best-performing model from aggregated outer CV results
- Applies selected radiomic features
- Computes SHAP explanations for:
    • Outer CV dataset
    • Final hold-out test set
- Produces class-wise SHAP summary plots
- Exports feature metadata (image type & radiomic class) for reporting

Purpose:
- Provide explainability figures
- Support interpretation of radiomics-based classification models
- Generate supplementary material for reviewers

Note:
- It is used post-hoc explainability analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# Project modules
from src.load_data import load_and_clean
from src.split_and_check import split_and_check
from src.models import get_models_and_params

from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler
from neurocombat_sklearn import CombatModel

# ============================
# LABEL NAMES
# ============================

label_names = {
    0: "Adenocarcinoma (ADC)",
    1: "Squamous Cell Carcinoma (SCC)",
}

# ============================
# PATHS
# ============================

leaderboard_csv = "Results/Outer_cv_results/average_leaderboard.csv"
features_csv = "Results/Selected Features/selected_features_new.csv"
data_path = "Data/Radiomic_Features_All.xlsx"
holdout_idx_file = "Results/holdout_indices.npy"

output_dir = Path("results/shap_plots_final_new")
output_dir.mkdir(parents=True, exist_ok=True)

# ============================
# LOAD BEST MODEL
# ============================

df = pd.read_csv(leaderboard_csv)

best_row = df.sort_values("F1_mean", ascending=False).iloc[0]

best_fs = best_row["FS_method"]
best_clf = best_row["Classifier"]
best_k = best_row["Top_k"]

print("\n=== BEST MODEL (FROM AVERAGE LEADERBOARD) ===")
print(best_row)

# ============================
# LOAD DATA
# ============================

X, y, center = load_and_clean(data_path)

# ============================
# LOAD GLOBAL FEATURES
# ============================

df_features = pd.read_csv(features_csv)

features = df_features["feature_name"].tolist()

# ============================
# LOAD MODELS
# ============================

models, _ = get_models_and_params()

# ============================================================
# OUTER SHAP
# ============================================================

print("\n========== OUTER SHAP ==========")


_, best_folds, _, X_clean = split_and_check(
    X=X,
    y=y,
    centers=center,
    n_splits=3,
    random_state=42,
    n_trials=20,
    output_dir="data/split_and_check/split_report"
)

outer_cv = PredefinedSplit(test_fold=best_folds - 1)

all_train_idx = []
all_test_idx = []

for train_idx, test_idx in outer_cv.split():
    all_train_idx.extend(train_idx)
    all_test_idx.extend(test_idx)

all_train_idx = np.unique(all_train_idx)
all_test_idx = np.unique(all_test_idx)

X_train_outer = X_clean.iloc[all_train_idx]
X_test_outer = X_clean.iloc[all_test_idx]
y_train_outer = y[all_train_idx]

X_train_sel = X_train_outer[features]
X_test_sel = X_test_outer[features]

model_outer = models[best_clf]
model_outer.fit(X_train_sel, y_train_outer)

explainer = shap.Explainer(model_outer, X_train_sel)
shap_values = explainer(X_test_sel, check_additivity=False)

outer_dir = output_dir / "outer"
outer_dir.mkdir(exist_ok=True)

for c in range(shap_values.values.shape[2]):
    shap_class = shap_values.values[:, :, c]

    plt.figure()
    shap.summary_plot(
        shap_class,
        X_test_sel,
        feature_names=features,
        max_display=20,
        show=False
    )
    
    ax = plt.gca()
    ax.set_xlim(-0.15, 0.15)

    plt.title(f"OUTER SHAP - {label_names[c]}")

    plt.savefig(
        outer_dir / f"shap_outer_class{c}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ============================================================
# HOLD-OUT SHAP (FINAL)
# ============================================================

print("\n========== HOLD-OUT SHAP ==========")

holdout_idx = np.load(holdout_idx_file)
train_idx = np.setdiff1d(np.arange(len(X)), holdout_idx)

X_train_full = X.iloc[train_idx]
y_train_full = y[train_idx]
center_train = center.iloc[train_idx]

X_holdout = X.iloc[holdout_idx]
y_holdout = y[holdout_idx]
center_holdout = center.iloc[holdout_idx]

# COMBAT + SCALING
center_codes_train = pd.factorize(center_train)[0].reshape(-1, 1)
center_codes_holdout = pd.factorize(center_holdout)[0].reshape(-1, 1)

combat = CombatModel()
X_train_h = combat.fit_transform(X_train_full, center_codes_train)
X_holdout_h = combat.transform(X_holdout, center_codes_holdout)

scaler = StandardScaler()

X_train_s = pd.DataFrame(scaler.fit_transform(X_train_h), columns=X.columns)
X_holdout_s = pd.DataFrame(scaler.transform(X_holdout_h), columns=X.columns)

X_train_sel = X_train_s[features]
X_holdout_sel = X_holdout_s[features]

model_final = models[best_clf]
model_final.fit(X_train_sel, y_train_full)

explainer = shap.Explainer(model_final, X_train_sel)
shap_values = explainer(X_holdout_sel, check_additivity=False)

holdout_dir = output_dir / "holdout"
holdout_dir.mkdir(exist_ok=True)

for c in range(shap_values.values.shape[2]):
    shap_class = shap_values.values[:, :, c]

    plt.figure()
    shap.summary_plot(
        shap_class,
        X_holdout_sel,
        feature_names=features,
        max_display=20,
        show=False
    )

    ax = plt.gca()
    ax.set_xlim(-0.15, 0.15)
    
    plt.title(f"HOLD-OUT SHAP - {label_names[c]}")

    plt.savefig(
        holdout_dir / f"shap_holdout_class{c}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print("\n SHAP completed (outer + holdout).")

# ============================================================
# FEATURE CSV 
# ============================================================

def extract_image_type(f):
    if "wavelet" in f:
        return f.split("_")[0]
    elif "original" in f:
        return "original"
    return "other"

def extract_feature_class(f):
    for cls in ["glcm", "glrlm", "glszm", "gldm", "firstorder", "shape"]:
        if cls in f.lower():
            return cls.upper()
    return "OTHER"

records = []

for f in features:
    records.append({
        "dataset": "outer",
        "model": best_clf,
        "feature_name": f,
        "image_type": extract_image_type(f),
        "feature_class": extract_feature_class(f)
    })

for f in features:
    records.append({
        "dataset": "holdout",
        "model": best_clf,
        "feature_name": f,
        "image_type": extract_image_type(f),
        "feature_class": extract_feature_class(f)
    })

df_export = pd.DataFrame(records)

csv_path = output_dir / "feature_summary.csv"
df_export.to_csv(csv_path, index=False)

print(f"\n Feature CSV saved at: {csv_path}")
