"""
Main Pipeline: Radiomics-Based ML for Lung Cancer Subtype Classification

This script implements the full end-to-end machine learning pipeline, including:

- Data loading and cleaning
- Center-aware hold-out split (fixed, reproducible)
- Balanced outer cross-validation (PredefinedSplit)
- ComBat harmonization and feature preprocessing
- Feature selection (LASSO, Boruta, mRMR, RFE-SVM, ReliefF, RF importance)
- Inner cross-validation with hyperparameter tuning
- Outer cross-validation for unbiased performance estimation
- Automatic selection of best models per fold

Notes:
- The hold-out set is strictly separated and used only for final evaluation
- All preprocessing steps (ComBat, scaling, feature selection) are applied within each fold
  to prevent data leakage
- Results are saved incrementally for reproducibility and traceability
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Project modules
from src.load_data import load_and_clean
from src.split_and_check import split_and_check
from src.preprocessing import variance_filter, correlation_filter, stat_filter
from src.feature_selection import (
    fs_lasso,
    fs_rf_importance,
    fs_boruta,
    fs_rfe_svm,
    fs_mrmr,
    fs_relieff
)
from src.models import get_models_and_params
from src.evaluation import run_experiments
from src.explainability import run_explainability

# sklearn
from sklearn.model_selection import PredefinedSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# ComBat
from neurocombat_sklearn import CombatModel

# ============================================================
# LOG SETUP
# ============================================================
Path("results").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename="results/main_run.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # print also to console

# ============================================================
# 0. HEADER
# ============================================================
logger.info("Development of ML Model for NSCLC Classification using Radiomic Features")
logger.info("Loading dataset...")

# ============================================================
# 1. LOAD DATA
# ============================================================
base = Path("data")
path = base / "Radiomic_Features_All.xlsx"

try:
    X, y, center = load_and_clean(path)
    print("\nLOAD REPORT")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print("Class distribution:")
    print(pd.Series(y).value_counts())
    print("\nCenter distribution:")
    print(pd.Series(center).value_counts())
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

# ============================================================
# 1b. LABEL + CENTER AWARE HOLD-OUT (10%)
# ============================================================
np.random.seed(42)
df_meta = pd.DataFrame({"label": y, "center": center})

holdout_indices = []
for (lbl, ctr), group in df_meta.groupby(["label", "center"]):
    n_total = len(group)
    n_hold = max(1, int(round(0.10 * n_total))) if n_total >= 5 else 0
    if n_hold > 0:
        selected = np.random.choice(group.index, size=n_hold, replace=False)
        holdout_indices.extend(selected)

holdout_indices = np.array(sorted(holdout_indices))
train_indices = np.setdiff1d(np.arange(len(y)), holdout_indices)

# ============================================================
# SAVE FIXED HOLD-OUT INDICES (for reproducibility & XAI)
# ============================================================
holdout_indices_path = Path("results/holdout_indices.npy")
np.save(holdout_indices_path, holdout_indices)

logger.info(f"Fixed hold-out indices saved to {holdout_indices_path}")

X_main = X.iloc[train_indices].copy()
y_main = y[train_indices]
center_main = center.iloc[train_indices].copy()

X_holdout = X.iloc[holdout_indices].copy()
y_holdout = y[holdout_indices]
center_holdout = center.iloc[holdout_indices].copy()

print(f"HOLD-OUT SAMPLES: {len(X_holdout)}")

print("\nHold-out label distribution:")
print(pd.Series(y_holdout).value_counts())

print("\nHold-out center distribution:")
print(pd.Series(center_holdout).value_counts())


print(f"REMAINING TRAINING SAMPLES: {len(X_main)}")

# ============================================================
# 2. BALANCED FOLDS (predefined outer CV)
# ============================================================
print("\nRunning split_and_check:\n")
try:
    best_splits, best_folds, split_report, X_clean = split_and_check(
        X=X_main,
        y=y_main,
        centers=center_main,
        n_splits=3,
        random_state=42,
        n_trials=20,
        output_dir="data/split_and_check/split_report"
    )
    print("Best seed:", split_report["best_seed"])
    print("mean_label_std:", split_report["mean_label_std"])
    if "mean_center_std" in split_report:
        print("mean_center_std:", split_report["mean_center_std"])
except Exception as e:
    logger.error(f"split_and_check failed: {e}")
    X_clean = X_main.copy()
    best_folds = np.zeros(len(X_main))

# ============================================================
# 3. PREDEFINED OUTER CV
# ============================================================
outer_folds = best_folds - 1
outer_cv = PredefinedSplit(test_fold=outer_folds)
print("\nOuter CV folds:", outer_cv.get_n_splits())

# ============================================================
# 4. LOAD MODELS + PARAMETERS
# ============================================================
models, param_grids = get_models_and_params()
center_codes = pd.factorize(center_main)[0].reshape(-1, 1)


# ============================================================
# 5. OUTER LOOP
# ============================================================
print("\nStarting Outer Cross-Validation Loop:\n")

inner_file = "results/inner_cv_results_new.csv"
outer_file = "results/outer_cv_results_new.csv"
features_file = "results/selected_features_new.csv"

stepwise_range = range(10, 101, 20)

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(), start=1):
    print(f"\n===== OUTER FOLD {fold_idx} =====")

    try:
        X_train = X_clean.iloc[train_idx].copy()
        X_test = X_clean.iloc[test_idx].copy()
        y_train = y_main[train_idx]
        y_test = y_main[test_idx]
        centers_train = center_codes[train_idx]
        centers_test = center_codes[test_idx]
    except Exception as e:
        logger.error(f"Data split failed for fold {fold_idx}: {e}")
        continue

    # --------------------------------------------------------
    # ComBat
    # --------------------------------------------------------
    try:
        print("Applying ComBat harmonization:")
        combat = CombatModel()
        X_train_h = combat.fit_transform(X_train, centers_train)
        X_test_h = combat.transform(X_test, centers_test)
        X_train_h = pd.DataFrame(X_train_h, columns=X_clean.columns)
        X_test_h = pd.DataFrame(X_test_h, columns=X_clean.columns)
    except Exception as e:
        logger.warning(f"ComBat failed: {e}")
        X_train_h, X_test_h = X_train.copy(), X_test.copy()

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------
    try:
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train_h), columns=X_clean.columns)
        X_test_s = pd.DataFrame(scaler.transform(X_test_h), columns=X_clean.columns)
    except Exception as e:
        logger.warning(f"Scaling failed: {e}")
        X_train_s, X_test_s = X_train_h.copy(), X_test_h.copy()

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------
    try:
        X_train_v = variance_filter(X_train_s, threshold=0.01)
        X_test_v = X_test_s[X_train_v.columns]

        X_train_c = correlation_filter(X_train_v, threshold=0.85)
        X_test_c = X_test_v[X_train_c.columns]

        X_train_final = stat_filter(X_train_c, y_train, alpha=0.1)
        X_test_final = X_test_c[X_train_final.columns]
    except Exception as e:
        logger.warning(f"Preprocessing filters failed: {e}")
        X_train_final, X_test_final = X_train_s.copy(), X_test_s.copy()

    # --------------------------------------------------------
    # Feature Selection
    # --------------------------------------------------------
    fs_registry = {
        "LASSO": lambda X, y, k: fs_lasso(X, y)[:k],
        "RF_importance": lambda X, y, k: fs_rf_importance(X, y, top_k=k),
        "Boruta": lambda X, y, k: fs_boruta(X, y)[:k],
        "RFE_SVM": lambda X, y, k: fs_rfe_svm(X, y, n_features=k),
        "mRMR": lambda X, y, k: fs_mrmr(X, y, top_k=k),
        "ReliefF": lambda X, y, k: fs_relieff(X, y, top_k=k),
    }

    for fs_name, fs_func in fs_registry.items():
        for k in stepwise_range:

            print(f"Running {fs_name} with top-{k} features")

            try:
                feats = fs_func(X_train_final, y_train, k)

                if feats and len(feats) >= 5:

                    print(f"{fs_name} (top-{k}): {len(feats)} features selected.")

                    X_train_sel = X_train_final[feats]
                    X_test_sel = X_test_final[feats]

                    # --------------------------------------------------------
                    # INNER CV
                    # --------------------------------------------------------
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

                    inner_results = run_experiments(
                        selected_datasets={f"Fold{fold_idx}_{fs_name}_top{k}": (X_train_sel, y_train)},
                        models=models,
                        param_grids=param_grids,
                        cv=inner_cv
                    )

                    # --------------------------------------------------------
                    # SAVE INNER
                    # --------------------------------------------------------
                    inner_results["Outer_Fold"] = fold_idx
                    inner_results["FS_method"] = fs_name
                    inner_results["Top_k"] = k

                    pd.DataFrame(inner_results).to_csv(
                        inner_file,
                        mode="a",
                        index=False,
                        header=not Path(inner_file).exists()
                    )

                    # ========================================================
                    # SAVE ALL MODELS 
                    # ========================================================
                    for _, row in inner_results.iterrows():

                        model_name = row["Classifier"]
                        model = models[model_name]

                        model.fit(X_train_sel, y_train)
                        y_pred = model.predict(X_test_sel)

                        f1 = f1_score(y_test, y_pred, average="weighted")
                        acc = accuracy_score(y_test, y_pred)

                        try:
                            if hasattr(model, "predict_proba"):
                                y_test_bin = label_binarize(y_test, classes=np.unique(y_main))
                                y_proba = model.predict_proba(X_test_sel)
                                auc = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
                            else:
                                auc = np.nan
                        except:
                            auc = np.nan

                        outer_row = {
                            "Outer_Fold": fold_idx,
                            "FS_method": fs_name,
                            "Top_k": k,
                            "Classifier": model_name,
                            "Inner_F1": row["F1_score"],
                            "Test_F1": f1,
                            "Test_Accuracy": acc,
                            "Test_AUC": auc
                        }

                        pd.DataFrame([outer_row]).to_csv(
                            outer_file,
                            mode="a",
                            index=False,
                            header=not Path(outer_file).exists()
                        )

                        features_row = {
                            "Outer_Fold": fold_idx,
                            "FS_method": fs_name,
                            "Classifier": model_name,
                            "Best_k": k,
                            "Selected_Features": ",".join(feats)
                        }

                        pd.DataFrame([features_row]).to_csv(
                            features_file,
                            mode="a",
                            index=False,
                            header=not Path(features_file).exists()
                        )

                else:
                    print(f"{fs_name} (top-{k}): skipped (too few features).")

            except Exception as e:
                logger.error(f"{fs_name} top-{k} FAILED: {e}")

print(" All results are being saved in real-time to CSV files.")

