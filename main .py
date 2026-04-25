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


# LOG SETUP

Path("results").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename="results/main_run.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # print also to console


# 0. HEADER

logger.info("Development of ML Model for NSCLC Classification using Radiomic Features")
logger.info("Loading dataset...")


# 1. LOAD DATA

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


# 1b. LABEL + CENTER AWARE HOLD-OUT (10%)

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


# SAVE FIXED HOLD-OUT INDICES (for reproducibility & XAI)

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


# 2. BALANCED FOLDS (predefined outer CV)

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


# 3. OUTER CV

outer_folds = best_folds - 1
outer_cv = PredefinedSplit(test_fold=outer_folds)
print("\nOuter CV folds:", outer_cv.get_n_splits())


# 4. LOAD MODELS + PARAMETERS

models, param_grids = get_models_and_params()
center_codes = pd.factorize(center_main)[0].reshape(-1, 1)


# 5. OUTER LOOP

print("\nStarting Outer Cross-Validation Loop:\n")
unique_folds = sorted(set(best_folds))

inner_file = "results/inner_cv_results.csv"
outer_file = "results/outer_cv_results.csv"
features_file = "results/selected_features.csv"

stepwise_range = range(10, 101, 20)  # top-k features

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

    # ComBat harmonization
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

    # Scaling
    try:
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train_h), columns=X_clean.columns)
        X_test_s = pd.DataFrame(scaler.transform(X_test_h), columns=X_clean.columns)
    except Exception as e:
        logger.warning(f"Scaling failed: {e}")
        X_train_s, X_test_s = X_train_h.copy(), X_test_h.copy()

    # Preprocessing filters
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

    # Feature Selection
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

                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    inner_results = run_experiments(
                        selected_datasets={f"Fold{fold_idx}_{fs_name}_top{k}": (X_train_sel, y_train)},
                        models=models,
                        param_grids=param_grids,
                        cv=inner_cv
                    )

                    # Compute Inner AUC
                    try:
                        inner_auc_list = []
                        for model_name, (X_inner, y_inner) in inner_results.items():
                            mdl = models[model_name]
                            mdl.fit(X_inner, y_inner)
                            if hasattr(mdl, "predict_proba"):
                                y_bin = label_binarize(y_inner, classes=np.unique(y_main))
                                y_proba = mdl.predict_proba(X_inner)
                                inner_auc_list.append(roc_auc_score(y_bin, y_proba, average="weighted", multi_class="ovr"))
                            else:
                                inner_auc_list.append(np.nan)
                        inner_results["Inner_AUC"] = np.mean(inner_auc_list)
                    except Exception as e:
                        logger.warning(f"Inner AUC computation failed: {e}")
                        inner_results["Inner_AUC"] = np.nan

                    inner_results["Outer_Fold"] = fold_idx
                    inner_results["FS_method"] = fs_name
                    inner_results["Top_k"] = k
                    pd.DataFrame(inner_results).to_csv(inner_file, mode="a", index=False, header=not Path(inner_file).exists())

                    # Train & test best model
                    try:
                        best_row = inner_results.sort_values("F1_score", ascending=False).iloc[0]
                        best_model_name = best_row["Classifier"]
                        best_model = models[best_model_name]
                        best_model.fit(X_train_sel, y_train)
                        y_pred = best_model.predict(X_test_sel)
                        f1 = f1_score(y_test, y_pred, average="weighted")
                        acc = accuracy_score(y_test, y_pred)
                        try:
                            if hasattr(best_model, "predict_proba"):
                                y_test_bin = label_binarize(y_test, classes=np.unique(y_main))
                                y_proba = best_model.predict_proba(X_test_sel)
                                auc = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
                            else:
                                auc = np.nan
                        except:
                            auc = np.nan

                        outer_row = {
                            "Outer_Fold": fold_idx,
                            "FS_method": fs_name,
                            "Top_k": k,
                            "Classifier": best_model_name,
                            "Inner_F1": best_row["F1_score"],
                            "Test_F1": f1,
                            "Test_Accuracy": acc,
                            "Test_AUC": auc
                        }
                        pd.DataFrame([outer_row]).to_csv(outer_file, mode="a", index=False, header=not Path(outer_file).exists())

                        features_row = {
                            "Outer_Fold": fold_idx,
                            "FS_method": fs_name,
                            "Classifier": best_model_name,
                            "Best_k": k,
                            "Selected_Features": ",".join(feats)
                        }
                        pd.DataFrame([features_row]).to_csv(features_file, mode="a", index=False, header=not Path(features_file).exists())
                    except Exception as e:
                        logger.error(f"Training/testing failed for {fs_name} top-{k}: {e}")
                else:
                    print(f"{fs_name} (top-{k}): skipped (too few features).")
            except Exception as e:
                logger.error(f"{fs_name} top-{k} FAILED: {e}")

print(" All results are being saved in real-time to CSV files.")


# 6. ANALYSIS: Top-3 models feature classes + CSV

try:
    print("\nAnalyzing feature classes for top-3 models...")
    df_outer = pd.read_csv(outer_file)
    top3_rows = df_outer.sort_values("Test_F1", ascending=False).head(3)
    print("\nTop-3 models based on Test F1:")
    print(top3_rows[["FS_method", "Classifier", "Top_k", "Test_F1", "Test_AUC"]])
    top3_rows[["FS_method", "Classifier", "Top_k", "Test_F1", "Test_AUC"]].to_csv("results/top3_models.csv", index=False)

    df_features = pd.read_csv(features_file)
    top3_features = []
    for idx, row in top3_rows.iterrows():
        feats_row = df_features[
            (df_features["Outer_Fold"] == row["Outer_Fold"]) &
            (df_features["FS_method"] == row["FS_method"]) &
            (df_features["Classifier"] == row["Classifier"]) &
            (df_features["Best_k"] == row["Top_k"])
        ]
        if not feats_row.empty:
            top3_features.extend(feats_row.iloc[0]["Selected_Features"].split(','))
    top3_features = list(set(top3_features))
    print(f"\nTotal unique features from top-3 models: {len(top3_features)}")

    feature_classes = [f.split('_')[0] for f in top3_features]
    class_counts = Counter(feature_classes)
    print("\nFeature classes distribution in top-3 models:")
    for cls, count in class_counts.most_common():
        print(f"{cls}: {count} features")
    df_classes = pd.DataFrame(class_counts.items(), columns=["Feature_Class", "Count"])
    df_classes.to_csv("results/top3_features_classes.csv", index=False)
    print("\n Top-3 models and feature classes saved to CSV in results/")
except Exception as e:
    logger.error(f"Top-3 analysis failed: {e}")


# 6b. PLOTS

try:
    # 1. Bar plot: Test_F1 για top-3 models
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top3_rows, x="Classifier", y="Test_F1", hue="FS_method", palette="Set2")
    plt.title("Top-3 Models Test F1 Score")
    plt.ylabel("Test F1")
    plt.xlabel("Classifier")
    plt.ylim(0, 1)
    plt.legend(title="FS method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("results/top3_models_f1.png", dpi=300)
    plt.show()

    # 2. Bar plot: Test_AUC για top-3 models
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top3_rows, x="Classifier", y="Test_AUC", hue="FS_method", palette="Set2")
    plt.title("Top-3 Models Test AUC")
    plt.ylabel("Test AUC")
    plt.xlabel("Classifier")
    plt.ylim(0, 1)
    plt.legend(title="FS method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("results/top3_models_auc.png", dpi=300)
    plt.show()

    # 3. Bar plot: Feature classes count
    df_classes_sorted = df_classes.sort_values("Count", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_classes_sorted, x="Feature_Class", y="Count", palette="coolwarm")
    plt.title("Feature Classes Distribution in Top-3 Models")
    plt.ylabel("Number of Features")
    plt.xlabel("Feature Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/top3_feature_classes.png", dpi=300)
    plt.show()

    # 4. Heatmap
    plt.figure(figsize=(12, 6))
    pivot_f1 = df_outer.pivot_table(index="FS_method", columns="Classifier", values="Test_F1")
    sns.heatmap(pivot_f1, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Test F1 Score Heatmap (Outer CV)")
    plt.tight_layout()
    plt.savefig("results/heatmap_test_f1.png", dpi=300)
    plt.show()

    # 5. Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="FS_method", y="Test_F1", data=df_outer, palette="Set3")
    sns.stripplot(x="FS_method", y="Test_F1", data=df_outer, color="black", size=3, jitter=True)
    plt.title("Distribution of Test F1 per Feature Selection Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/boxplot_fs_test_f1.png", dpi=300)
    plt.show()
except Exception as e:
    logger.warning(f"Plots failed: {e}")


