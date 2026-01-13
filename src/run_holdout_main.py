# run_holdout_main.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from neurocombat_sklearn import CombatModel
from src.load_data import load_and_clean
from src.models import get_models_and_params
from src.explainability import run_explainability 

# ----------------------------
# Paths
# ----------------------------
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
outer_file = results_dir / "outer_cv_results.csv"
features_file = results_dir / "selected_features.csv"
data_file = Path("data/Radiomic_Features_All.xlsx")

# ----------------------------
# Load data
# ----------------------------
X, y, center = load_and_clean(data_file)

# ----------------------------
# Recompute hold-out split (same as main)
# ----------------------------
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

X_main = X.iloc[train_indices].copy()
y_main = y[train_indices]
X_holdout = X.iloc[holdout_indices].copy()
y_holdout = y[holdout_indices]
center_main = center.iloc[train_indices].copy()
center_holdout = center.iloc[holdout_indices].copy()

print(f"HOLD-OUT SAMPLES: {len(X_holdout)}")
print(f"REMAINING TRAINING SAMPLES: {len(X_main)}")

# ----------------------------
# Load top model info from main results
# ----------------------------
df_outer = pd.read_csv(outer_file)
top3_rows = df_outer.sort_values("Test_F1", ascending=False).head(3)
best_overall_row = top3_rows.iloc[0]

df_features = pd.read_csv(features_file)
feats_row = df_features[
    (df_features["FS_method"] == best_overall_row["FS_method"]) &
    (df_features["Classifier"] == best_overall_row["Classifier"]) &
    (df_features["Best_k"] == best_overall_row["Top_k"])
].iloc[0]
best_features = feats_row["Selected_Features"].split(',')

# ----------------------------
# ComBat harmonization and scaling
# ----------------------------
center_codes = pd.factorize(center_main)[0].reshape(-1, 1)
combat = CombatModel()
X_main_h = combat.fit_transform(X_main, center_codes)
X_holdout_h = combat.transform(X_holdout, pd.factorize(center_holdout)[0].reshape(-1, 1))

scaler = StandardScaler()
X_main_s = pd.DataFrame(scaler.fit_transform(X_main_h), columns=X.columns)
X_holdout_s = pd.DataFrame(scaler.transform(X_holdout_h), columns=X.columns)

X_main_sel = X_main_s[best_features]
X_holdout_sel = X_holdout_s[best_features]

# ----------------------------
# Load model
# ----------------------------
models, _ = get_models_and_params()
best_model = models[best_overall_row["Classifier"]]
best_model.fit(X_main_sel, y_main)

# ----------------------------
# Evaluate on hold-out set
# ----------------------------
y_holdout_pred = best_model.predict(X_holdout_sel)
f1_holdout = f1_score(y_holdout, y_holdout_pred, average="weighted")
acc_holdout = accuracy_score(y_holdout, y_holdout_pred)

try:
    if hasattr(best_model, "predict_proba"):
        y_holdout_bin = label_binarize(y_holdout, classes=np.unique(y_main))
        y_proba = best_model.predict_proba(X_holdout_sel)
        auc_holdout = roc_auc_score(y_holdout_bin, y_proba, average="weighted", multi_class="ovr")
    else:
        auc_holdout = np.nan
except:
    auc_holdout = np.nan

# Save hold-out results
holdout_results = pd.DataFrame([{
    "FS_method": best_overall_row["FS_method"],
    "Classifier": best_overall_row["Classifier"],
    "Top_k": best_overall_row["Top_k"],
    "F1_score": f1_holdout,
    "Accuracy": acc_holdout,
    "AUC": auc_holdout
}])
holdout_results.to_csv(results_dir / "holdout_results.csv", index=False)

print(f"HOLD-OUT RESULTS saved to {results_dir / 'holdout_results.csv'}")
print(f"F1: {f1_holdout:.3f}, Accuracy: {acc_holdout:.3f}, AUC: {auc_holdout:.3f}")

# ----------------------------
# Explainability
# ----------------------------
print("Running explainability analysis on HOLD-OUT set...")

try:
    explain_results = run_explainability(
        model=best_model,
        X_train=X_main_sel,
        X_test=X_holdout_sel,
        y_train=y_main,
        y_test=y_holdout,
        feature_names=X_holdout_sel.columns,
        output_dir=results_dir / "explainability_demo",
        fold_name="holdout"
    )
    print(" Explainability results saved in results/explainability/holdout")
    # Print main outputs
    for k in ["global_shap_csv", "top10_common_features", "waterfall_sample0"]:
        if k in explain_results:
            print(f"{k}: {explain_results[k]}")
except Exception as e:
    print(f" Explainability failed: {e}")
