# ============================================================
# evaluation.py 
# ============================================================

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import StratifiedKFold, HalvingRandomSearchCV, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, make_scorer, accuracy_score,
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import StackingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.switch_backend("Agg")


def run_experiments(
    selected_datasets,
    models=None,
    param_grids=None,
    cv=None,
    scoring=None,
    random_state=42
):
    if cv is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    results = []
    os.makedirs("data/confusion_matrices", exist_ok=True)

    if scoring is None:
        scoring = make_scorer(f1_score, average="weighted")

    for fs_name, data_pair in selected_datasets.items():

        try:
            X_sel, y_local = data_pair
        except Exception:
            print(f"Invalid dataset entry for {fs_name}. Expected (X, y).")
            continue

        n_features = X_sel.shape[1]
        print(f"\n[INFO] Running Halving Search for feature set: {fs_name} ({n_features} features)")

        # Αν cv είναι int, create StratifiedKFold
        if isinstance(cv, int):
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            skf = cv

        # --------------------------------------------------------
        # Balancing με ADASYN
        # --------------------------------------------------------
        print("Balancing training data with ADASYN...")
        ada = ADASYN(sampling_strategy="auto", random_state=random_state, n_neighbors=3)
        X_bal, y_bal = ada.fit_resample(X_sel, y_local)
        print(f"Balanced dataset: {len(y_local)} → {len(y_bal)} samples")

        # Safety check 
        if len(np.unique(y_bal)) < 2:
            print(f"Skipping {fs_name} — only one class present after balancing.")
            continue

        # Περιορισμός n_splits ώστε να μην ξεπερνά τον αριθμό της μικρότερης κλάσης
        min_class_size = np.min(np.bincount(y_bal))
        n_splits_safe = min(skf.get_n_splits(), min_class_size)
        if n_splits_safe < 2:
            n_splits_safe = 2  # πάντα τουλάχιστον 2 splits
        skf_safe = StratifiedKFold(n_splits=n_splits_safe, shuffle=True, random_state=random_state)
        print(f"Using {n_splits_safe} splits for inner CV (safe)")

        # --------------------------------------------------------
        # Εκτέλεση για κάθε μοντέλο
        # --------------------------------------------------------
        for model_name, clf in models.items():
            print(f"\nEvaluating model: {model_name}")

            estimator = clf

            # Check αν είναι StackingClassifier
            if isinstance(estimator, StackingClassifier):
                estimator.cv = n_splits_safe

            search = HalvingRandomSearchCV(
                estimator=estimator,
                param_distributions=param_grids.get(model_name, {}),
                scoring=scoring,
                cv=skf_safe,
                factor=4,
                min_resources='smallest',
                random_state=random_state,
                n_jobs=-1,
                verbose=1,
                error_score="raise"
            )

            try:
                le_global = LabelEncoder()
                y_bal_enc = le_global.fit_transform(y_bal)

                search.fit(X_bal, y_bal_enc)

                y_pred = cross_val_predict(
                    search.best_estimator_,
                    X_bal, y_bal_enc,
                    cv=skf_safe, n_jobs=-1
                )
                y_true = y_bal_enc

            except ValueError as e:

                # fallback για προβλήματα με labels (XGBoost/LightGBM)
                if "Invalid classes inferred" in str(e):
                    print(f"Fixing label issue for {model_name}...")

                    preds_all, true_all = [], []

                    for train_idx, val_idx in skf_safe.split(X_bal, y_bal):

                        X_tr = X_bal.iloc[train_idx]
                        X_val = X_bal.iloc[val_idx]

                        y_tr = y_bal[train_idx]
                        y_val = y_bal[val_idx]

                        le_fold = LabelEncoder()
                        y_tr_enc = le_fold.fit_transform(y_tr)

                        valid_mask = np.isin(y_val, le_fold.classes_)
                        y_val_enc = le_fold.transform(y_val[valid_mask])

                        clf_safe = search.estimator
                        clf_safe.fit(X_tr, y_tr_enc)
                        preds = clf_safe.predict(X_val[valid_mask])

                        preds_all.extend(preds)
                        true_all.extend(y_val_enc)

                    y_true = np.array(true_all)
                    y_pred = np.array(preds_all)

                else:
                    raise e

            # --------------------------------------------------------
            # Metrics
            # --------------------------------------------------------
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average="weighted")
            rec = recall_score(y_true, y_pred, average="weighted")
            f1 = f1_score(y_true, y_pred, average="weighted")
            cm = confusion_matrix(y_true, y_pred)

            # Compute AUC for multiclass (if available)
            try:
                y_true_bin = pd.get_dummies(y_true).values
                if hasattr(search.best_estimator_, "predict_proba"):
                    y_pred_prob = search.best_estimator_.predict_proba(X_bal)
                    auc = roc_auc_score(y_true_bin, y_pred_prob, average="weighted", multi_class="ovr")
                else:
                    auc = np.nan
            except Exception:
                auc = np.nan

            print(f"CV Metrics → Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

            # Αποθήκευση Confusion Matrix
            cm_path = f"data/confusion_matrices/cm_{fs_name}_{model_name}.png"
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.title(f"Confusion Matrix — {fs_name} | {model_name}")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=300)
            plt.close()
            print(f"Saved confusion matrix → {cm_path}")

            best_params_safe = getattr(search, "best_params_", {})

            results.append({
                "FS_method": fs_name,
                "Classifier": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_score": f1,
                "AUC": auc,
                "Best_params": best_params_safe
            })

    # --------------------------------------------------------
    # Save results (append if CSV exists)
    # --------------------------------------------------------
    results_df = pd.DataFrame(results)
    csv_path = "data/halving_results.csv"

    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        results_df = pd.concat([old_df, results_df], ignore_index=True)

    results_df.to_csv(csv_path, index=False)

    # --------------------------------------------------------
    # Plot results
    # --------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="F1_score", y="Classifier",
        data=results_df, hue="FS_method",
        dodge=True
    )
    plt.xlabel("Weighted F1-score")
    plt.title("Halving Random Search — Cross-Fold Results")
    plt.tight_layout()
    plt.savefig("data/halving_results.png", dpi=300)
    plt.close()

    print(f"Results saved to {csv_path}")
    return results_df
