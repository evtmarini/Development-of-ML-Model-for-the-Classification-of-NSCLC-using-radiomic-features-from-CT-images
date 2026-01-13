#==============================================================
# Module: stepwise_fs.py
# Description: Leak-free Step-wise feature selection evaluation
#              for multiple models using radiomic features.
#              Uses inner CV on training data only.
#==============================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np

def run_stepwise_fs(
    X_train_final, y_train,
    fs_per_fold, models, fold_idx, step=10,
    cv_folds=3, output_dir="results/stepwise"
):
    """
    Leak-free stepwise feature selection using inner CV only.
    
    Parameters
    ----------
    X_train_final : pd.DataFrame
        Preprocessed training data
    y_train : array-like
        Training labels
    fs_per_fold : dict
        Dictionary with feature selection results per fold
    models : dict
        Dictionary of ML models
    fold_idx : int
        Current outer fold index
    step : int
        Step size for number of features
    cv_folds : int
        Number of folds for inner CV
    output_dir : str
        Directory to save plots and results

    Returns
    -------
    results : pd.DataFrame
        Step-wise metrics for all models and FS methods
    peak_points : pd.DataFrame
        Peak mean F1-score points per model and FS method
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    inner_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for fs_name, feats_ranked in fs_per_fold[fold_idx].items():

        max_feats = min(len(feats_ranked), 300)  # περιορισμός για ταχύτητα
        n_steps = list(range(step, max_feats + 1, step))

        metrics_df = pd.DataFrame(columns=["n_features", "model", "mean_F1", "Accuracy", "Precision", "Recall"])

        for k in n_steps:
            top_feats = feats_ranked[:k]
            X_train_k = X_train_final[top_feats]

            for model_name, model in models.items():
                f1_scores = []
                acc_scores = []
                prec_scores = []
                rec_scores = []

                # Inner CV
                for train_idx, val_idx in inner_cv.split(X_train_k, y_train):
                    X_tr, X_val = X_train_k.iloc[train_idx], X_train_k.iloc[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]

                    # --- Fix για k-NN ---
                    if isinstance(model, KNeighborsClassifier):
                        n_train_samples = X_tr.shape[0]
                        if model.n_neighbors > n_train_samples:
                            model.set_params(n_neighbors=n_train_samples)

                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)

                    f1_scores.append(f1_score(y_val, y_pred, average="weighted"))
                    acc_scores.append(accuracy_score(y_val, y_pred))
                    prec_scores.append(precision_score(y_val, y_pred, average="weighted"))
                    rec_scores.append(recall_score(y_val, y_pred, average="weighted"))

                mean_f1 = np.mean(f1_scores)
                mean_acc = np.mean(acc_scores)
                mean_prec = np.mean(prec_scores)
                mean_rec = np.mean(rec_scores)

                metrics_df.loc[len(metrics_df)] = [k, model_name, mean_f1, mean_acc, mean_prec, mean_rec]

                all_results.append({
                    "outer_fold": fold_idx,
                    "FS_method": fs_name,
                    "model": model_name,
                    "n_features": k,
                    "mean_F1": mean_f1,
                    "Accuracy": mean_acc,
                    "Precision": mean_prec,
                    "Recall": mean_rec
                })

        # Plot mean F1 vs n_features for each model
        plt.figure(figsize=(10, 6))
        for model_name in metrics_df["model"].unique():
            sub = metrics_df[metrics_df["model"] == model_name]
            plt.plot(sub["n_features"], sub["mean_F1"], marker='o', label=model_name)
        plt.title(f"Step-wise FS | Fold {fold_idx} | FS={fs_name}")
        plt.xlabel("Number of Features")
        plt.ylabel("Mean F1 Score (weighted, inner CV)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"Fold{fold_idx}_{fs_name}_stepwise.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()

    # --------------------------------------------------------
    # Peak points: n_features με μέγιστο mean F1 ανά model & FS
    # --------------------------------------------------------
    df_results = pd.DataFrame(all_results)
    peak_points = df_results.loc[df_results.groupby(["FS_method", "model"])["mean_F1"].idxmax()].reset_index(drop=True)

    # Save results
    df_results.to_csv(os.path.join(output_dir, f"Fold{fold_idx}_stepwise_results_leakfree.csv"), index=False)
    peak_points.to_csv(os.path.join(output_dir, f"Fold{fold_idx}_peak_points_leakfree.csv"), index=False)

    print(f"\nLeak-free Step-wise FS completed for Fold {fold_idx}.")
    print(f"Results saved to {output_dir}")
    print("Peak points per model and FS method saved.")

    return df_results, peak_points
