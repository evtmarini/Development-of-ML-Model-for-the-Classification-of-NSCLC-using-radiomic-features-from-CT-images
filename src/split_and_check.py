import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
import os


def split_and_check(X, y, centers=None, n_splits=3, random_state=42, n_trials=20, output_dir="data/split_and_check/split_report"):
    """
    Find the most balanced stratified folds and optionally balance across centers.
    Use 'center' info only for diagnostics (not in modeling), and remove any bias columns from X.
    Returns: best_splits, best_folds, best_report, X_clean
    """

    os.makedirs(output_dir, exist_ok=True)
    img_dir = os.path.join(output_dir, "folds_images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"\n Creating stratified folds ({n_splits}-fold CV)")

    
    #  Detect and handle 'center' column
    bias_keywords = ["center"]
    detected_centers = [c for c in X.columns if any(k in c.lower() for k in bias_keywords)]

    if centers is None and detected_centers:
        centers = X[detected_centers[0]]
        print(f"Using '{detected_centers[0]}' column for fold composition visualization.")
    elif centers is not None:
        print(" External 'centers' vector provided for balancing.")
    else:
        print("No center information provided — using class-based stratification only.")

    # Drop any bias-related columns to avoid confounding effects
    if detected_centers:
        X = X.drop(columns=detected_centers)
        print(f" Dropped bias columns from X: {detected_centers}")


    #  Compute multiple stratified splits to find the most balanced one
    y = np.array(y)
    best_std, best_seed, best_splits, best_folds, best_report = np.inf, None, None, None, None

    for trial in range(n_trials):
        seed = random_state + trial
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_assignments = np.zeros(len(y), dtype=int)
        splits = []

        for fold_idx, (_, val_idx) in enumerate(cv.split(X, y)):
            fold_assignments[val_idx] = fold_idx + 1
            splits.append(val_idx)

        df_summary = pd.DataFrame({"Fold": fold_assignments, "Label": y})
        label_dist = pd.crosstab(df_summary["Fold"], df_summary["Label"], normalize="index") * 100
        label_std = label_dist.std(axis=0).mean()

        if label_std < best_std:
            best_std, best_seed, best_splits, best_folds = label_std, seed, splits, fold_assignments
            best_report = {"mean_label_std": label_std, "best_seed": seed}

    print(f"Best stratified split found at seed={best_seed} | mean_label_std = {best_std:.2f}%")



    #  Verify uniqueness
    all_indices = np.concatenate(best_splits)
    if len(all_indices) == len(np.unique(all_indices)):
        print("Verified: all samples are unique across folds.")
    else:
        print("Warning: overlap detected between folds!")

   
    # Visualization (fold and center composition)
    df_summary = pd.DataFrame({"Fold": best_folds, "Label": y})
    if centers is not None:
        df_summary["Center"] = np.array(centers)
    else:
        df_summary["Center"] = "N/A"


    # =====================================================================
    # PRINT FOLD REPORTS (LABELS, CENTERS, LABELS×CENTERS)
    # =====================================================================
    print("\n==================== FOLD SUMMARY TABLES ====================")

    # Samples per Label per Fold
    label_table = pd.crosstab(df_summary["Fold"], df_summary["Label"])
    print("\n Samples per fold per label:")
    print(label_table)

    # Samples per Center per Fold
    center_table = pd.crosstab(df_summary["Fold"], df_summary["Center"])
    print("\n Samples per fold per center:")
    print(center_table)

    # Samples per Label × Center × Fold
    if centers is not None and len(np.unique(centers)) > 1:
        multi_table = pd.crosstab([df_summary["Fold"], df_summary["Center"]], df_summary["Label"])
        print("\n Samples per fold per center per label:")
        print(multi_table)

    print("============================================================\n")
    # =====================================================================



    # === Fold composition by class ===
    label_counts = pd.crosstab(df_summary["Fold"], df_summary["Label"])
    label_counts.plot(kind="bar", figsize=(8, 5), colormap="viridis", edgecolor="black")
    plt.title("Fold Composition — Samples per Class")
    plt.xlabel("Fold")
    plt.ylabel("Number of Samples")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "fold_composition_classes_bar.png"), dpi=300)
    plt.close()

    # === If centers exist, visualize distribution ===
    if centers is not None and len(np.unique(centers)) > 1:
        center_counts = pd.crosstab(df_summary["Fold"], df_summary["Center"])
        center_counts.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="Set2", edgecolor="black")
        plt.title("Fold Composition — Samples per Center (Stacked)")
        plt.xlabel("Fold")
        plt.ylabel("Samples")
        plt.legend(title="Center", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, "fold_composition_centers_stacked.png"), dpi=300)
        plt.close()

        # Compute variability
        center_dist = pd.crosstab(df_summary["Fold"], df_summary["Center"], normalize="index") * 100
        mean_center_std = center_dist.std(axis=0).mean()
        best_report["mean_center_std"] = mean_center_std
        print(f" mean_center_std = {mean_center_std:.2f}% (center distribution variability)")

    # === Simple PCA visualization ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Fold": df_summary["Fold"].astype(str)})

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Fold", palette="tab10", alpha=0.7, s=40)
    plt.title("PCA Projection — Fold Separation Overview")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "pca_folds_simple.png"), dpi=300)
    plt.close()

    print(" Saved fold visualizations successfully.")

    
    # Return split info and cleaned X (no center columns)
    return best_splits, best_folds, best_report, X
