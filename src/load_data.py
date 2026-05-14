"""
NSCLC Radiomics Data Loading & Initial Preprocessing

- Load dataset from Excel
- Inspect label and center distributions
- Remove small classes
- Encode labels
- Handle missing values

Returns:
    X: feature matrix
    y: encoded labels
    center: center information
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os


def load_and_clean(
    path,
    id_col="case_id",
    target_col="label",
    center_col="center",
    min_class_size=100,
    plot_dir="plots/initial data"
):

    # ============================================================
    # CREATE PLOT DIRECTORY
    # ============================================================
    os.makedirs(plot_dir, exist_ok=True)

    # ============================================================
    # LOAD DATASET
    # ============================================================
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"Error reading Excel file: {e}")

    print(f"\nLoaded file: {os.path.basename(path)} | Shape: {df.shape}")

    # ============================================================
    # EXTRACT CENTER COLUMN
    # ============================================================
    center = df[center_col].astype(str) if center_col in df.columns else None

    if center is not None:
        print(f"Detected center column with {center.nunique()} unique centers.")
    else:
        print("No center column found.")

    # ============================================================
    # INITIAL LABEL DISTRIBUTION
    # ============================================================
    print("\nInitial label distribution:")
    print(df[target_col].value_counts())

    df[target_col].value_counts().plot(kind="bar")
    plt.title("Initial Label Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/initial_label_distribution.png")
    plt.close()

    # ============================================================
    # INITIAL CENTER DISTRIBUTION
    # ============================================================
    if center is not None:
        print("\nInitial center distribution:")
        print(center.value_counts())

        center.value_counts().plot(kind="bar")
        plt.title("Initial Center Distribution")
        plt.xlabel("Center")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/initial_center_distribution.png")
        plt.close()

    # ============================================================
    # SEPARATE TARGET & FEATURES
    # ============================================================
    y = df[target_col].copy()

    X = df.drop(
        columns=[c for c in [id_col, target_col, center_col] if c in df.columns]
    )

    X = X.select_dtypes(include=[np.number])

    # ============================================================
    # REMOVE SMALL CLASSES
    # ============================================================
    unique, counts = np.unique(y, return_counts=True)

    small_classes = [
        cls for cls, count in zip(unique, counts)
        if count < min_class_size
    ]

    if small_classes:

        print(f"\nRemoving small classes (<{min_class_size} samples): {small_classes}")

        initial_n = len(y)

        mask = ~y.isin(small_classes)

        removed_n = initial_n - np.sum(mask)

        # APPLY FILTER
        X = X[mask]
        y = y[mask]

        if center is not None:
            center = center[mask]

        # RESET INDICES (CRITICAL FIX)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if center is not None:
            center = center.reset_index(drop=True)

        print(f"Removed {removed_n} samples.")
        print(f"Remaining samples: {len(y)}")

        # ========================================================
        # REMAINING LABEL DISTRIBUTION
        # ========================================================
        print("\nRemaining label distribution:")
        print(pd.Series(y).value_counts())

        pd.Series(y).value_counts().plot(kind="bar")
        plt.title("Remaining Label Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/remaining_label_distribution.png")
        plt.close()

        # ========================================================
        # REMAINING CENTER DISTRIBUTION
        # ========================================================
        if center is not None:

            print("\nRemaining center distribution:")
            print(center.value_counts())

            center.value_counts().plot(kind="bar")
            plt.title("Remaining Center Distribution")
            plt.xlabel("Center")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/remaining_center_distribution.png")
            plt.close()

    else:
        print("All classes meet the minimum class size requirement.")

    # ============================================================
    # LABEL ENCODING
    # ============================================================
    le = LabelEncoder()

    y_encoded = le.fit_transform(y)

    print("\nLabel encoding mapping:")

    for original, encoded in zip(le.classes_, range(len(le.classes_))):
        print(f"{original} → {encoded}")

    # ============================================================
    # CLEAN NUMERIC FEATURES
    # ============================================================
    X = X.replace([np.inf, -np.inf], np.nan)

    # DROP ALL-NaN COLUMNS
    all_nan = X.columns[X.isna().all()].tolist()

    if all_nan:
        print(f"\nDropping {len(all_nan)} empty columns with only NaN.")
        X = X.dropna(axis=1, how="all")

    # IMPUTE REMAINING NaNs
    X = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns
    )

    # ============================================================
    # ENSURE CENTER EXISTS
    # ============================================================
    if center is None:
        center = pd.Series(["Unknown"] * len(X), name="center")

    print(f"\nFinal cleaned dataset: {X.shape[0]} samples × {X.shape[1]} features")

    return X, y_encoded, center
