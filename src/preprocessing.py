
# preprocessing.py — Feature filtering and transformation utils

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import mannwhitneyu, kruskal
from sklearn.preprocessing import PowerTransformer


# Variance filter — removes near-constant features
def variance_filter(X, threshold=0.001):
    """
    Removes features with variance below the given threshold.
    Args:
        X (pd.DataFrame): Input features
        threshold (float): Variance threshold
    Returns:
        pd.DataFrame: Filtered feature set
    """
    vt = VarianceThreshold(threshold=threshold)
    X_filtered = vt.fit_transform(X)
    kept_cols = X.columns[vt.get_support()]
    print(f"Variance filter removed {X.shape[1] - len(kept_cols)} features (threshold={threshold})")
    return pd.DataFrame(X_filtered, columns=kept_cols)


# Correlation filter — removes highly correlated features
def correlation_filter(X, threshold=0.85):
    """
    Removes one feature from each highly correlated pair (> threshold).
    Keeps the feature with the lower mean correlation to others.
    Args:
        X (pd.DataFrame): Input features
        threshold (float): Correlation threshold
    Returns:
        pd.DataFrame: Filtered feature set
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []

    for column in upper.columns:
        high_corr = upper[column][upper[column] > threshold].index.tolist()
        for hc in high_corr:
            mean_corr_col = corr[column].mean()
            mean_corr_hc = corr[hc].mean()
            drop = column if mean_corr_col > mean_corr_hc else hc
            if drop not in to_drop:
                to_drop.append(drop)

    print(f"Correlation filter removed {len(to_drop)} features (threshold={threshold})")
    return X.drop(columns=to_drop)


# Statistical filter — Kruskal-Wallis or Mann-Whitney test
def stat_filter(X, y, alpha=0.1):
    """
    Selects features significantly associated with target (Kruskal or Mann-Whitney).
    Args:
        X (pd.DataFrame): Input features
        y (array-like): Class labels
        alpha (float): Significance level
    Returns:
        pd.DataFrame: Statistically significant features
    """
    classes = np.unique(y)
    selected = []

    for col in X.columns:
        groups = [X[y == cls][col] for cls in classes]
        if len(classes) == 2:
            stat, p = mannwhitneyu(*groups)
        else:
            stat, p = kruskal(*groups)
        if p < alpha:
            selected.append(col)

    print(f"Stat filter kept {len(selected)} significant features (alpha={alpha})")
    return X[selected]

# Optional: Power transform (normalize skewed data)
def power_transform(X):
    """
    Applies Yeo–Johnson power transform to reduce skewness.
    """
    pt = PowerTransformer(method="yeo-johnson")
    X_transformed = pt.fit_transform(X)
    print("Applied power transform (Yeo–Johnson).")
    return pd.DataFrame(X_transformed, columns=X.columns)
