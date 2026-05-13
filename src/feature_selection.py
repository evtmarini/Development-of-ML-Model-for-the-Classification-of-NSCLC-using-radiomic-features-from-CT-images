"""
Feature Selection Module 
Implements multiple feature selection strategies covering:

- Filter methods (statistical & information-based)
- Wrapper methods (model-based selection)
- Embedded methods (regularization & tree-based importance)

Selected methods used in the study:
    - LASSO (embedded)
    - Random Forest importance (embedded)
    - Boruta (wrapper)
    - RFE-SVM (wrapper)
    - mRMR (filter)
    - ReliefF (filter)

Includes:
    - sklearn-compatible FeatureSelector class

Designed for high-dimensional radiomics data.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
from skrebate import ReliefF
from sklearn.base import BaseEstimator, TransformerMixin


# FILTER METHODS

# mRMR (minimum Redundancy Maximum Relevance)
# Selects features with high relevance to target and low redundancy between them
def fs_mrmr(X, y, top_k=30):
    print(f"Running mRMR (fallback) for {top_k} features...")
    mi = mutual_info_classif(X, y, random_state=42)
    scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    selected = []
    for feat in scores.index:
        if len(selected) >= top_k:
            break
        if selected and X[selected].corrwith(X[feat]).abs().max() > 0.85:
            continue
        selected.append(feat)
    print(f"mRMR selected {len(selected)} features.")
    return selected

# ReliefF algorithm
# Estimates feature importance based on nearest neighbor differences
def fs_relieff(X, y, top_k=30):
    print(f"Running ReliefF for {top_k} features...")
    X_scaled = StandardScaler().fit_transform(X)
    relief = ReliefF(n_neighbors=20, n_features_to_select=top_k, n_jobs=-1)
    relief.fit(X_scaled, y)
    feats = X.columns[relief.top_features_[:top_k]].tolist()
    print(f"ReliefF selected {len(feats)} features.")
    return feats


# WRAPPER METHODS

# Boruta wrapper method
# Identifies all relevant features using Random Forest and shadow features
def fs_boruta(X, y):
    print("Running Boruta feature selection...")
    rf = RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced",
        max_depth=7,
        random_state=42
    )
    bor = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        perc=80,
        random_state=42
    )
    bor.fit(X.values, y)
    selected = X.columns[bor.support_].tolist()
    print(f"Boruta selected {len(selected)} features.")
    return selected

# Recursive Feature Elimination with linear SVM
# Iteratively removes least important features based on model weights
def fs_rfe_svm(X, y, n_features=30):
    print(f"Running RFE with linear SVM (target {n_features})...")
    estimator = SVC(kernel="linear", random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]), step=0.1)
    rfe.fit(StandardScaler().fit_transform(X), y)
    feats = X.columns[rfe.support_].tolist()
    print(f"RFE-SVM selected {len(feats)} features.")
    return feats
  


# EMBEDDED METHODS

# L1-regularized Logistic Regression (LASSO)
# Performs embedded feature selection by shrinking coefficients to zero
def fs_lasso(X, y):
    print("Running tuned L1-LASSO selection...")
    X_scaled = StandardScaler().fit_transform(X)
    lasso = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=2,
        max_iter=10000,
        class_weight="balanced",
        random_state=42
    )
    lasso.fit(X_scaled, y)
    coef_mean = np.mean(np.abs(lasso.coef_), axis=0)
    feats = X.columns[coef_mean > 1e-5].tolist()
    print(f"LASSO retained {len(feats)} features.")
    return feats

        
# Random Forest feature importance
# Ranks features based on impurity reduction across trees
def fs_rf_importance(X, y, top_k=30):
    print("Running Random Forest importance selection...")
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    rf.fit(X, y)
    feats = pd.Series(rf.feature_importances_, index=X.columns).nlargest(top_k).index.tolist()
    print(f"RF-importance selected {len(feats)} features.")
    return feats


# SKLEARN-COMPATIBLE WRAPPER
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method, top_k=30):
        self.method = method
        self.top_k = top_k
        self.selected_features_ = None

    def fit(self, X, y):
        selected = self.method(pd.DataFrame(X), y, top_k=self.top_k) \
            if 'top_k' in self.method.__code__.co_varnames \
            else self.method(pd.DataFrame(X), y)
        self.selected_features_ = (
            selected.columns if isinstance(selected, pd.DataFrame)
            else list(selected)
        )
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise RuntimeError("FeatureSelector not fitted yet.")
        return pd.DataFrame(X)[self.selected_features_]
