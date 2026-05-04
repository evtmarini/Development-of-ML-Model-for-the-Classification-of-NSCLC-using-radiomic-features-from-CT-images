"""
Model Definition Module

Defines machine learning models and hyperparameter grids used in the study.

Model categories include:
- Linear models (Logistic Regression)
- Kernel-based methods (SVM)
- Tree-based ensemble models (Random Forest, XGBoost, LightGBM)
- Ensemble learning methods (Stacking, Voting)

All models are configured with:
- Class balancing (class_weight="balanced" where applicable)
- Reproducibility (fixed random_state)
- Proper preprocessing pipelines (scaling, optional PCA)

Designed for radiomics-based binary classification tasks.
"""

from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier,
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_models_and_params():

    # ==========================================================
    # BASE MODELS
    # ==========================================================

    # Random Forest
    # Non-linear ensemble model robust to noise and feature interactions
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )

    # Support Vector Machine (RBF kernel)
    # Effective in high-dimensional spaces; PCA reduces noise and redundancy
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.9, random_state=42)),
        ("clf", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Logistic Regression (L2 regularization)
    # Interpretable baseline linear model
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            random_state=42
        ))
    ])

    # ==========================================================
    # TREE-BASED MODELS
    # ==========================================================

    # XGBoost
    # Gradient boosting algorithm optimized for structured/tabular data
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",  # Binary classification metric
        random_state=42
    )

    # LightGBM
    # Efficient gradient boosting with leaf-wise tree growth
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        class_weight="balanced",
        random_state=42
    )

    # ==========================================================
    # ENSEMBLE METHODS
    # ==========================================================

    # Stacking Ensemble
    # Combines heterogeneous base learners to improve generalization
    stacking_model = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("svm", svm),
            ("xgb", xgb)
        ],
        final_estimator=GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        passthrough=True,
        n_jobs=-1
    )

    # Soft Voting Ensemble
    # Aggregates probabilistic predictions from base models
    soft_voting = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("svm", svm)
        ],
        voting="soft",
        weights=[1, 1],
        n_jobs=-1
    )

    # ==========================================================
    # MODEL REGISTRY
    # ==========================================================

    models = {
        "Random Forest": rf,
        "SVM (RBF)": svm,
        "Logistic Regression": log_reg,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "Stacking Ensemble": stacking_model,
        "Soft Voting Ensemble": soft_voting
    }

    # ==========================================================
    # HYPERPARAMETER GRIDS
    # ==========================================================

    params = {

        "Random Forest": {
            "n_estimators": [300, 600, 1000],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"]
        },

        "SVM (RBF)": {
            "pca__n_components": [0.85, 0.9, 0.95],
            "clf__C": [0.1, 1, 10, 50],
            "clf__gamma": [1e-4, 1e-3, 0.01, 0.1, "scale"]
        },

        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1, 10, 100]
        },

        "XGBoost": {
            "n_estimators": [300, 500, 800],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0]
        },

        "LightGBM": {
            "n_estimators": [300, 500, 800],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [-1, 5, 10],
            "num_leaves": [31, 63, 127]
        },

        "Stacking Ensemble": {
            "final_estimator__n_estimators": [100, 200, 300],
            "final_estimator__learning_rate": [0.03, 0.05, 0.1],
            "final_estimator__max_depth": [2, 3, 4]
        },

        "Soft Voting Ensemble": {
            "weights": [(1, 1), (2, 1), (1, 2)]
        }
    }

    return models, params
