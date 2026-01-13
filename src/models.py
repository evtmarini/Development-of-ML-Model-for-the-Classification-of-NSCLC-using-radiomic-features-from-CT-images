from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier,
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_models_and_params():

    # ----------------------------------------------------------
    # Base models
    # ----------------------------------------------------------

    # Random Forest
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )

    # SVM (RBF) with PCA
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

    # Logistic Regression (L2)
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            random_state=42
        ))
    ])

    # k-Nearest Neighbors
    knn = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(weights="distance"))
    ])

    # Multi-layer Perceptron (MLP)
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            learning_rate_init=0.001,
            max_iter=1200,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42
        ))
    ])

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    )

    # LightGBM
    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        class_weight="balanced",
        random_state=42
    )

    # Stacking Ensemble
    stacking_model = StackingClassifier(
        estimators=[("rf", rf), ("svm", svm)],
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
    soft_voting = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm)],
        voting="soft",
        weights=[1, 1],
        n_jobs=-1
    )

    # ----------------------------------------------------------
    # Models dictionary
    # ----------------------------------------------------------
    models = {
        "Random Forest": rf,
        "SVM (RBF)": svm,
        "Logistic Regression": log_reg,
        "kNN": knn,
        "MLP (Neural Net)": mlp,
        "XGBoost": xgb,
        "LightGBM": lgbm,
        "Stacking Ensemble": stacking_model,
        "Soft Voting Ensemble": soft_voting
    }

    # ----------------------------------------------------------
    # Parameter grids
    # ----------------------------------------------------------
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

        "kNN": {
            "clf__n_neighbors": [3, 5, 7, 9],
            "clf__weights": ["uniform", "distance"]
        },

        "MLP (Neural Net)": {
            "clf__hidden_layer_sizes": [(64,), (128, 64), (128, 64, 32)],
            "clf__activation": ["relu", "tanh"],
            "clf__learning_rate_init": [0.0005, 0.001, 0.01],
            "clf__alpha": [0.0001, 0.0005, 0.001]
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
