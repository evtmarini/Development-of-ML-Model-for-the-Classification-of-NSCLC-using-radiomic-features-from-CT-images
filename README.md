# Explainable Radiomics-Based Machine Learning Model for Lung Cancer Subtype Classification

Development of an explainable machine learning pipeline for lung cancer subtype classification using radiomics features extracted from CT imaging.

---

## Overview

This repository contains the full implementation of a machine learning pipeline for the classification of lung cancer subtypes (Adenocarcinoma, Squamous Cell Carcinoma) using radiomics features derived from CT images.

The project was developed as part of a postgraduate thesis:

“Explainable Radiomics-Based ML Model for Lung Cancer Subtype Classification”

The pipeline integrates:

- Data preprocessing and normalization  
- ComBat harmonization for multi-center data  
- Feature selection using multiple strategies  
- Training of ML models (LightGBM, XGBoost, Random Forest, Ensembles)  
- Explainability using SHAP and LIME  

**Objective:**  
To develop a robust, generalizable, and interpretable machine learning model for lung cancer subtype classification using radiomics features.

---

## Background

Lung cancer is the leading cause of cancer-related deaths worldwide (~19%).

- NSCLC (~85%)
  - Adenocarcinoma (ADC)  
  - Squamous Cell Carcinoma (SCC)  
  - Large Cell Carcinoma  

- SCLC (~15%)
  - More aggressive subtype  

Accurate subtype classification is essential for:
- Treatment planning  
- Prognosis estimation  
- Personalized medicine  

Radiomics provides a non-invasive approach by extracting quantitative features from medical images.

---


## Project Structure

| Component | Description |
|:----------|:------------|
| `src/` | Core modules of the pipeline |
| `src/load_data.py` | Data loading and preprocessing |
| `src/split_and_check.py` | Center-aware splitting and validation |
| `src/feature_selection.py` | Feature selection algorithms |
| `src/models.py` | Machine learning models and hyperparameters |
| `src/evaluation.py` | Nested cross-validation framework |
| `src/explainability.py` | SHAP-based explainability module |
| `scripts/main.py` | Main pipeline execution script |
| `scripts/holdout_pipeline.py` | Final hold-out evaluation |
| `scripts/shap_reporting.py` | SHAP visualization and reporting |
| `data/` | Input radiomics dataset |
| `results/` | Generated results and figures |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
---

## Pipeline

The pipeline follows a modular and reproducible workflow:

1. Data Loading and Cleaning  
2. Train/Test Splitting (center-aware)  
3. ComBat Harmonization  
4. Preprocessing and Normalization  
5. Feature Selection  
   - LASSO  
   - Boruta  
   - mRMR  
   - RFE-SVM  
   - ReliefF  
6. Model Training  
   - LightGBM  
   - XGBoost  
   - Random Forest  
   - SVM  
   - Ensemble methods  
7. Evaluation  
   - Nested Cross-Validation  
   - Hold-out testing  
8. Explainability  
   - SHAP  
   - LIME  

---

## Evaluation Strategy

- Nested Cross-Validation  
  - Inner CV for hyperparameter tuning and feature selection  
  - Outer CV for unbiased performance estimation  

- Hold-out dataset (10%)  
  - Final generalization evaluation  

All experiments were performed using fixed random seeds (random_state = 42) to ensure reproducibility.

---

## Results

### Inner Cross-Validation

Top-performing configurations:

- Soft Voting + ReliefF (Top-k = 90) → wF1 = 0.83 ± 0.02  
- XGBoost + RFE-SVM (Top-k = 90) → wF1 = 0.83 ± 0.01  
- Soft Voting + RF importance (Top-k = 90) → wF1 = 0.83 ± 0.01  

Performance differences were minimal, indicating robust and consistent model behavior.

Lowest performance:
- Logistic Regression + mRMR (Top-k = 10)  
  → wF1 = 0.48 ± 0.02, AUC = 0.56  

---

### Outer Cross-Validation

- Random Forest + ReliefF (Top-k = 50)  
  → wF1 = 0.80 ± 0.03, AUC = 0.84  

- Stacking Ensemble + RFE-SVM (Top-k = 90)  
  → wF1 = 0.80, AUC = 0.81  

- Stacking Ensemble + ReliefF (Top-k = 70)  
  → wF1 = 0.80 ± 0.02, AUC = 0.79  

These results demonstrate stable and robust generalization performance across models and feature subsets.

---

### Hold-out Set

Best-performing configuration:

- ReliefF + Random Forest (Top-k = 50)

Performance:

- wF1-score: 0.814  
- AUC: 0.854  

This indicates strong predictive performance on unseen data.

---

## Explainability

Model interpretability was assessed using post-hoc explainability techniques:

- SHAP (global feature importance)  
- LIME (local explanations)  

Explainability was performed on both outer cross-validation and independent hold-out datasets to ensure consistency.

Key findings:

- Dominance of wavelet-transformed texture features  
- Important feature families:
  - GLCM  
  - GLRLM  
  - GLSZM  
  - GLDM  

Class-specific patterns:
- ADC: entropy and low-dependence features  
- SCC: variance and dependence-related features  

Feature importance patterns were consistent across validation and hold-out sets.

---

## Feature Analysis

Analysis of the optimal feature subset (Top-k = 50):

- Dominance of higher-order radiomic features  
- Main image domains:
  - Wavelet-transformed images  
  - Laplacian of Gaussian (LoG)  

Feature distribution:
- Texture features (GLCM, GLRLM) dominate  
- GLDM and GLSZM follow  
- First-order features are limited  

This highlights the importance of multi-scale texture heterogeneity.

---

## Data

- Radiomics features extracted using PyRadiomics  
- Labeled dataset for subtype classification  

Note: Raw CT images are not included.

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
