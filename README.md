# Explainable Radiomics-Based Machine Learning Model for Lung Cancer Subtype Classification

---

## Overview

This repository contains the full implementation of a machine learning pipeline for the classification of lung cancer subtypes (Adenocarcinoma, Squamous Cell Carcinoma) using radiomics features derived from CT images.

The project was developed as part of a postgraduate thesis.

---


## Project Structure

| Component | Description |
|:----------|:------------|
| `main.py` | Main pipeline execution script |
| `src/` | Core modules of the pipeline |
| `src/load_data.py` | Data loading and preprocessing |
| `src/split_and_check.py` | Center-aware splitting and validation |
| `src/preprocessing.py` | Center-aware splitting and validation |
| `src/evaluation.py` | Nested cross-validation framework |
| `src/feature_selection.py` | Feature selection algorithms |
| `src/models.py` | Machine learning models and hyperparameters |
| `src/explainability.py` | SHAP-based explainability module |
| `scripts/hold_out.py` | Final hold-out evaluation |
| `Data/` | Input radiomics dataset |
| `Results/` | Generated results and figures |
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
6. Model Training  
7. Evaluation (Cross-Validation, Hold-out testing)
8. Explainability  


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

## Tested Environment

* Python 3.8
* Windows 11
* Visual Studio Code


## Installation

Clone the repository:

```bash
git clone https://github.com/evtmarini/Development-of-ML-Model-for-the-Classification-of-NSCLC-using-radiomic-features-from-CT-images.git
cd Development-of-ML-Model-for-the-Classification-of-NSCLC-using-radiomic-features-from-CT-images


```
Create a virtual environment:

```bash
python -m venv myvenv
```

Activate the environment (Windows):

```bash
.\myvenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```




## Usage



```

Run full pipeline:

```bash
python main.py
```

Run hold-out evaluation only:

```bash
python -m src.hold_out
```

Run explainability analysis only:

```bash
python -m src.explainability
```
