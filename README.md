# Explainable Radiomics-Based Machine Learning Model for Lung Cancer Subtype Classification

Development of an explainable ML pipeline for lung cancer subtype classification using radiomics features extracted from CT imaging.

---

## Overview

This repository contains the full implementation of a machine learning pipeline for the classification of lung cancer subtypes (Adenocarcinoma , Squamous Cell Carcinoma) using radiomics features derived from CT images.

The project was developed as part of a postgraduate thesis:

“Explainable Radiomics-Based ML Model for Lung Cancer Subtype Classification”

The pipeline integrates:

- Data preprocessing and normalization  
- ComBat harmonization for multi-center data  
- Feature selection using multiple strategies  
- Training of ML models (LightGBM, XGBoost, Random Forest, Ensembles)  
- Explainability using SHAP and LIME  

Objective:  
To develop a robust, generalizable, and interpretable machine learning model for lung cancer subtype classification.

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
