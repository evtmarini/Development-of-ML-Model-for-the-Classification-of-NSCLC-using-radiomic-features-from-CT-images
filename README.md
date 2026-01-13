
---

# NSCLC Classification Pipeline

Development of ML model for non-small cell lung cancer detection

# SYNOPSIS

Αυτό το repository περιέχει τον πλήρη κώδικα για την ανάπτυξη ενός **Machine Learning pipeline** που ταξινομεί φαινοτύπους **NSCLC και SCLC** από CT εικόνες, χρησιμοποιώντας **ραδιομικά χαρακτηριστικά** εξαγόμενα από αξονικές τομογραφίες (CT).

Το έργο αποτελεί μέρος της διπλωματικής εργασίας:
*"Development of ML model for non-small cell lung cancer detection"*

Το pipeline περιλαμβάνει:

* Προεπεξεργασία δεδομένων και normalization
* **ComBat harmonization** για πολυκεντρικά δεδομένα
* Επιλογή χαρακτηριστικών μέσω πολλαπλών μεθόδων
* Εκπαίδευση ισχυρών ταξινομητών (LightGBM, XGBoost, Soft Voting Ensemble)
* Ανάλυση ερμηνευσιμότητας με **SHAP** και **LIME**

Στόχος είναι η **βελτιστοποίηση της απόδοσης** και η **εξασφάλιση γενικευσιμότητας**.

---

# Εισαγωγή

Ο καρκίνος του πνεύμονα είναι η κύρια αιτία θανάτων από καρκίνο παγκοσμίως (~19% το 2022).
Το NSCLC (Non-Small Cell Lung Cancer) αντιπροσωπεύει περίπου το 85% των περιπτώσεων και περιλαμβάνει:

* Αδενοκαρκίνωμα (Adenocarcinoma)
* Πλακώδες καρκίνωμα (Squamous Cell Carcinoma)
* Μεγαλοκυτταρικό καρκίνωμα (Large-Cell Carcinoma)

Το SCLC (Small Cell Lung Cancer) είναι πιο επιθετικό και αντιπροσωπεύει ~15% των περιπτώσεων.

Η ακριβής διάκριση των υποτύπων είναι κρίσιμη για:

* Στοχευμένη θεραπεία
* Πρόγνωση ασθενούς
* Εξατομικευμένη ιατρική

Η **Ραδιομική (Radiomics)** παρέχει μια **μη επεμβατική** προσέγγιση, εξάγοντας ποσοτικά χαρακτηριστικά από ιατρικές εικόνες και επιτρέποντας την ανάπτυξη αξιόπιστων μοντέλων πρόβλεψης.

---

## Δομή Αρχείων

```
NSCLC Classification/
│
├── main.py
│
├── data/
│   ├── Radiomic_Features_All.xlsx
│   └── labeled_radiomics_features.csv
│
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── split_and_check.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── explainability.py
│   └── __init__.py
│
├── results/
│   ├── inner_cv_results.csv
│   ├── outer_cv_results.csv
│   ├── selected_features.csv
│   ├── top3_models.csv
│   ├── top3_features_classes.csv
│   ├── holdout_results.csv
│   └── explainability/
│       ├── shap_bar_plot.png
│       ├── shap_summary_plot.png
│       └── lime_example.html
```

---

## Περιγραφή Pipeline

Το pipeline αποτελείται από τα εξής στάδια:

| Στάδιο                     | Module                     | Περιγραφή                                                                                                 |
| -------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------- |
| 1. Φόρτωση δεδομένων       | `src/load_data.py`         | Ανάγνωση αρχείων Excel/CSV, έλεγχος δεδομένων, encoding labels και center, διαχείριση ελλειπών τιμών      |
| 2. Διαχωρισμός             | `src/split_and_check.py`   | Center- & label-aware splitting, ισορροπημένα outer CV folds, έλεγχος ετερογένειας                        |
| 3. Harmonization           | —                          | **ComBat harmonization** για διόρθωση batch / center effects                                              |
| 4. Προεπεξεργασία          | `src/preprocessing.py`     | Κανονικοποίηση, φιλτράρισμα χαμηλής διακύμανσης, αφαίρεση συσχετισμένων χαρακτηριστικών, στατιστικά tests |
| 5. Επιλογή χαρακτηριστικών | `src/feature_selection.py` | LASSO, RF importance, Boruta, RFE-SVM, mRMR, ReliefF                                                      |
| 6. Εκπαίδευση μοντέλων     | `src/models.py`            | LightGBM, XGBoost, Random Forest, Soft Voting Ensemble, Logistic Regression, SVM                          |
| 7. Αξιολόγηση              | `src/evaluation.py`        | **Nested Cross-Validation**, Outer CV, Inner CV, hold-out evaluation, F1-score, Accuracy, ROC-AUC         |
| 8. Ερμηνευσιμότητα         | `src/explainability.py`    | **SHAP & LIME**, ανάλυση κρίσιμων χαρακτηριστικών για τις προβλέψεις                                      |

---

# Δεδομένα

Τα δεδομένα περιλαμβάνονται στον φάκελο `data/`:

* **Radiomic features** εξαγόμενα από CT εικόνες μέσω PyRadiomics
* **labeled_radiomics_features.csv**: χαρακτηριστικά με ετικέτες υποτύπων
* **Radiomic_Features_All.xlsx**: αναφορά χαρακτηριστικών με metadata

---

## Evaluation Strategy

Η αξιολόγηση του μοντέλου βασίστηκε σε μια **συστηματική διαδικασία cross-validation** για εξασφάλιση γενικευσιμότητας και αποφυγή overfitting.

### 1. Inner Cross-Validation (Inner CV)

* 3-fold stratified splits εντός κάθε outer fold
* Επιλογή βέλτιστων hyperparameters και top-k χαρακτηριστικών (10-100)
* Μέθοδοι επιλογής χαρακτηριστικών: LASSO, RF_importance, Boruta, RFE_SVM, mRMR, ReliefF
* Αποτελέσματα καταγράφηκαν στο `inner_cv_results.csv`

### 2. Outer Cross-Validation (Outer CV)

* 3 εξωτερικά folds, ισορροπημένα σε κλάσεις και centers
* Εκπαίδευση μοντέλων με χαρακτηριστικά από inner CV
* Αξιολόγηση LightGBM, XGBoost, Soft Voting Ensemble
* Αποτελέσματα αποθηκεύτηκαν στο `outer_cv_results.csv`
* Επιλεγμένα χαρακτηριστικά στο `selected_features.csv`

### 3. Hold-Out Set

* 10% του dataset, ανεξάρτητο από οποιαδήποτε διαδικασία εκπαίδευσης
* Τελική εκτίμηση γενικευσιμότητας
* Αποτελέσματα στο `holdout_results.csv`

---

## Κορυφαία Μοντέλα & Αποτελέσματα

| Feature Selection | Classifier           | Top-k | Outer F1 | Outer AUC |
| ----------------- | -------------------- | ----- | -------- | --------- |
| RFE_SVM           | LightGBM             | 50    | 0.746    | 0.845     |
| RFE_SVM           | Soft Voting Ensemble | 50    | 0.732    | 0.827     |
| mRMR              | Soft Voting Ensemble | 60    | 0.729    | 0.821     |

**Hold-out set (10%)**:
F1 = 0.684, Accuracy = 0.717, AUC = 0.772

---

## Ερμηνευσιμότητα

Η ανάλυση **SHAP/LIME** στο hold-out set αποκάλυψε τα πιο κρίσιμα χαρακτηριστικά για διάκριση NSCLC φαινοτύπων:

* wavelet-transformed texture features
* log-sigma 3D texture features
* first-order features
* shape features

Αυτό υποστηρίζει την κλινική ερμηνεία των μοντέλων.

---

## Συμπεράσματα

Το pipeline επιβεβαιώνει ότι ένα **σωστά σχεδιασμένο ML workflow**, με:

* Σωστή επιλογή χαρακτηριστικών
* Ισχυρούς ταξινομητές (ensemble/gradient boosting)
* Αυστηρή cross-validation

μπορεί να προσφέρει **αξιόπιστη διάκριση φαινοτύπων NSCLC/SCLC**, ανοίγοντας τον δρόμο για **ενσωμάτωση σε κλινικά συστήματα υποστήριξης αποφάσεων**.

---

Αν θέλεις, μπορώ να φτιάξω και μια **γραφική έκδοση README** με **διαγράμματα pipeline και top-3 μοντέλα**, ώστε να είναι **εντυπωσιακό για GitHub**.

Θέλεις να το κάνω;
