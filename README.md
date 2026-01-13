# Development-of-ML-Model-for-the-Classification-of-NSCLC-using-radiomic-features-from-CT-images
Development of ML Model for the Classification of NSCLC using radiomic features from CT images


Project Overview

Αυτό το repository περιέχει τον πλήρη κώδικα για το machine learning pipeline που αναπτύχθηκε:

«Development of ML model for non-small cell lung cancer classification»


Στην παρούσα μελέτη αναπτύχθηκε ένα ολοκληρωμένο pipeline μηχανικής μάθησης για την ταξινόμηση φαινοτύπων NSCLC και SCLC από CT εικόνες, χρησιμοποιώντας ραδιομικά χαρακτηριστικά.

Στόχος του έργου είναι η βελτιστοποίηση της απόδοσης και η εξασφάλιση γενικευσιμότητας σε πολυκεντρικά δεδομένα υψηλής διάστασης, μέσω:

Προεπεξεργασίας

ComBat harmonization

Κανονικοποίησης

Επιλογής χαρακτηριστικών μέσω πολυάριθμων μεθόδων

Εκπαίδευσης ισχυρών ταξινομητών

Ανάλυσης ερμηνευσιμότητας με SHAP και LIME

Το pipeline και ο κώδικας είναι διαθέσιμα για περαιτέρω έρευνα και ανάπτυξη μέσω του GitHub repository.

Εισαγωγή

Ο καρκίνος του πνεύμονα αποτελεί την κυριότερη αιτία θανάτου από καρκίνο παγκοσμίως, με το NSCLC να αντιστοιχεί περίπου στο 85% των περιπτώσεων.

Οι βασικοί υποτύποι NSCLC περιλαμβάνουν:

Αδενοκαρκίνωμα

Πλακώδες καρκίνωμα

Μεγαλοκυτταρικό καρκίνωμα

Η ακριβής διάκριση των υποτύπων είναι κρίσιμη για:

επιλογή θεραπευτικής στρατηγικής

πρόγνωση ασθενούς

εξατομικευμένη ιατρική

Η ραδιομική προσφέρει μια μη επεμβατική προσέγγιση, εξάγοντας ποσοτικά χαρακτηριστικά από ιατρικές εικόνες, τα οποία μπορούν να αξιοποιηθούν από αλγορίθμους μηχανικής μάθησης.

Pipeline Architecture

Το pipeline έχει σχεδιαστεί σύμφωνα με best practices για radiomics και medical ML, με αυστηρό διαχωρισμό δεδομένων και nested cross-validation.

Στάδιο	Module	Περιγραφή
1. Data Loading	src/load_data.py	Φόρτωση ραδιομικών δεδομένων, καθαρισμός, encoding labels & centers
2. Data Splitting	src/split_and_check.py	Center- & label-aware splitting, δημιουργία ισορροπημένων outer CV folds
3. Harmonization	—	ComBat harmonization για διόρθωση batch / center effects
4. Preprocessing	src/preprocessing.py	Φιλτράρισμα χαμηλής διακύμανσης, συσχέτισης και στατιστικής σημαντικότητας
5. Feature Selection	src/feature_selection.py	LASSO, RF importance, Boruta, RFE-SVM, mRMR, ReliefF
6. Modeling	src/models.py	Εκπαίδευση και παραμετροποίηση ML classifiers (LightGBM, Soft Voting Ensemble, Logistic Regression, SVM)
7. Evaluation	src/evaluation.py	Nested CV, F1-score, Accuracy, ROC-AUC
8. Explainability	src/explainability.py	Ανάλυση ερμηνευσιμότητας στο hold-out set (SHAP, LIME)
Feature Selection Strategy

Εφαρμόζονται πολλαπλές μέθοδοι feature selection ώστε να μελετηθεί η επίδρασή τους στην απόδοση των μοντέλων:

LASSO

Random Forest feature importance

Boruta

Recursive Feature Elimination (SVM-based)

mRMR

ReliefF

Για κάθε μέθοδο:

αξιολογούνται διαφορετικά μεγέθη υποσυνόλων χαρακτηριστικών (top-k)

η επιλογή γίνεται εντός nested cross-validation

αποφεύγεται information leakage

Σημαντικά χαρακτηριστικά που ξεχώρισαν:

wavelet-transformed 3D texture features

log-sigma 3D texture features

first-order και shape features

Evaluation Strategy

Η αξιολόγηση ακολουθεί τα εξής cross validations:

Outer Cross-Validation – Αμερόληπτη εκτίμηση γενίκευσης

Κορυφαίο μοντέλο: RFE_SVM + LightGBM με 50 χαρακτηριστικά

F1 = 0.746, AUC = 0.845

Άλλα ισχυρά μοντέλα: RFE_SVM + Soft Voting Ensemble, mRMR + Soft Voting Ensemble

Inner Cross-Validation – Βελτιστοποίηση feature selection & hyperparameters

Independent Hold-Out Set (10%) – Τελική αξιολόγηση

F1 = 0.684, Accuracy = 0.717, AUC = 0.772

Ερμηνευσιμότητα: SHAP και LIME αποκάλυψαν τα κρίσιμα χαρακτηριστικά που καθορίζουν τις προβλέψεις, υποστηρίζοντας την κλινική ερμηνεία.

Outputs & Results

Όλα τα αποτελέσματα αποθηκεύονται αυτόματα στον φάκελο results/:

inner_cv_results.csv

outer_cv_results.csv

selected_features.csv

top3_models.csv

top3_features_classes.csv

holdout_results.csv

Παράγονται επίσης:

bar plots

boxplots

heatmaps

κατανομή feature classes

Το pipeline υπογραμμίζει ότι συνδυαστικά pipelines επιλογής χαρακτηριστικών με ισχυρούς ensemble ή gradient boosting ταξινομητές προσφέρουν αξιόπιστη ταξινόμηση NSCLC, ανοίγοντας τον δρόμο για κλινική ενσωμάτωση σε συστήματα υποστήριξης αποφάσεων.

Project Structure
nsclc-classification-pipeline/
│
├── src/
│   ├── load_data.py
│   ├── split_and_check.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── models.py
│   ├── evaluation.py
│   └── explainability.py
│
├── data/
│   └── Radiomic_Features_All.xlsx
│
├── results/
│   ├── *.csv
│   ├── *.png
│   └── explainability/
│
└── main.py


