# ❤️ Heart Disease Prediction — ML Classification Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org)

A complete, end-to-end machine learning pipeline that predicts whether a patient has heart disease using clinical measurements. Built as a mini-project for the CS 481 Artificial Intelligence course.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Key Finding — Silent Ischemia](#-key-finding--silent-ischemia)
- [ML Pipeline](#-ml-pipeline)
- [Results](#-results)
- [Feature Importance](#-feature-importance)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)

---

## 🔍 Project Overview

Heart disease is the leading cause of death worldwide. This project builds a supervised binary classification system trained on patient clinical data to assist early detection.

**The core challenge:** Many high-risk patients are completely asymptomatic — they feel nothing, yet have an 79% probability of disease. Traditional symptom-based screening misses them entirely. Machine learning is uniquely positioned to detect these hidden patterns.

**Goal:** Maximize **Recall** (catch as many sick patients as possible) while keeping overall performance high — because a missed diagnosis is far more dangerous than a false alarm.

---

##  Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| **Patients** | 918 |
| **Features** | 11 clinical measurements |
| **Target** | `HeartDisease` — 1 = disease, 0 = healthy |
| **Class balance** | 55.3% disease / 44.7% healthy |

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Numerical | Patient age in years (28–77) |
| `Sex` | Categorical | M = Male, F = Female |
| `ChestPainType` | Categorical | TA, ATA, NAP, ASY |
| `RestingBP` | Numerical | Resting blood pressure (mm Hg) |
| `Cholesterol` | Numerical | Serum cholesterol (mg/dl) |
| `FastingBS` | Binary | Fasting blood sugar > 120 mg/dl |
| `RestingECG` | Categorical | Normal / ST / LVH |
| `MaxHR` | Numerical | Max heart rate during stress test |
| `ExerciseAngina` | Binary | Chest pain during exercise (Y/N) |
| `Oldpeak` | Numerical | ST depression at peak exercise |
| `ST_Slope` | Categorical | Slope of the ST segment (Up/Flat/Down) |

> **Data quality note:** 172 patients had `Cholesterol = 0` and some had `RestingBP = 0` — medically impossible values treated as hidden missing data and imputed with the median.

---

##  Key Finding — Silent Ischemia

The most surprising and clinically significant finding from the EDA:

| Chest Pain Type | Disease Rate | Interpretation |
|-----------------|-------------|----------------|
| **ASY — Asymptomatic** | **79.0%** | **Highest risk** — zero symptoms, sickest patients |
| TA — Typical Angina | 43.5% | Moderate risk — patients usually seek help |
| NAP — Non-Anginal Pain | 35.5% | Lower risk |
| ATA — Atypical Angina | 13.9% | Lowest risk |

> Patients who report **no chest pain at all** have the highest disease rate at 79%. This phenomenon — called **silent ischemia** — means traditional triage based on reported symptoms would completely miss the highest-risk group. This is the strongest argument for ML-based screening in clinical settings.

---

##  ML Pipeline

```
Raw Data → EDA → Data Cleaning → Encoding → Train/Test Split → Scaling → Model Training → Evaluation → Hyperparameter Tuning → Final Model
```

### Steps in Detail

1. **Exploratory Data Analysis** — 6 visualizations covering class balance, age distribution, cholesterol, chest pain type, MaxHR/Oldpeak separation, and correlation heatmap
2. **Data Cleaning** — Median imputation for hidden zero values; duplicate removal
3. **Encoding** — `LabelEncoder` for categorical variables (tree models; no one-hot needed)
4. **Stratified Train/Test Split** — 80/20 split with `stratify=y` to preserve class ratio
5. **Feature Scaling** — `StandardScaler` applied *after* split to prevent data leakage
6. **Feature Engineering** — Age binned into risk groups (`AgeGroup`)
7. **Model Training** — 5 classifiers evaluated with consistent metrics
8. **Evaluation** — Accuracy, Precision, Recall, F1-Score, AUC-ROC
9. **Hyperparameter Tuning** — `GridSearchCV` with 5-fold stratified cross-validation on Random Forest
10. **Comparison** — Default RF vs. Tuned RF

### Why Recall is the Primary Metric

The dataset has a slight class imbalance (55.3% disease). More critically, a **False Negative** (predicting healthy when the patient is sick) is far more dangerous than a False Positive. A model predicting "always disease" would score 55.3% accuracy while being useless — confirming that accuracy alone is misleading here. **Recall** and **F1-Score** are the primary evaluation metrics.

---

##  Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** 🏆 | **87.5%** | **87.6%** | **90.2%** | **88.9%** | 0.923 |
| SVM | 87.0% | 86.1% | 91.2% | 88.6% | 0.919 |
| Gradient Boosting | 85.9% | 88.0% | 86.3% | 87.1% | **0.925** |
| Logistic Regression | 84.8% | 84.9% | 88.2% | 86.5% | 0.899 |
| Decision Tree | 80.4% | 82.4% | 82.4% | 82.4% | 0.863 |

**Winner: Random Forest** — Best balance of Recall and F1-Score. Catches **9 out of 10 sick patients**.

Hyperparameter tuning with `GridSearchCV` further improved Recall and F1-Score by optimizing `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.

---

##  Feature Importance (Tuned Random Forest)

| Rank | Feature | Importance | Clinical Meaning |
|------|---------|------------|-----------------|
| 1 | `ST_Slope` | 0.298 | ECG shape at peak exercise — strongest ischemia signal |
| 2 | `ChestPainType` | 0.156 | ASY patients = highest risk (silent ischemia) |
| 3 | `ExerciseAngina` | 0.102 | Coronary supply failure under stress |
| 4 | `MaxHR` | 0.098 | Diseased hearts cannot sustain high rates |
| 5 | `Oldpeak` | 0.096 | ST depression quantifies myocardial stress |

> **Validation:** All top-5 features come from the **exercise stress test** — the gold-standard non-invasive test for coronary artery disease. The model learned medically valid patterns, not noise.

---

## 📁 Repository Structure

```
heart-disease-prediction/
├── README.md                          ← This file
├── requirements.txt                   ← Python dependencies
├── .gitignore                         ← Files to exclude
├── heart.csv                          ← Dataset (918 patients, 11 features)
├── Heart_Disease_mini_project.ipynb   ← Complete Jupyter notebook
├── Heart_Disease_Report.docx          ← Full written report
├── Heart_Disease_Presentation.pdf     ← 5-minute presentation slides
├── model/
│   └── heart_model.pkl                ← Saved trained Random Forest model
└── images/
    ├── silent_ischemia_chart.png
    ├── feature_importance.png
    ├── roc_curves.png
    └── confusion_matrix.png
```

---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the notebook
```bash
jupyter notebook Heart_Disease_mini_project.ipynb
```

### 4. Run all cells
`Kernel → Restart & Run All`

> The notebook is self-contained — it loads the dataset, runs the full pipeline, and saves the trained model to `model/heart_model.pkl`.

---

##  Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
joblib
```

See `requirements.txt` for pinned versions.

---

##  License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

##  Acknowledgements

- Dataset: [fedesoriano](https://www.kaggle.com/fedesoriano) on Kaggle
