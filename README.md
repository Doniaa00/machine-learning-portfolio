# Machine Learning Case Studies

Applied machine learning case studies built on real-world datasets. Each project covers the full workflow — exploratory analysis, cleaning, modeling, evaluation, and a written report — with an emphasis on **choosing the right metric for the problem** and **explaining what the model learned**.

## Projects

| Project | Task | Best model | Key result |
|---|---|---|---|
| [Heart Disease Prediction](practical-projects/Heart-Disease-Prediction/) | Binary classification (918 patients, 11 clinical features) | Random Forest, tuned with `GridSearchCV` | **91.2% recall**, F1 0.894, ROC-AUC 0.929 on a held-out test set |

### Heart Disease Prediction

Predicts heart disease from routine clinical measurements, optimizing for **recall** because a missed diagnosis costs far more than a false alarm.

- Compared 5 classifiers (Logistic Regression, Decision Tree, Random Forest, SVM, Gradient Boosting) under identical conditions, then tuned Random Forest with 5-fold stratified cross-validation.
- Treated medically impossible zero values (`Cholesterol = 0`, `RestingBP = 0`) as hidden missing data rather than real measurements.
- **Key finding:** asymptomatic patients have the *highest* disease rate (79%) — "silent ischemia" — so symptom-based triage would miss the highest-risk group.
- The top 5 predictors all come from the exercise stress test, consistent with clinical practice.

Deliverables: [notebook](practical-projects/Heart-Disease-Prediction/Heart_Disease_mini%20project.ipynb) · [written report](practical-projects/Heart-Disease-Prediction/Heart_Disease_Report.docx) · [presentation](practical-projects/Heart-Disease-Prediction/Heart%20Disease%20Prediction%20presentation.pdf)

## Repository structure

```
machine-learning-portfolio/
├── README.md
└── practical-projects/
    └── Heart-Disease-Prediction/
        ├── README.md
        ├── requirements.txt
        ├── LICENSE
        ├── Heart_Disease_mini project.ipynb
        ├── Heart_Disease_Report.docx
        └── Heart Disease Prediction presentation.pdf
```

## How to run

Each project has its own `requirements.txt` and setup instructions in its README.

```bash
git clone https://github.com/Doniaa00/machine-learning-portfolio.git
cd machine-learning-portfolio/practical-projects/Heart-Disease-Prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset is not stored in this repository. Download `heart.csv` from [Kaggle — Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction), place it next to the notebook, then run:

```bash
jupyter notebook "Heart_Disease_mini project.ipynb"
```

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## More of my work

- [steg-fraud-detection](https://github.com/Doniaa00/steg-fraud-detection) — LightGBM fraud detection on 4.5M utility invoices, with a live dashboard
- [rag-paper-assistant](https://github.com/Doniaa00/rag-paper-assistant) — hybrid-retrieval RAG over ML research papers *(in progress)*

## Author

**Donia Mabrouk** — [LinkedIn](https://www.linkedin.com/in/donia-mabrouk-19030932a/) · [GitHub](https://github.com/Doniaa00)
