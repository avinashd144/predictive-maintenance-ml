# Predictive Maintenance in Heavy Machinery
### Machine Learning Approach Using Vibration and Temperature Sensor Data

**MSc Data Analytics Dissertation | Dublin Business School | August 2025**  
**Author:** Avinash Desai | [linkedin.com/in/avinash-desai4](https://linkedin.com/in/avinash-desai4) | [github.com/avinashd144](https://github.com/avinashd144)

---

## Project Overview

This project applies supervised machine learning to predict equipment failures in heavy machinery using the **AI4I 2020 Predictive Maintenance Dataset** (10,000 records, 14 features). The goal is to classify machine failure events from multivariate sensor data — including air temperature, process temperature, rotational speed, torque, and tool wear — before they occur.

The work connects directly to 5 years of real-world industrial IoT analytics at John Deere Technology Center, where predictive maintenance and duty cycle analytics were applied to agricultural machinery at scale.

### Research Questions

1. To what extent can ML models trained on multivariate sensor data accurately predict mechanical failures before they occur?
2. How can ML algorithms improve predictive maintenance efficiency and cost-effectiveness in heavy machinery?

---

## Results Summary

| Model               | Accuracy | Precision | Recall | F1-Score | AUC  |
|---------------------|----------|-----------|--------|----------|------|
| Logistic Regression | 0.81     | 0.13      | 0.70   | 0.22     | 0.84 |
| Random Forest       | 0.96     | 0.48      | 0.64   | 0.55     | 0.92 |
| **XGBoost** ✅      | **0.96** | **0.49**  | **0.65** | **0.56** | **0.92** |
| SVM                 | 0.91     | 0.25      | 0.70   | 0.36     | 0.90 |
| KNN                 | 0.88     | 0.20      | 0.72   | 0.31     | 0.85 |

> **Recall was prioritised as the primary metric** — in predictive maintenance, a missed failure (false negative) carries a higher operational cost than a false alarm.

**Recommended model:** Tuned XGBoost with calibrated decision thresholds for recall-optimised deployment.

---

## Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — UCI Machine Learning Repository

| Property         | Detail                                      |
|------------------|---------------------------------------------|
| Rows             | 10,000                                      |
| Features         | 14 (6 used for modelling after preprocessing) |
| Target variable  | `Machine failure` (binary: 0 = No Failure, 1 = Failure) |
| Class imbalance  | ~96.5% No Failure / ~3.5% Failure           |
| Failure modes    | TWF, HDF, PWF, OSF, RNF (logical OR → target) |

**Key features:** Machine Type (L/M/H), Air Temperature [K], Process Temperature [K], Rotational Speed [rpm], Torque [Nm], Tool Wear [min]

> Dataset file: `data/Dataset1.xlsx`

---

## Methodology

```
Data Collection (Dataset1.xlsx)
        ↓
Data Loading & Cleaning
        ↓
EDA & Feature Engineering
        ↓
Preprocessing (Encoding → Scaling → SMOTE)
        ↓
         ┌─────────────────────┬──────────────────────┐
         ↓                     ↓                      ↓
Baseline Models          Ensemble/Boosting      K-Means Diagnostic
(LR, KNN, SVM)           (RF, XGBoost)
         └─────────────────────┴──────────────────────┘
                              ↓
             Hyperparameter Tuning (GridSearchCV)
                              ↓
         Evaluation (Confusion Matrix, ROC, PR Curve, AUC)
                              ↓
          Interpretation & Feature Importance
                              ↓
             Conclusion & Deployment Recommendations
```

### Key Techniques

- **SMOTE** (Synthetic Minority Oversampling Technique) — applied to training data only to address class imbalance (training set balanced to 6,737 / 6,737)
- **GridSearchCV** with Stratified K-Fold cross-validation for hyperparameter tuning
- **StandardScaler** for feature normalisation (essential for LR, SVM, KNN)
- **Label Encoding** for categorical Machine Type feature
- **K-Means Clustering** (k=2, PCA projection) as unsupervised diagnostic

---

## Repository Structure

```
predictive-maintenance-heavy-machinery/
│
├── notebooks/
│   └── Avinash_CodeCA2_20032381_AppliedResearchProject.ipynb   # Full ML pipeline
│
├── data/
│   └── Dataset1.xlsx                                            # AI4I 2020 dataset
│
├── reports/
│   ├── Avinash_ReportCA2_20032381_AppliedResearchProject.pdf   # 80-page thesis
│   └── Avinash_SlidesCA2_20032381_AppliedResearchProject.pptx  # Presentation deck
│
└── README.md
```

---

## Tech Stack

| Category              | Tools & Libraries                                      |
|-----------------------|--------------------------------------------------------|
| Language              | Python 3.9+                                            |
| Data Manipulation     | pandas, NumPy                                          |
| Visualisation         | Matplotlib, Seaborn                                    |
| Machine Learning      | scikit-learn, XGBoost                                  |
| Imbalance Handling    | imbalanced-learn (SMOTE)                               |
| Dimensionality Red.   | PCA (for K-Means visualisation)                        |
| Environment           | Google Colab (primary), Jupyter Notebook (compatible)  |

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/avinashd144/predictive-maintenance-heavy-machinery.git
cd predictive-maintenance-heavy-machinery
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn openpyxl
```

### 3. Run in Google Colab (Recommended)
- Upload `Dataset1.xlsx` to `/content/` in Colab
- Open `notebooks/Avinash_CodeCA2_20032381_AppliedResearchProject.ipynb`
- Run all cells sequentially

### 4. Run Locally in Jupyter
```bash
jupyter notebook notebooks/Avinash_CodeCA2_20032381_AppliedResearchProject.ipynb
```
> Update the `input_file` path in the notebook from `/content/Dataset1.xlsx` to `../data/Dataset1.xlsx`

---

## Key Findings

- **XGBoost and Random Forest** achieved the best overall performance (AUC = 0.92), delivering the best balance of recall and precision
- **Torque [Nm] and Rotational Speed [rpm]** were the most important predictors of failure across both tree-based models
- **SMOTE** significantly improved minority class detection — training set expanded from 7,000 → 13,474 samples post-resampling
- **K-Means clustering** (k=2) revealed failure-enriched subpopulations, suggesting machine operating regime is a latent predictor of failure risk
- **Logistic Regression** provided interpretable feature coefficients useful for stakeholder communication, despite lower precision
- For recall-critical deployment: recommend XGBoost with threshold calibration (target Recall ≥ 0.85) + two-stage human verification workflow

---

## Industry Connection

This project extends my 5 years of industrial data analytics experience at John Deere Technology Center (Pune), where I worked on:
- IoT telemetry pipelines processing tens of GBs of CAN bus and sensor data
- Engine oil prognosis algorithms achieving 90% accuracy vs SME assessments
- Duty cycle analytics and predictive maintenance for agricultural sprayer equipment

The academic methodology applied here (SMOTE, ensemble models, GridSearchCV, recall-focused evaluation) directly mirrors industry best practice for imbalanced failure classification problems.

---

## Academic Context

| Item               | Detail                                                          |
|--------------------|-----------------------------------------------------------------|
| Degree             | MSc Data Analytics — First Class Honours (1:1)                  |
| Institution        | Dublin Business School, Ireland                                 |
| Module             | B9DA113 Applied Research Project                                |
| Supervisor         | Sajal Kaur Minhas                                               |
| Submission         | August 2025                                                     |

---

## License

This project is shared for portfolio and educational purposes. The AI4I 2020 dataset is publicly available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) under its original license.

---

*Part of my data analytics portfolio — [github.com/avinashd144](https://github.com/avinashd144)*

## Author
Avinash Desai — Data Analyst | PL-300 Certified | MSc Data Analytics (1:1) DBS

🔗 LinkedIn https://www.linkedin.com/in/avinash-desai4/

📧 desaiavinash93@gmail.com
