# Predictive Maintenance in Heavy Machinery using ML
### Machine Learning Approach Using Vibration and Temperature Sensor Data

**MSc Data Analytics Dissertation | Dublin Business School | August 2025**  
**Author:** Avinash Desai | [linkedin.com/in/avinash-desai4](https://linkedin.com/in/avinash-desai4) | [github.com/avinashd144](https://github.com/avinashd144)

---

## Project Overview

This project applies supervised machine learning to predict equipment failures in heavy machinery using the **AI4I 2020 Predictive Maintenance Dataset** (10,000 records, 14 features). The goal is to classify machine failure events from multivariate sensor data — including air temperature, process temperature, rotational speed, torque, and tool wear — before they occur.

The work connects directly to 5 years of real-world industrial IoT analytics at **John Deere Technology Center**, where predictive maintenance and duty cycle analytics were applied to agricultural machinery at scale — including IoT telemetry pipelines, engine oil prognosis algorithms, and duty cycle analytics across multiple sprayer variants.

### Research Questions

1. To what extent can ML models trained on multivariate sensor data accurately predict mechanical failures before they occur?
2. How can ML algorithms improve predictive maintenance efficiency and cost-effectiveness in heavy machinery?

---

## Results Summary

> **Enhanced notebook** — 3 physics-based engineered features added (`Power_W`, `Temp_Diff_K`, `Torque_x_ToolWear`) improving Random Forest F1 from 0.55 → 0.68 and Precision from 0.48 → 0.65 vs original thesis results.

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.85 | 0.16 | 0.72 | 0.26 | 0.87 |
| **Random Forest** ✅ | **0.97** | **0.65** | **0.71** | **0.68** | **0.93** |
| XGBoost | 0.98 | 0.70 | 0.66 | 0.68 | 0.92 |
| SVM | 0.93 | 0.29 | 0.64 | 0.40 | 0.88 |
| KNN | 0.90 | 0.22 | 0.66 | 0.33 | 0.86 |

> **Recommended model: Random Forest** — highest AUC (0.93) among models achieving Recall ≥ 0.65, with the best precision-recall balance (F1 = 0.68).

> **Recall prioritised as primary metric** — in predictive maintenance, a missed failure (false negative) carries a higher operational cost than a false alarm.

> **Threshold analysis:** Lowering the XGBoost decision threshold achieves Recall = 0.86 at the cost of lower precision — appropriate for recall-critical deployment where missing a failure is more costly than a false alarm.

---

## Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — UCI Machine Learning Repository

| Property | Detail |
|---|---|
| Rows | 10,000 |
| Features | 14 (9 used for modelling: 6 original + 3 engineered) |
| Target variable | `Machine failure` (binary: 0 = No Failure, 1 = Failure) |
| Class imbalance | ~96.5% No Failure / ~3.5% Failure |
| Failure modes | TWF, HDF, PWF, OSF, RNF (logical OR → target) |

**Key features:** Machine Type (L/M/H), Air Temperature [K], Process Temperature [K], Rotational Speed [rpm], Torque [Nm], Tool Wear [min]

**Dataset note:** Some rows have `Machine failure = 1` with all five mode flags = 0. This is a known characteristic of the AI4I 2020 dataset — likely related to random failures not captured by the named modes. The target variable `Machine failure` is used directly for modelling and this inconsistency does not affect results.

> Dataset file: `data/Dataset1.xlsx`

---

## Feature Engineering

Three physics-based features were engineered directly from the failure mode equations documented in the dataset:

| Feature | Formula | Maps To | Failure Trigger |
|---|---|---|---|
| `Power_W` | Torque × (RPM × 2π/60) | PWF (Power Failure) | Power < 3500W or > 9000W |
| `Temp_Diff_K` | Process Temp − Air Temp | HDF (Heat Dissipation Failure) | Temp Diff < 8.6K |
| `Torque_x_ToolWear` | Torque × Tool Wear | OSF (Overstrain Failure) | Exceeds variant threshold |

These three failure modes account for the majority of failure events. Engineering these features gives the model a direct signal for the most common failure causes — and each is fully defensible in a technical interview.

---

## Methodology

```
Data Collection (Dataset1.xlsx)
        ↓
Data Loading & Validation
        ↓
Exploratory Data Analysis
(Target distribution, failure modes, distributions by failure class, correlation)
        ↓
Feature Engineering
(Power_W, Temp_Diff_K, Torque_x_ToolWear)
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
          Model Comparison + Threshold Analysis
                              ↓
             Conclusions & Deployment Recommendations
```

### Key Techniques

- **Feature Engineering** — 3 physics-based features derived from failure mode equations
- **SMOTE** — applied to training data only; balanced training set from 7,000 → 13,474 samples
- **GridSearchCV** with Stratified K-Fold (5 folds) for hyperparameter tuning
- **StandardScaler** — feature normalisation for LR, SVM, KNN
- **Threshold Analysis** — operating point calibration for Recall ≥ 0.85
- **K-Means Clustering** (k=2) — unsupervised diagnostic on original data (not SMOTE data)

---

## Repository Structure

```
predictive-maintenance-ml/
│
├── notebooks/
│   ├── predictive_maintenance_notebook.ipynb        # Enhanced ML pipeline (use this)
│   └── archive/
│       └── predictive_maintenance_notebook_original.ipynb  # Original thesis version
│
├── data/
│   └── Dataset1.xlsx                                # AI4I 2020 dataset
│
├── reports/
│   ├── predictive_maintenance_thesis_report.pdf     # 80-page MSc thesis
│   └── predictive_maintenance_slides.pptx           # Presentation deck
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Category | Tools & Libraries |
|---|---|
| Language | Python 3.9+ |
| Data Manipulation | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Dimensionality Reduction | PCA (for K-Means visualisation) |
| Environment | Jupyter Notebook (local) / Google Colab (cloud) |

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/avinashd144/predictive-maintenance-ml.git
cd predictive-maintenance-ml
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally in Jupyter
```bash
jupyter notebook notebooks/predictive_maintenance_notebook.ipynb
```
> The notebook uses `DATA_PATH = '../data/Dataset1.xlsx'` for local runs — no path changes needed.

### 4. Run in Google Colab
- Upload `Dataset1.xlsx` to `/content/` in Colab
- Change `DATA_PATH` in Section 1 from `'../data/Dataset1.xlsx'` to `'/content/Dataset1.xlsx'`
- Run all cells sequentially

---

## Key Findings

- **Random Forest** is the recommended production model — AUC = 0.93, Recall = 0.71, F1 = 0.68
- **Physics-based feature engineering** improved Random Forest F1 by +13 points and Precision by +17 points vs original thesis results
- **Torque and Rotational Speed** are the most important predictors across tree-based models
- **SMOTE** significantly improved minority class detection — training set expanded from 7,000 to 13,474 samples
- **K-Means clustering** (k=2) revealed a high-risk operating regime with 5.66% failure rate vs 1.35% in the low-risk cluster — a 4.2× difference
- **Threshold calibration** on XGBoost achieves Recall = 0.86 by lowering the decision threshold — suitable for recall-critical deployment

---

## Industry Connection

This project extends 5 years of industrial data analytics experience at **John Deere Technology Center (Pune)**, including:

- IoT telemetry pipelines processing tens of GBs of CAN bus and sensor data from field-deployed agricultural sprayers
- Engine oil prognosis algorithms achieving 90% accuracy vs SME assessments — shifting from reactive to predictive maintenance
- Duty cycle analytics across 6–10 sprayer models using Power, RPM, and torque signals

The feature engineering approach here — deriving `Power_W` from torque and RPM — directly mirrors the signal processing methodology used in real John Deere telemetry analytics.

---

## Academic Context

| Item | Detail |
|---|---|
| Degree | MSc Data Analytics — First Class Honours (1:1) |
| Institution | Dublin Business School, Ireland |
| Module | B9DA113 Applied Research Project |
| Supervisor | Sajal Kaur Minhas |
| Submission | August 2025 |

---

## License

This project is shared for portfolio and educational purposes. The AI4I 2020 dataset is publicly available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) under its original license.

---

*Part of my data analytics portfolio — [github.com/avinashd144](https://github.com/avinashd144)*

---

## Author

**Avinash Desai** — Data Analyst | PL-300 Certified | MSc Data Analytics (1:1) DBS

🔗 [LinkedIn](https://www.linkedin.com/in/avinash-desai4/)  
📧 desaiavinash93@gmail.com
