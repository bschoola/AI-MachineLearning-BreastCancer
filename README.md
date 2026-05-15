# 🎗️ Tech Challenge — Predictive Analysis of Breast Cancer

Breast tumor classification project — **benign or malignant** — using the public **Breast Cancer Wisconsin** dataset, with complete exploratory analysis and comparison of four machine learning models.

---

## 👥 Team

| Name |
|---|
| [Bruno Gouveia Schoola](https://github.com/bschoola) |
| [Ricardo Stebulaitis](https://github.com/stebulaitis) |

---

## 🎯 Objective

Identify the most relevant variables in the dataset and determine which classification model performs best at distinguishing between benign and malignant tumors, with special focus on the **Recall** metric — minimizing false negatives, which represent the greatest clinical risk.

---

## 📦 Dataset

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** [Kaggle / UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- **Samples:** 569 patients
- **Features:** 30 numerical variables (mean, standard error, and worst value of 10 cellular characteristics)
- **Target:** `diagnosis` — Benign (B) or Malignant (M)
- **Class imbalance:** ~63% Benign / ~37% Malignant (mild)

---

## 🗂️ Notebook Structure

```
1. Data Loading
   ├── Library imports
   ├── CSV loading via GitHub
   ├── Null value check
   └── Column translation to Portuguese

2. Exploratory Data Analysis (EDA)
   ├── 2.1 General Analysis
   │   ├── Shape and data types
   │   ├── Removal of irrelevant column (Unnamed: 32)
   │   └── Class balance check
   ├── 2.2 Univariate Analysis
   │   ├── Distribution of mean variables
   │   ├── Comparison by diagnosis (boxplots)
   │   ├── Feature variability
   │   ├── Standard error analysis
   │   └── Correlation matrix
   ├── 2.3 Bivariate Analysis
   │   ├── Percentage difference between classes
   │   ├── T-tests for statistical significance
   │   ├── Correlation with target
   │   ├── Feature combinations
   │   └── Pairplot of top features
   └── 2.4 Multivariate Analysis
       ├── Feature Importance (Random Forest)
       ├── Variable reduction test (cross-val)
       └── Class separability via PCA

3. Modeling
   ├── 3.1 Setup (stratified train/test split — 80/20)
   └── 3.2 Training and Evaluation
       ├── Logistic Regression (Pipeline + StandardScaler)
       ├── Random Forest (200 estimators)
       ├── Gradient Boosting
       ├── KNN (k=5, Pipeline + StandardScaler)
       ├── 10-fold cross-validation (Recall and F1)
       ├── SHAP analysis (explainability)
       └── Feature Importance via coefficients
```

---

## 🔬 Selected Features

After univariate, bivariate, and multivariate analyses, the 4 variables chosen for modeling were:

| Feature | Description |
|---|---|
| `area_pior` | Worst value of tumor area |
| `textura_pior` | Worst value of texture (standard deviation of gray-scale values) |
| `pontos_concavos_pior` | Worst value of number of concave points on the contour |
| `concavidade_pior` | Worst value of severity of concavities in the contour |

> Reducing from 30 to 4 variables maintained equivalent performance to the full model, eliminating noise and redundancies (especially among `radius`, `perimeter`, and `area`).

---

## 🤖 Tested Models

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ✅ Best | ✅ | ✅ Best | ✅ Best | ✅ |
| Random Forest | High | High | High | High | High |
| Gradient Boosting | High | High | High | High | High |
| KNN (k=5) | Good | Good | Good | Good | Good |

> Exact values are generated upon notebook execution. Refer to the output cells for precise numbers.

---

## 🏆 Chosen Model: Logistic Regression

**Logistic Regression** achieved the best performance in **Recall** and **F1-Score** during 10-fold cross-validation.

**Why Recall was chosen as the primary metric:**
- A **False Negative** (classifying a malignant tumor as benign) poses a direct risk to the patient's life by delaying diagnosis and treatment.
- In a hospital triage system, F1-Score is also relevant as it balances Precision and Recall.

**Explainability (SHAP):**
- The variable `area_pior` has the greatest impact on model predictions.
- SHAP was applied for individual analysis of benign and malignant cases.

---

## 🛠️ Technologies and Libraries

```python
pandas / numpy           # Data manipulation
matplotlib / seaborn     # Visualization
scipy                    # Statistical tests (T-test)
scikit-learn             # Models, Pipeline, PCA, metrics, cross-validation
shap                     # Model explainability
```

---

## 🚀 How to Run

The project is available on Google Colab, with no local setup required:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/bschoola/075cb2170450074136e6a299974a5967/1-fiap_techchallenge.ipynb)

**Steps:**
1. Click the badge above to open the notebook
2. Click **"Run all"** (`Runtime > Run all`)
3. The dataset loads automatically via a public GitHub URL

---

## 🎬 Video Presentation

The full project walkthrough is available on YouTube:

🔗 [Watch on YouTube](https://www.youtube.com/watch?v=4VWrv6xcAyE&list=PLAVmCKF3axITmuobDyZf9BevKBgzBjyEb)

---

## 📊 Key Findings

- Malignant tumors tend to be **larger** and show **greater structural irregularity**
- Variables from the `_worst` group are more discriminative than those from the `_mean` group
- **Standard error** variables (`_se`) have low predictive power
- Strong **redundancy** exists among `radius`, `perimeter`, and `area` — only one representative is needed
- The problem is **linearly separable** with the 4 selected features (confirmed via PCA)
- **Logistic Regression** outperformed more complex models on this dataset
