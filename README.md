# Multi-Class Obesity Risk Level Classifier

**Course:** BITS Pilani M.Tech AI/ML — Machine Learning (Assignment 1)  
**Student Name:** Pranay Kardekar  
**BITS ID:** 2026AK05670  
**Repository:** https://github.com/pkardekar97/uci-obesity-level-classifier  
**Live Application URL:** https://pranaykardekar-2026ak05670-obesity-classifier.streamlit.app/  

---

## 1. Problem Statement & Dataset Overview

This project implements an end-to-end Machine Learning pipeline to classify individuals into 7 distinct obesity-related categories based on eating habits, demographic profile, and physical lifestyle patterns.

* **Data Source:** UCI Machine Learning Repository — Estimation of Obesity Levels Based On Eating Habits and Physical Condition
* **Instance Count:** 2,087 unique patient records (24 duplicate records removed during data cleaning).
* **Target Classes (7 levels):**
  1. Insufficient_Weight
  2. Normal_Weight
  3. Overweight_Level_I
  4. Overweight_Level_II
  5. Obesity_Type_I
  6. Obesity_Type_II
  7. Obesity_Type_III

### Preventing Target Leakage
In clinical standards, Body Mass Index (BMI) is mathematically defined as:
BMI = Weight (kg) / [Height (m)]^2

Because target classes are derived directly from clinical BMI thresholds, retaining Weight and Height features results in trivial target leakage. To construct a legitimate diagnostic model assessing lifestyle and demographic factors, **Height and Weight were deliberately removed from the feature set**, leaving 14 predictive features.

---

## 2. Feature Schema & Pipeline Preprocessing

The 14 behavioral and demographic features used are:
* **Continuous / Discrete Numerical (6):** Age, FCVC (vegetable intake frequency), NCP (main meals/day), CH2O (daily water consumption), FAF (physical activity frequency), TUE (time using tech devices).
* **Categorical / Nominal (8):** Gender, family_history_with_overweight, FAVC (high caloric food), CAEC (food between meals), SMOKE, SCC (calorie monitoring), CALC (alcohol consumption), MTRANS (mode of transportation).

### Scikit-Learn Preprocessing Pipeline
To guarantee zero data contamination between splits:
1. **Partitioning:** Stratified 80/20 train-test split (test_size=0.20, random_state=42, stratify=y).
2. **Numeric Pipeline:** Imputation (SimpleImputer with median) followed by StandardScaler.
3. **Categorical Pipeline:** Imputation (SimpleImputer with most_frequent) followed by OneHotEncoder(handle_unknown='ignore', sparse_output=False).
4. **ColumnTransformer:** Assembles feature pipelines, fitted strictly on the training partition and applied downstream to test instances.

---

## 3. Comparative Model Performance

All 5 classifiers were evaluated on the held-out test split (N = 418) across 6 multi-class evaluation metrics:

| Model Classifier | Accuracy | AUC (OvR) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | MCC (Multi-Class) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** | **0.8038** | **0.9575** | **0.8031** | **0.8038** | **0.7997** | **0.7716** |
| **K-Nearest Neighbors (k=7)** | 0.7775 | 0.9327 | 0.7788 | 0.7775 | 0.7754 | 0.7410 |
| **Logistic Regression (L2)** | 0.7321 | 0.9388 | 0.7303 | 0.7321 | 0.7282 | 0.6868 |
| **Decision Tree (d=8)** | 0.7201 | 0.8967 | 0.7208 | 0.7201 | 0.7185 | 0.6738 |
| **Gaussian Naive Bayes** | 0.6124 | 0.8524 | 0.6119 | 0.6124 | 0.5898 | 0.5511 |

---

## 4. Repository Structure

```text
uci-obesity-level-classifier/
├── data/
│   └── raw_obesity_data.csv
├── model/
│   ├── meta.joblib
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
├── metrics_utils.py
├── train_models.py
├── test_data.csv
├── results.csv
├── app.py
├── requirements.txt
└── README.md

```

---
## 5. Local Setup and Reproduction

### 1. Clone repository

git clone https://github.com/pkardekar97/uci-obesity-level-classifier.git

cd uci-obesity-level-classifier

### 2. Create and activate virtual environment

python -m venv venv

.\venv\Scripts\activate

### 3. Install required packages

pip install -r requirements.txt

### 4. Retrain models & regenerate results

python train_models.py

### 5. Launch interactive Streamlit dashboard

streamlit run app.py
