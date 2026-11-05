# Multi-Class Obesity Risk Level Classifier

**Course:** Machine Learning (Assignment 2)
**Name:** Pranay Kardekar
**BITS ID:** 2026AK05670
**Github Repository Link:** https://github.com/pkardekar97/uci-obesity-level-classifier
**Live Application URL:** https://pranaykardekar-2026ak05670-obesity-classifier.streamlit.app/

---

## a. Problem Statement
The objective of this project is to build, evaluate, and deploy machine learning classification models to predict individuals' obesity risk levels based on their demographic attributes, dietary habits, and physical activity patterns. Obesity is a major multi-class health classification problem where early risk estimation can guide preventive clinical interventions.

---

## b. Dataset Description
- **Dataset:** Estimation of Obesity Levels Based On Eating Habits and Physical Condition (UCI Machine Learning Repository)
- **Instances:** 2,111 records
- **Features:** 16 predictive features (demographics, dietary habits, lifestyle behaviors)
- **Target Variable:** `NObeyesdad` (7 classes: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II, Obesity Type III)
- **Preprocessing:** Numerical features scaled via standard normalization; categorical variables transformed via one-hot encoding; train-test split evaluated under reproducible cross-validated hyperparameter tuning.

---

## c. Github Repository Link
https://github.com/pkardekar97/uci-obesity-level-classifier

---

## d. Models Used & Performance Comparison

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.6244 | 0.8828 | 0.6218 | 0.6244 | 0.6099 | 0.5653 |
| Decision Tree | 0.7488 | 0.8632 | 0.7495 | 0.7488 | 0.7461 | 0.7075 |
| KNN | 0.7584 | 0.9117 | 0.7548 | 0.7584 | 0.7485 | 0.7205 |
| Naive Bayes | 0.5431 | 0.8617 | 0.5978 | 0.5431 | 0.5132 | 0.4851 |
| Support Vector Machine | 0.8038 | 0.9605 | 0.8059 | 0.8038 | 0.8020 | 0.7720 |

### Model Observations & Winner Analysis

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Provides a modest baseline (Accuracy: 0.6244) but struggles to capture complex non-linear feature interactions between dietary intake, physical activity, and multi-class target boundaries. |
| Decision Tree | Captures hierarchical decision splits effectively with balanced Precision and Recall (~0.749), though slight overfitting is observed relative to its high training accuracy. |
| KNN | Demonstrates strong neighborhood separation across scaled continuous attributes, achieving the second highest accuracy (0.7584) and a solid AUC of 0.9117 with k=3. |
| Naive Bayes | Exhibits the lowest overall performance (Accuracy: 0.5431, MCC: 0.4851) due to strong conditional feature independence assumptions being violated by correlated lifestyle metrics. |
| Support Vector Machine | Outperforms all models across every evaluation metric, achieving the highest Accuracy (0.8038), AUC (0.9605), F1 (0.8020), and MCC (0.7720) through effective high-dimensional margin separation with an RBF kernel. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** is the definitive winner, providing superior generalization, the highest discriminatory power (AUC 0.9605), and top-tier multi-class balance across all metric categories. |
