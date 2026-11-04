\# Multi-Class Obesity Risk Level Classifier



\*\*Course:\*\* BITS Pilani M.Tech AI/ML — Machine Learning (Assignment 1)  

\*\*Student Name:\*\* Pranay Kardekar  

\*\*BITS ID:\*\* 2026AK05670  

\*\*Repository:\*\* \[https://github.com/pkardekar97/uci-obesity-level-classifier](https://github.com/pkardekar97/uci-obesity-level-classifier)  

\*\*Live Application URL:\*\* \[https://pranaykardekar-2026ak05670-obesity-classifier.streamlit.app/](https://pranaykardekar-2026ak05670-obesity-classifier.streamlit.app/)  



\---



\## 1. Problem Statement \& Dataset Overview



This project implements an end-to-end Machine Learning pipeline to classify individuals into 7 distinct obesity-related categories based on eating habits, demographic profile, and physical lifestyle patterns.



\* \*\*Data Source:\*\* \[UCI Machine Learning Repository — Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://archive.icsuci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

\* \*\*Instance Count:\*\* 2,087 unique patient records (24 duplicate records removed during data cleaning).

\* \*\*Target Classes (7 levels):\*\*

&#x20; 1. `Insufficient\_Weight`

&#x20; 2. `Normal\_Weight`

&#x20; 3. `Overweight\_Level\_I`

&#x20; 4. `Overweight\_Level\_II`

&#x20; 5. `Obesity\_Type\_I`

&#x20; 6. `Obesity\_Type\_II`

&#x20; 7. `Obesity\_Type\_III`



\### Preventing Target Leakage

In clinical standards, Body Mass Index (BMI) is mathematically defined as:

$$\\text{BMI} = \\frac{\\text{Weight (kg)}}{\[\\text{Height (m)}]^2}$$

Because the target classes are derived directly from clinical BMI thresholds, retaining `Weight` and `Height` features results in trivial target leakage. To construct a legitimate diagnostic model assessing lifestyle and demographic factors, \*\*`Height` and `Weight` were deliberately removed from the feature set\*\*, leaving 14 predictive features.



\---



\## 2. Feature Schema \& Pipeline Preprocessing



The 14 behavioral and demographic features used are:

\* \*\*Continuous / Discrete Numerical (6):\*\* `Age`, `FCVC` (vegetable intake frequency), `NCP` (main meals/day), `CH2O` (daily water consumption), `FAF` (physical activity frequency), `TUE` (time using tech devices).

\* \*\*Categorical / Nominal (8):\*\* `Gender`, `family\_history\_with\_overweight`, `FAVC` (high caloric food), `CAEC` (food between meals), `SMOKE`, `SCC` (calorie monitoring), `CALC` (alcohol consumption), `MTRANS` (mode of transportation).



\### Scikit-Learn Preprocessing Pipeline

To guarantee zero data contamination between splits:

1\. \*\*Partitioning:\*\* Stratified 80/20 train-test split (`test\_size=0.20`, `random\_state=42`, `stratify=y`).

2\. \*\*Numeric Pipeline:\*\* Imputation (`SimpleImputer(strategy='median')`) followed by `StandardScaler`.

3\. \*\*Categorical Pipeline:\*\* Imputation (`SimpleImputer(strategy='most\_frequent')`) followed by `OneHotEncoder(handle\_unknown='ignore', sparse\_output=False)`.

4\. \*\*ColumnTransformer:\*\* Assembles feature pipelines, fitted strictly on the training partition and applied downstream to test instances.



\---



\## 3. Comparative Model Performance



All 5 classifiers were evaluated on the held-out test split ($N = 418$) across 6 multi-class evaluation metrics:



| Model Classifier | Accuracy | AUC (OvR) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | MCC (Multi-Class) |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: |

| \*\*Support Vector Machine (RBF)\*\* | \*\*0.8038\*\* | \*\*0.9575\*\* | \*\*0.8031\*\* | \*\*0.8038\*\* | \*\*0.7997\*\* | \*\*0.7716\*\* |

| \*\*K-Nearest Neighbors ($k=7$)\*\* | 0.7775 | 0.9327 | 0.7788 | 0.7775 | 0.7754 | 0.7410 |

| \*\*Logistic Regression (L2)\*\* | 0.7321 | 0.9388 | 0.7303 | 0.7321 | 0.7282 | 0.6868 |

| \*\*Decision Tree ($d=8$)\*\* | 0.7201 | 0.8967 | 0.7208 | 0.7201 | 0.7185 | 0.6738 |

| \*\*Gaussian Naive Bayes\*\* | 0.6124 | 0.8524 | 0.6119 | 0.6124 | 0.5898 | 0.5511 |



\*Note: Multi-class ROC-AUC uses the One-vs-Rest (OvR) formulation with macro averaging. Multi-class MCC uses Gorodkin’s formulation.\*



\---



\## 4. Repository Structure



```text

uci-obesity-level-classifier/

├── data/

│   └── raw\_obesity\_data.csv       # Sourced raw UCI dataset

├── model/

│   ├── meta.joblib                # Target labels \& metadata

│   ├── logistic\_regression.joblib # Serialized model pipeline

│   ├── decision\_tree.joblib       # Serialized model pipeline

│   ├── knn.joblib                 # Serialized model pipeline

│   ├── naive\_bayes.joblib         # Serialized model pipeline

│   └── svm.joblib                 # Serialized model pipeline

├── metrics\_utils.py               # Shared 6-metric multi-class computation module

├── train\_models.py                # Cleaning, pipeline construction \& training script

├── test\_data.csv                  # Held-out 20% test split with ground truth labels

├── results.csv                    # Benchmarked model comparison matrix

├── app.py                         # Interactive Streamlit evaluation application

├── requirements.txt               # Deployment dependencies

└── README.md                      # Comprehensive project documentation
```

\---



\## 5. Local Setup and Reproduction

Clone repository:

git clone https://github.com/pkardekar97/uci-obesity-level-classifier.git

cd uci-obesity-level-classifier



Create and activate virtual environment:

python -m venv venv

.\\venv\\Scripts\\activate



Install required packages:

pip install -r requirements.txt



Retrain models \& regenerate results:

python train\_models.py



Launch interactive Streamlit dashboard:

streamlit run app.py

