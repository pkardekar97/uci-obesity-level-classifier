import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from metrics_utils import compute_metrics

TARGET = "NObeyesdad"
CLASS_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]
DROP_COLS = ["Height", "Weight"]

# Load data and remove duplicates
df = pd.read_csv("data/ObesityDataSet_raw_and_data_sinthetic.csv")
df = df.drop_duplicates()
print(f"Total instances after removing duplicates: {len(df)}")

X = df.drop(columns=[TARGET])
y = df[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)})
assert y.notna().all(), "Unmapped target classes detected."

feature_cols = [c for c in X.columns if c not in DROP_COLS]
numeric_features = X[feature_cols].select_dtypes("number").columns.tolist()
categorical_features = [c for c in feature_cols if c not in numeric_features]

def build_pipeline(classifier):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])
    return Pipeline([("prep", preprocessor), ("clf", classifier)])

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Export root test_data.csv for Streamlit upload
test_export_df = X_test.copy()
test_export_df[TARGET] = y_test.map(dict(enumerate(CLASS_ORDER))).values
test_export_df.to_csv("test_data.csv", index=False)
print("Saved root test_data.csv successfully.")

model_definitions = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000),
        {"clf__C": [0.1, 1, 10, 100]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"clf__max_depth": [4, 6, 8, 12, None], "clf__min_samples_leaf": [1, 5]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [3, 5, 7, 11]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-1]},
    ),
    "SVM": (
        SVC(probability=True, random_state=42),
        {"clf__C": [1, 10, 50], "clf__gamma": ["scale", 0.05]},
    ),
}

os.makedirs("model", exist_ok=True)
metric_records = []

for name, (classifier, grid) in model_definitions.items():
    print(f"Training {name}...")
    search = GridSearchCV(build_pipeline(classifier), grid, cv=5, scoring="f1_weighted", n_jobs=-1)
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_

    metrics = compute_metrics(best_pipe, X_test, y_test)
    metrics["Train Acc"] = best_pipe.score(X_train, y_train)
    metrics["Best Params"] = str(search.best_params_)
    metric_records.append({"Model": name, **metrics})

    slug = name.replace(" ", "_").lower()
    joblib.dump(best_pipe, f"model/{slug}.joblib")
    print(f"Done: {name}")

joblib.dump({"classes": CLASS_ORDER, "target": TARGET}, "model/meta.joblib")

summary_df = pd.DataFrame(metric_records).set_index("Model")
summary_df.to_csv("results.csv")
print("\n=== Model Evaluation Summary on Held-Out Test Set ===")
print(summary_df.round(4))