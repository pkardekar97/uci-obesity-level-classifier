from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def compute_metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    multi = len(set(y)) > 2
    avg = "weighted" if multi else "binary"

    if multi:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted")
    else:
        auc = roc_auc_score(y, proba[:, 1])

    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average=avg, zero_division=0),
        "Recall": recall_score(y, pred, average=avg, zero_division=0),
        "F1": f1_score(y, pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }