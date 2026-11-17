"""Chọn metric + tính metric trên tập test.

Lý do chọn metric (viết vào báo cáo):
- Accuracy dễ "đánh lừa" khi imbalance (đoán toàn lớp đa số cũng được accuracy cao) -> chỉ báo cáo kèm, không dùng chọn mô hình.
- Nhị phân + imbalance (lớp dương < 30%): metric chính = Average Precision (diện tích PR curve),
  vì PR curve tập trung vào lớp dương, nhạy hơn ROC-AUC khi lớp dương hiếm. Kèm Precision/Recall/F1.
- Nhị phân cân bằng: ROC-AUC.
- Đa lớp: macro-F1 (mỗi lớp nặng như nhau, không bị lớp lớn lấn át) + balanced accuracy.
"""
import warnings

import numpy as np
from sklearn import metrics as skm
from sklearn.exceptions import UndefinedMetricWarning

import config

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def primary_metric(task: str, minority_ratio: float) -> str:
    if config.PRIMARY_METRIC != "auto":
        return config.PRIMARY_METRIC
    if task == "binary":
        return "average_precision" if minority_ratio < 0.30 else "roc_auc"
    return "f1_macro"


def cv_scoring(task: str) -> dict:
    """Dict scorer dùng cho cross_validate (tên chuẩn của sklearn)."""
    if task == "binary":
        return {k: k for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1",
                               "roc_auc", "average_precision"]}
    return {k: k for k in ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
                           "f1_macro", "f1_weighted"]}


def get_scores(model, X):
    """Điểm liên tục cho ROC/PR: ưu tiên xác suất, nếu không có (SVC) dùng decision_function."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.shape[1] == 2 else p
    return model.decision_function(X)


def evaluate(model, X, y, task: str) -> dict:
    pred = model.predict(X)
    out = {
        "accuracy": skm.accuracy_score(y, pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y, pred),
    }
    if task == "binary":
        s = get_scores(model, X)
        out.update({
            "precision": skm.precision_score(y, pred, zero_division=0),
            "recall": skm.recall_score(y, pred, zero_division=0),
            "f1": skm.f1_score(y, pred, zero_division=0),
            "mcc": skm.matthews_corrcoef(y, pred),
            "roc_auc": skm.roc_auc_score(y, s),
            "average_precision": skm.average_precision_score(y, s),
        })
    else:
        out.update({
            "precision_macro": skm.precision_score(y, pred, average="macro", zero_division=0),
            "recall_macro": skm.recall_score(y, pred, average="macro", zero_division=0),
            "f1_macro": skm.f1_score(y, pred, average="macro", zero_division=0),
            "f1_weighted": skm.f1_score(y, pred, average="weighted", zero_division=0),
            "mcc": skm.matthews_corrcoef(y, pred),
        })
    return {k: float(v) for k, v in out.items()}


def score_from_preds(metric: str, y, pred, s) -> float:
    """Tính metric chính từ dự đoán có sẵn (dùng cho bootstrap, đỡ phải predict lại)."""
    if metric == "average_precision":
        return skm.average_precision_score(y, s)
    if metric == "roc_auc":
        return skm.roc_auc_score(y, s)
    if metric == "accuracy":
        return skm.accuracy_score(y, pred)
    if metric == "balanced_accuracy":
        return skm.balanced_accuracy_score(y, pred)
    if metric in ("precision", "recall", "f1"):
        return getattr(skm, f"{metric}_score")(y, pred, zero_division=0)
    if metric.endswith("_macro") or metric == "f1_weighted":
        avg = "macro" if metric.endswith("_macro") else "weighted"
        fn = {"f1": skm.f1_score, "precision": skm.precision_score, "recall": skm.recall_score}[metric.split("_")[0]]
        return fn(y, pred, average=avg, zero_division=0)
    return skm.f1_score(y, pred, average="macro", zero_division=0)


def sort_key_metric_cols(task: str):
    """Thứ tự cột hiển thị trong bảng."""
    if task == "binary":
        return ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "average_precision"]
    return ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "mcc"]
