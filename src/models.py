"""Kho mô hình + không gian tìm tham số. Mỗi mô hình được bọc trong Pipeline(prep -> clf)."""
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
from preprocess import build_preprocessor

PRETTY = {
    "logreg": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "hist_gb": "Gradient Boosting (HistGB)",
    "svm_rbf": "SVM (RBF)",
    "mlp": "MLP (Neural Net)",
}

# Mô hình có predict_proba (dùng được cho soft-voting / stacking)
HAS_PROBA = {"logreg", "decision_tree", "random_forest", "hist_gb", "mlp"}


def _registry():
    cw = "balanced" if config.USE_CLASS_WEIGHT else None
    s = config.SEED
    # (estimator mặc định, không gian tham số cho RandomizedSearchCV; tiền tố "clf__")
    return {
        "logreg": (
            LogisticRegression(max_iter=3000, class_weight=cw, random_state=s),
            {"clf__C": loguniform(1e-3, 1e2)},
        ),
        "decision_tree": (
            DecisionTreeClassifier(class_weight=cw, random_state=s),
            {"clf__max_depth": [3, 4, 5, 7, 10, 15, None],
             "clf__min_samples_leaf": randint(1, 60),
             "clf__min_samples_split": randint(2, 60),
             "clf__criterion": ["gini", "entropy"],
             "clf__ccp_alpha": uniform(0, 0.01)},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=s, n_jobs=1),
            {"clf__n_estimators": randint(100, 400),
             "clf__max_depth": [None, 6, 10, 15, 25],
             "clf__min_samples_leaf": randint(1, 25),
             "clf__max_features": ["sqrt", "log2", 0.5]},
        ),
        "hist_gb": (
            HistGradientBoostingClassifier(class_weight=cw, random_state=s),
            {"clf__learning_rate": loguniform(0.01, 0.3),
             "clf__max_leaf_nodes": randint(8, 64),
             "clf__max_depth": [None, 3, 5, 8],
             "clf__min_samples_leaf": randint(10, 100),
             "clf__l2_regularization": loguniform(1e-3, 10),
             "clf__max_iter": randint(100, 400)},
        ),
        "svm_rbf": (
            SVC(kernel="rbf", class_weight=cw, random_state=s),
            {"clf__C": loguniform(0.1, 100), "clf__gamma": loguniform(1e-4, 1e-1)},
        ),
        "mlp": (  # MLPClassifier không có class_weight -> ghi rõ trong báo cáo nếu dữ liệu imbalance
            MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, max_iter=300, random_state=s),
            {"clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32)],
             "clf__alpha": loguniform(1e-5, 1e-1),
             "clf__learning_rate_init": loguniform(1e-4, 1e-2)},
        ),
    }


def active_models(n_train: int):
    """Danh sách tên mô hình thực sự chạy (tự bỏ SVM nếu data quá lớn)."""
    names = []
    for m in config.MODELS:
        if m == "svm_rbf" and n_train > config.SVM_MAX_ROWS:
            print(f"[models] Bỏ SVM-RBF vì n_train={n_train} > SVM_MAX_ROWS={config.SVM_MAX_ROWS} (SVM quá chậm). "
                  f"Ghi chú lý do này vào báo cáo.")
            continue
        names.append(m)
    return names


def make_pipeline(name: str, ds, selector=None, params=None) -> Pipeline:
    """Pipeline(prep [-> select] -> clf). `params` là dict tham số đã tuning (khoá dạng 'clf__x')."""
    est, _ = _registry()[name]
    from sklearn.base import clone
    est = clone(est)
    steps = [("prep", build_preprocessor(ds.num_cols, ds.cat_cols))]
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("clf", est))
    pipe = Pipeline(steps)
    if params:
        pipe.set_params(**params)
    return pipe


def param_space(name: str):
    return _registry()[name][1]
