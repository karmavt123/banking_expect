"""Bước 5 - Ensemble: Soft Voting và Stacking trên top-K mô hình đã tuning (loại SVM vì không có predict_proba)."""
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

import config
from metrics import primary_metric
from models import HAS_PROBA, PRETTY
from utils import cv_splitter, read_csv_if_exists, save_csv


def run(ds):
    print("== ENSEMBLE ==")
    primary = primary_metric(ds.task, ds.info["minority_ratio"])
    tun = read_csv_if_exists("tuning_summary.csv")
    if tun is None:
        raise RuntimeError("Cần chạy bước 'tune' trước khi chạy ensemble.")
    cand = tun[tun.key.isin(HAS_PROBA)].sort_values("tuned_cv", ascending=False).head(config.ENSEMBLE_TOP_K)
    if len(cand) < 2:
        print("  Không đủ mô hình có predict_proba để ensemble, bỏ qua.")
        return None
    members = [(k, clone(joblib.load(config.MODEL_DIR / f"{k}_tuned.joblib"))) for k in cand.key]
    print("  thành viên:", ", ".join(PRETTY[k] for k, _ in members))

    cw = "balanced" if config.USE_CLASS_WEIGHT else None
    ensembles = {
        "voting": VotingClassifier(members, voting="soft", n_jobs=1),
        "stacking": StackingClassifier(members, final_estimator=LogisticRegression(max_iter=3000, class_weight=cw),
                                       cv=3, n_jobs=1),
    }
    rows = []
    for name, ens in ensembles.items():
        res = cross_validate(ens, ds.X_train, ds.y_train, cv=cv_splitter(), scoring=primary, n_jobs=config.N_JOBS)
        sc = res["test_score"]
        ens.fit(ds.X_train, ds.y_train)
        joblib.dump(ens, config.MODEL_DIR / f"{name}.joblib")
        rows.append({"model": f"Ensemble: {name}", "key": name, "metric": primary, "cv_mean": sc.mean(),
                     "cv_std": sc.std(), "members": " + ".join(PRETTY[k] for k, _ in members)})
        print(f"  {name:<9} {primary}={sc.mean():.4f} ± {sc.std():.4f}")
    df = pd.DataFrame(rows)
    save_csv(df, "ensemble_cv.csv")
    return df
