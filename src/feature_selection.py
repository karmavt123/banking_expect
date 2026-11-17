"""Bước 4 - Thí nghiệm lựa chọn đặc trưng (feature selection) nằm TRONG pipeline (không leakage).
So sánh: giữ hết đặc trưng  vs  lọc theo ANOVA-F (top p%)  vs  L1 (LinearSVC)  vs  độ quan trọng của Random Forest."""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config
from metrics import primary_metric
from models import PRETTY
from utils import cv_splitter, read_csv_if_exists, save_csv, save_fig


def _pick_reference(ds):
    tun = read_csv_if_exists("tuning_summary.csv")
    if tun is None:
        raise RuntimeError("Cần chạy bước 'tune' trước khi chạy feature selection.")
    if config.FS_MODEL != "auto":
        return config.FS_MODEL
    return tun.sort_values("tuned_cv", ascending=False).key.iloc[0]


def run(ds):
    print("== FEATURE SELECTION ==")
    primary = primary_metric(ds.task, ds.info["minority_ratio"])
    ref = _pick_reference(ds)
    tuned = joblib.load(config.MODEL_DIR / f"{ref}_tuned.joblib")
    print(f"  mô hình tham chiếu: {PRETTY[ref]} (tham số đã tuning)")
    cw = "balanced" if config.USE_CLASS_WEIGHT else None
    s = config.SEED

    selectors = {"Không chọn (tất cả đặc trưng)": None}
    for p in config.FS_PERCENTILES:
        if p < 100:
            selectors[f"ANOVA-F top {p}%"] = lambda p=p: SelectPercentile(f_classif, percentile=p)
    selectors["L1 (LinearSVC)"] = lambda: SelectFromModel(
        LinearSVC(C=0.05, penalty="l1", dual=False, class_weight=cw, max_iter=5000, random_state=s))
    selectors["RandomForest importance (>= median)"] = lambda: SelectFromModel(
        RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=s, n_jobs=1), threshold="median")

    rows = []
    for label, make_sel in selectors.items():
        steps = [("prep", clone(tuned.named_steps["prep"]))]
        if make_sel is not None:
            steps.append(("select", make_sel() if callable(make_sel) else make_sel))
        steps.append(("clf", clone(tuned.named_steps["clf"])))
        pipe = Pipeline(steps)
        res = cross_validate(pipe, ds.X_train, ds.y_train, cv=cv_splitter(), scoring=primary, n_jobs=config.N_JOBS)
        sc = res["test_score"]
        pipe.fit(ds.X_train, ds.y_train)   # fit 1 lần để đếm số đặc trưng được giữ
        n_total = len(pipe.named_steps["prep"].get_feature_names_out())
        n_keep = int(pipe.named_steps["select"].get_support().sum()) if "select" in pipe.named_steps else n_total
        rows.append({"variant": label, "n_features_kept": n_keep, "n_features_total": n_total,
                     f"{primary}_mean": sc.mean(), f"{primary}_std": sc.std()})
        print(f"  {label:<38} {n_keep:>4}/{n_total:<4} {primary}={sc.mean():.4f} ± {sc.std():.4f}")
    df = pd.DataFrame(rows)
    df.insert(0, "model", PRETTY[ref])
    save_csv(df, "feature_selection.csv")

    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(df) + 1.5))
    d = df.iloc[::-1]
    ax.barh(d["variant"], d[f"{primary}_mean"], xerr=d[f"{primary}_std"], color="#F58518", capsize=3)
    lo = (d[f"{primary}_mean"] - d[f"{primary}_std"]).min()
    ax.set_xlim(max(0, lo - 0.03), min(1, d[f"{primary}_mean"].max() + d[f"{primary}_std"].max() + 0.02))
    ax.set_xlabel(f"{primary} (CV)"); ax.set_title(f"Feature selection - {PRETTY[ref]}")
    save_fig(fig, "feature_selection.png")
    return df
