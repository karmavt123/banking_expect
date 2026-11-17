"""Bước 2 - So sánh các mô hình với THAM SỐ MẶC ĐỊNH bằng Stratified K-Fold CV trên tập train.
Đây là baseline để sau đó đo tuning/ensemble có thực sự cải thiện không."""
import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate

import config
from metrics import cv_scoring, primary_metric
from models import PRETTY, active_models, make_pipeline
from utils import cv_splitter, save_csv, save_fig


def scoring_with_primary(task, primary):
    sc = cv_scoring(task)
    sc.setdefault(primary, primary)
    return sc


def run(ds):
    print("== BENCHMARK (tham số mặc định, CV) ==")
    primary = primary_metric(ds.task, ds.info["minority_ratio"])
    print(f"  metric chính: {primary}")
    scoring = scoring_with_primary(ds.task, primary)
    rows, folds = [], {}
    for name in active_models(len(ds.y_train)):
        t0 = time.time()
        pipe = make_pipeline(name, ds)
        res = cross_validate(pipe, ds.X_train, ds.y_train, cv=cv_splitter(), scoring=scoring,
                             n_jobs=config.N_JOBS, return_train_score=True)
        row = {"model": PRETTY[name], "key": name}
        for m in scoring:
            row[f"{m}_mean"] = res[f"test_{m}"].mean()
            row[f"{m}_std"] = res[f"test_{m}"].std()
        row[f"train_{primary}"] = res[f"train_{primary}"].mean()   # so với CV để thấy overfitting
        row["fit_time_s"] = res["fit_time"].mean()
        rows.append(row)
        folds[PRETTY[name]] = res[f"test_{primary}"]
        print(f"  {PRETTY[name]:<28} {primary}={row[f'{primary}_mean']:.4f} ± {row[f'{primary}_std']:.4f} "
              f"(train {row[f'train_{primary}']:.4f})  [{time.time() - t0:.0f}s]")
    df = pd.DataFrame(rows)
    df.insert(0, "primary_metric", primary)
    save_csv(df, "benchmark_cv.csv")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.boxplot(list(folds.values()))
    ax.set_xticks(range(1, len(folds) + 1)); ax.set_xticklabels(list(folds.keys()))
    ax.set_ylabel(primary); ax.set_title(f"CV {config.CV_FOLDS}-fold: {primary} (tham số mặc định)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    save_fig(fig, "benchmark_cv_boxplot.png")
    return df
