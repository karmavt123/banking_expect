"""Bước 3 - Tối ưu siêu tham số bằng RandomizedSearchCV (CV chỉ trên tập train).
Lưu ý khi viết báo cáo: điểm CV tốt nhất sau tuning hơi "lạc quan" vì chính CV đó được dùng để chọn tham số;
con số khách quan là điểm trên tập test ở bước final."""
import json
import time

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

import config
from metrics import primary_metric
from models import PRETTY, active_models, make_pipeline, param_space
from utils import cv_splitter, read_csv_if_exists, save_csv


def run(ds):
    print("== TUNING (RandomizedSearchCV) ==")
    primary = primary_metric(ds.task, ds.info["minority_ratio"])
    bench = read_csv_if_exists("benchmark_cv.csv")
    rows, best = [], {}
    for name in active_models(len(ds.y_train)):
        t0 = time.time()
        pipe = make_pipeline(name, ds)
        search = RandomizedSearchCV(pipe, param_space(name), n_iter=config.N_ITER_SEARCH, scoring=primary,
                                    cv=cv_splitter(), n_jobs=config.N_JOBS, random_state=config.SEED,
                                    refit=True, error_score="raise")
        search.fit(ds.X_train, ds.y_train)
        joblib.dump(search.best_estimator_, config.MODEL_DIR / f"{name}_tuned.joblib")
        base = None
        if bench is not None and f"{primary}_mean" in bench:
            r = bench[bench.key == name]
            base = float(r[f"{primary}_mean"].iloc[0]) if len(r) else None
        params = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                  for k, v in search.best_params_.items()}
        best[name] = {k: str(v) for k, v in params.items()}
        rows.append({"model": PRETTY[name], "key": name, "metric": primary,
                     "default_cv": base, "tuned_cv": float(search.best_score_),
                     "gain": (float(search.best_score_) - base) if base is not None else None,
                     "best_params": json.dumps(best[name], ensure_ascii=False)})
        print(f"  {PRETTY[name]:<28} default={base if base is None else round(base, 4)} -> "
              f"tuned={search.best_score_:.4f}  [{time.time() - t0:.0f}s]")
    df = pd.DataFrame(rows)
    save_csv(df, "tuning_summary.csv")
    (config.RESULTS_DIR / "best_params.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    return df
