"""Bước 6 - Đánh giá cuối trên tập TEST (chỉ chạm tập test ở bước này).
Mô hình "tốt nhất" được chọn theo điểm CV trên train (KHÔNG chọn theo test để tránh cherry-picking)."""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             classification_report, f1_score, precision_score, recall_score)
from sklearn.model_selection import cross_val_predict, learning_curve

import config
from metrics import evaluate, get_scores, primary_metric, score_from_preds, sort_key_metric_cols
from models import PRETTY, active_models, make_pipeline
from utils import cv_splitter, df_to_md, read_csv_if_exists, save_csv, save_fig


def _load_models(ds):
    """Trả về dict nhãn -> mô hình đã fit (default, tuned, ensemble)."""
    out = {}
    for name in active_models(len(ds.y_train)):
        out[f"{PRETTY[name]} (default)"] = make_pipeline(name, ds).fit(ds.X_train, ds.y_train)
        p = config.MODEL_DIR / f"{name}_tuned.joblib"
        if p.exists():
            out[f"{PRETTY[name]} (tuned)"] = joblib.load(p)
    for e in ("voting", "stacking"):
        p = config.MODEL_DIR / f"{e}.joblib"
        if p.exists():
            out[f"Ensemble: {e}"] = joblib.load(p)
    return out


def _best_by_cv(primary):
    """Chọn mô hình tốt nhất theo điểm CV (tuned + ensemble). Trả (nhãn, key mô hình đơn tốt nhất)."""
    cands = {}
    tun = read_csv_if_exists("tuning_summary.csv")
    if tun is not None:
        for _, r in tun.iterrows():
            cands[f"{r['model']} (tuned)"] = (r["tuned_cv"], r["key"])
    ens = read_csv_if_exists("ensemble_cv.csv")
    if ens is not None:
        for _, r in ens.iterrows():
            cands[r["model"]] = (r["cv_mean"], None)
    if not cands:
        return None, None
    best = max(cands, key=lambda k: cands[k][0])
    single = max((k for k, v in cands.items() if v[1]), key=lambda k: cands[k][0], default=None)
    return best, single


def run(ds):
    print("== FINAL (tập test) ==")
    primary = primary_metric(ds.task, ds.info["minority_ratio"])
    models = _load_models(ds)
    best_label, best_single = _best_by_cv(primary)

    # ---- bảng kết quả test ----
    rows, preds, scores = [], {}, {}
    for label, m in models.items():
        r = evaluate(m, ds.X_test, ds.y_test, ds.task)
        rows.append({"model": label, **r})
        preds[label] = m.predict(ds.X_test)
        scores[label] = get_scores(m, ds.X_test)
    cols = sort_key_metric_cols(ds.task)
    res = pd.DataFrame(rows)[["model"] + cols]
    res["chosen_by_cv"] = res["model"].eq(best_label)
    save_csv(res, "test_results.csv")
    shown = primary if primary in cols else cols[0]
    print(res[["model", shown]].to_string(index=False))
    print(f"  -> mô hình được chọn theo CV: {best_label}")

    # ---- biểu đồ default vs tuned ----
    metric_col = primary if primary in res.columns else cols[0]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = res["model"].tolist()
    colors = ["#9ecae1" if "(default)" in m else ("#e6550d" if "Ensemble" in m else "#3182bd") for m in order]
    ax.barh(order[::-1], res[metric_col].tolist()[::-1], color=colors[::-1])
    lo = res[metric_col].min()
    ax.set_xlim(max(0, lo - 0.05), min(1.0, res[metric_col].max() + 0.03))
    ax.set_xlabel(f"{metric_col} (test)"); ax.set_title("Kết quả trên tập test: default (nhạt) vs tuned vs ensemble (cam)")
    save_fig(fig, "test_comparison.png")

    # ---- ma trận nhầm lẫn ----
    show = [k for k in models if "(tuned)" in k or "Ensemble" in k] or list(models)
    n = len(show); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow), squeeze=False)
    for ax, k in zip(axes.ravel(), show):
        ConfusionMatrixDisplay.from_predictions(ds.y_test, preds[k], display_labels=ds.class_names, normalize="true",
                                                values_format=".2f", ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(k, fontsize=9); ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Ma trận nhầm lẫn (chuẩn hoá theo hàng = recall từng lớp)")
    save_fig(fig, "confusion_matrices.png")

    # ---- ROC / PR (nhị phân) ----
    if ds.task == "binary":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for k in show:
            RocCurveDisplay.from_predictions(ds.y_test, scores[k], name=k, ax=axes[0])
            PrecisionRecallDisplay.from_predictions(ds.y_test, scores[k], name=k, ax=axes[1])
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8); axes[0].set_title("ROC")
        axes[1].axhline(ds.y_test.mean(), color="k", ls="--", lw=0.8); axes[1].set_title("Precision-Recall (đường nét đứt = tỉ lệ lớp dương)")
        for a in axes:
            a.legend(fontsize=7)
        save_fig(fig, "roc_pr_curves.png")

    # ---- báo cáo theo lớp cho mô hình chọn ----
    if best_label in models:
        rep = classification_report(ds.y_test, preds[best_label], target_names=ds.class_names, digits=4, zero_division=0)
        (config.RESULTS_DIR / "classification_report_best.txt").write_text(f"{best_label}\n\n{rep}", encoding="utf-8")

    # ---- bootstrap khoảng tin cậy 95% của metric chính trên test ----
    rng = np.random.default_rng(config.SEED)
    y = ds.y_test
    B = 100 if config.QUICK else 300
    boot = {k: [] for k in show}
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < len(np.unique(y)):
            continue
        for k in show:
            boot[k].append(score_from_preds(primary, y[idx], preds[k][idx], scores[k][idx]))
    ci = pd.DataFrame([{"model": k, "metric": primary, "point": score_from_preds(primary, y, preds[k], scores[k]),
                        "ci95_low": np.percentile(v, 2.5), "ci95_high": np.percentile(v, 97.5)} for k, v in boot.items() if v])
    save_csv(ci, "bootstrap_ci.csv")

    # ---- phân tích ngưỡng (nhị phân): chọn ngưỡng tối ưu F1 bằng dự đoán out-of-fold trên TRAIN ----
    thr_df = None
    if ds.task == "binary" and best_label in models:
        use_label = best_single if best_single else best_label
        base = clone(models[use_label])
        method = "predict_proba" if hasattr(base, "predict_proba") else "decision_function"
        oof = cross_val_predict(base, ds.X_train, ds.y_train, cv=cv_splitter(), method=method, n_jobs=config.N_JOBS)
        oof = oof[:, 1] if oof.ndim == 2 else oof
        grid = np.quantile(oof, np.linspace(0.02, 0.98, 97))
        f1s = [f1_score(ds.y_train, (oof >= t).astype(int), zero_division=0) for t in grid]
        t_best = float(grid[int(np.argmax(f1s))])
        fitted = models[use_label]
        s_te = get_scores(fitted, ds.X_test)
        default_t = 0.5 if method == "predict_proba" else 0.0
        out = []
        for lab, t in (("ngưỡng mặc định", default_t), ("ngưỡng tối ưu F1 (chọn bằng CV trên train)", t_best)):
            p = (s_te >= t).astype(int)
            out.append({"model": use_label, "setting": lab, "threshold": t,
                        "precision": precision_score(ds.y_test, p, zero_division=0),
                        "recall": recall_score(ds.y_test, p, zero_division=0),
                        "f1": f1_score(ds.y_test, p, zero_division=0)})
        thr_df = pd.DataFrame(out)
        save_csv(thr_df, "threshold_analysis.csv")
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(grid, f1s, label="F1 (OOF trên train)")
        ax.axvline(t_best, color="r", ls="--", label=f"ngưỡng tối ưu = {t_best:.3f}")
        ax.set_xlabel("ngưỡng"); ax.set_ylabel("F1"); ax.set_title(f"Chọn ngưỡng - {use_label}"); ax.legend()
        save_fig(fig, "threshold_f1.png")

    # ---- learning curve + permutation importance cho mô hình đơn tốt nhất ----
    if best_single and best_single in models:
        m = models[best_single]
        sizes = np.linspace(0.2, 1.0, 4 if config.QUICK else 6)
        ts, tr, va = learning_curve(clone(m), ds.X_train, ds.y_train, train_sizes=sizes, cv=cv_splitter(),
                                    scoring=primary, n_jobs=config.N_JOBS, random_state=config.SEED)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for arr, lab, c in ((tr, "train", "#3182bd"), (va, "validation (CV)", "#e6550d")):
            ax.plot(ts, arr.mean(1), "o-", label=lab, color=c)
            ax.fill_between(ts, arr.mean(1) - arr.std(1), arr.mean(1) + arr.std(1), alpha=0.15, color=c)
        ax.set_xlabel("số mẫu train"); ax.set_ylabel(primary); ax.set_title(f"Learning curve - {best_single}"); ax.legend()
        save_fig(fig, "learning_curve.png")

        pi = permutation_importance(m, ds.X_test, ds.y_test, scoring=primary, n_repeats=3 if config.QUICK else 8,
                                    random_state=config.SEED, n_jobs=config.N_JOBS)
        imp = pd.DataFrame({"feature": ds.X_test.columns, "importance": pi.importances_mean, "std": pi.importances_std})
        imp = imp.sort_values("importance", ascending=False)
        save_csv(imp, "permutation_importance.csv")
        top = imp.head(15)[::-1]
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(top) + 1.5))
        ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#54a24b", capsize=2)
        ax.set_xlabel(f"giảm {primary} khi xáo trộn đặc trưng"); ax.set_title(f"Permutation importance - {best_single}")
        save_fig(fig, "permutation_importance.png")

    _write_summary(ds, primary, best_label, thr_df)
    return res


def _write_summary(ds, primary, best_label, thr_df):
    """Gom mọi bảng vào results/SUMMARY.md để copy sang báo cáo."""
    md = ["# SUMMARY (tự sinh - copy số liệu từ đây sang báo cáo)\n"]
    md.append(f"- Metric chính: **{primary}**  |  CV: {config.CV_FOLDS}-fold stratified  |  seed: {config.SEED}")
    md.append(f"- Bài toán: {ds.task}, lớp: {ds.class_names}, n_train={len(ds.y_train)}, n_test={len(ds.y_test)}")
    md.append(f"- Mô hình chọn theo CV: **{best_label}**\n")
    sections = [("Benchmark CV (tham số mặc định)", "benchmark_cv.csv", ["model", f"{primary}_mean", f"{primary}_std", f"train_{primary}", "fit_time_s"]),
                ("Tuning", "tuning_summary.csv", ["model", "default_cv", "tuned_cv", "gain", "best_params"]),
                ("Feature selection", "feature_selection.csv", None),
                ("Ensemble (CV)", "ensemble_cv.csv", None),
                ("Kết quả trên tập test", "test_results.csv", None),
                ("Khoảng tin cậy 95% (bootstrap trên test)", "bootstrap_ci.csv", None),
                ("Phân tích ngưỡng", "threshold_analysis.csv", None),
                ("Top đặc trưng quan trọng (permutation)", "permutation_importance.csv", None)]
    for title, fn, cols in sections:
        d = read_csv_if_exists(fn)
        if d is None:
            continue
        if cols:
            d = d[[c for c in cols if c in d.columns]]
        if fn == "permutation_importance.csv":
            d = d.head(10)
        md.append(f"\n## {title}\n")
        md.append(df_to_md(d))
    md.append("\n## Hình đã sinh\n")
    for p in sorted(config.FIG_DIR.glob("*.png")):
        md.append(f"- results/figures/{p.name}")
    (config.RESULTS_DIR / "SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print("  [md ] results/SUMMARY.md")
