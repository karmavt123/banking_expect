"""Bước 1 - Phân tích dữ liệu khám phá (EDA). Xuất results/eda_summary.md + hình trong results/figures/."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import df_to_md, save_fig


def run(ds):
    print("== EDA ==")
    df = ds.df
    X = df.drop(columns="__y__")
    y = df["__y__"]
    names = ds.class_names
    md = []

    # --- tổng quan ---
    md.append("# Tóm tắt EDA (tự sinh)\n")
    md.append("## Tổng quan\n")
    for k, v in ds.info.items():
        md.append(f"- **{k}**: {v}")
    md.append(f"- **loại bài toán**: {ds.task} ({len(names)} lớp: {names})")
    md.append(f"- **số mẫu train / test**: {len(ds.y_train)} / {len(ds.y_test)}")

    # --- phân bố nhãn ---
    cnt = y.value_counts().sort_index()
    cls = pd.DataFrame({"lớp": [names[i] for i in cnt.index], "số mẫu": cnt.values,
                        "tỉ lệ": (cnt / cnt.sum()).round(4).values})
    md.append("\n## Phân bố nhãn\n")
    md.append(df_to_md(cls))
    md.append(f"\nTỉ lệ lớp lớn nhất / nhỏ nhất = {cnt.max() / cnt.min():.1f}")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar([names[i] for i in cnt.index], cnt.values, color="#4C78A8")
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / cnt.sum():.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_title("Phân bố nhãn"); ax.set_ylabel("số mẫu"); ax.set_ylim(0, cnt.max() * 1.2)
    save_fig(fig, "eda_class_distribution.png")

    # --- missing ---
    miss = X.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    md.append("\n## Giá trị thiếu\n")
    if len(miss):
        mt = pd.DataFrame({"cột": miss.index, "số thiếu": miss.values, "% thiếu": (miss / len(X) * 100).round(2).values})
        md.append(df_to_md(mt.head(25), 2))
        fig, ax = plt.subplots(figsize=(6, max(2.5, 0.3 * min(len(miss), 25))))
        m = miss.head(25)[::-1]
        ax.barh(m.index, m.values / len(X) * 100, color="#E45756")
        ax.set_xlabel("% thiếu"); ax.set_title("Giá trị thiếu theo cột")
        save_fig(fig, "eda_missing_values.png")
    else:
        md.append("Không có giá trị thiếu (sau khi quy đổi NA_VALUES).")

    # --- numeric ---
    if ds.num_cols:
        md.append("\n## Đặc trưng số\n")
        desc = X[ds.num_cols].describe().T[["mean", "std", "min", "50%", "max"]]
        desc["skew"] = X[ds.num_cols].skew()
        q1, q3 = X[ds.num_cols].quantile(0.25), X[ds.num_cols].quantile(0.75)
        iqr = q3 - q1
        out = ((X[ds.num_cols] < q1 - 1.5 * iqr) | (X[ds.num_cols] > q3 + 1.5 * iqr)).mean() * 100
        desc["% outlier(IQR)"] = out
        desc = desc.round(3).reset_index().rename(columns={"index": "cột"})
        md.append(df_to_md(desc, 3))

        cols = ds.num_cols[:12]
        n = len(cols); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow), squeeze=False)
        for ax, c in zip(axes.ravel(), cols):
            ax.hist(X[c].dropna(), bins=30, color="#4C78A8"); ax.set_title(c, fontsize=9)
        for ax in axes.ravel()[n:]:
            ax.axis("off")
        fig.suptitle("Phân bố các đặc trưng số" + (" (12 cột đầu)" if len(ds.num_cols) > 12 else ""))
        save_fig(fig, "eda_numeric_hist.png")

        if len(ds.num_cols) >= 2:
            cc = ds.num_cols[:25]
            corr = X[cc].corr()
            fig, ax = plt.subplots(figsize=(0.45 * len(cc) + 3, 0.4 * len(cc) + 2.5))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cc))); ax.set_xticklabels(cc, rotation=90, fontsize=7)
            ax.set_yticks(range(len(cc))); ax.set_yticklabels(cc, fontsize=7); ax.grid(False)
            fig.colorbar(im, shrink=0.8); ax.set_title("Ma trận tương quan (Pearson)")
            save_fig(fig, "eda_correlation.png")
            c_all = X[ds.num_cols].corr().abs()
            pairs = [(a, b, c_all.loc[a, b]) for i, a in enumerate(ds.num_cols) for b in ds.num_cols[i + 1:]
                     if c_all.loc[a, b] > 0.8]
            if pairs:
                md.append("\nCặp đặc trưng tương quan cao (|r| > 0.8): " +
                          "; ".join(f"{a}~{b} ({r:.2f})" for a, b, r in sorted(pairs, key=lambda t: -t[2])[:15]))

    # --- categorical ---
    if ds.cat_cols:
        md.append("\n## Đặc trưng phân loại\n")
        rows = []
        for c in ds.cat_cols:
            vc = X[c].value_counts(dropna=True)
            rows.append({"cột": c, "số giá trị": X[c].nunique(), "giá trị phổ biến nhất": f"{vc.index[0]} ({vc.iloc[0] / len(X):.0%})" if len(vc) else ""})
        md.append(df_to_md(pd.DataFrame(rows)))
        if ds.task == "binary":
            pick = [c for c in ds.cat_cols if 2 <= X[c].nunique() <= 15][:6]
            if pick:
                base = y.mean()
                n = len(pick); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
                fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3 * nrow), squeeze=False)
                for ax, c in zip(axes.ravel(), pick):
                    r = pd.Series(y.values).groupby(X[c].fillna("missing").values).mean().sort_values()
                    ax.barh([str(i) for i in r.index], r.values, color="#72B7B2")
                    ax.axvline(base, color="k", ls="--", lw=1); ax.set_title(c, fontsize=9)
                for ax in axes.ravel()[n:]:
                    ax.axis("off")
                fig.suptitle(f"Tỉ lệ lớp dương '{names[1]}' theo từng giá trị (đường nét đứt = trung bình)")
                save_fig(fig, "eda_target_rate_by_category.png")

    path = config.RESULTS_DIR / "eda_summary.md"
    path.write_text("\n".join(md), encoding="utf-8")
    print(f"  [md ] {path.relative_to(config.ROOT)}")
