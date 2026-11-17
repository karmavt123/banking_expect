"""Đọc dữ liệu, làm sạch nhẹ, chia train/test, nhận diện kiểu cột."""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


@dataclass
class Dataset:
    df: pd.DataFrame            # toàn bộ dữ liệu sau làm sạch (có cột target đã mã hoá, tên "__y__")
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    num_cols: List[str]
    cat_cols: List[str]
    class_names: List[str]      # class_names[i] là tên của nhãn số i
    task: str                   # "binary" | "multiclass"
    info: dict = field(default_factory=dict)  # thống kê để ghi vào báo cáo


def find_data_file():
    if config.DATA_FILE:
        p = config.DATA_DIR / config.DATA_FILE
        if not p.exists():
            raise FileNotFoundError(f"Không thấy {p}. Kiểm tra lại config.DATA_FILE")
        return p
    exts = {".csv", ".tsv", ".txt", ".data", ".xlsx", ".xls"}
    cands = sorted(p for p in config.DATA_DIR.glob("*") if p.suffix.lower() in exts)
    if not cands:
        raise FileNotFoundError(
            f"Thư mục {config.DATA_DIR} chưa có file dữ liệu. Bỏ dataset vào đó rồi sửa config.py (mục 1)."
        )
    return cands[0]


def load_raw() -> pd.DataFrame:
    path = find_data_file()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, na_values=config.NA_VALUES)
    else:
        kw = dict(na_values=config.NA_VALUES)
        kw.update(config.READ_KWARGS)
        if "sep" not in kw and "delimiter" not in kw:
            kw.update(sep=None, engine="python")  # tự đoán dấu phân cách
        df = pd.read_csv(path, **kw)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def prepare() -> Dataset:
    if not config.TARGET:
        raise ValueError("Chưa đặt config.TARGET (tên cột nhãn). Mở src/config.py, sửa mục 1.")
    df = load_raw()
    info = {"file": str(find_data_file().name), "n_rows_raw": len(df), "n_cols_raw": df.shape[1]}
    if config.TARGET not in df.columns:
        raise KeyError(f"Cột target '{config.TARGET}' không có. Các cột hiện có: {list(df.columns)}")

    # --- làm sạch nhẹ ---
    # Bỏ dòng trùng TRƯỚC khi bỏ cột (nếu bỏ cột trước, các dòng chỉ khác nhau ở cột bị bỏ sẽ bị coi nhầm là trùng)
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    info["n_duplicates_removed"] = n0 - len(df)
    df = df.drop(columns=[c for c in config.DROP_COLUMNS if c in df.columns])
    n0 = len(df)
    df = df[df[config.TARGET].notna()].reset_index(drop=True)
    info["n_missing_target_removed"] = n0 - len(df)

    for c in config.NUMERIC_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # bỏ cột hằng số (không mang thông tin)
    const_cols = [c for c in df.columns if c != config.TARGET and df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=const_cols)
    info["constant_cols_dropped"] = const_cols

    if config.SUBSAMPLE and len(df) > config.SUBSAMPLE:
        df, _ = train_test_split(df, train_size=config.SUBSAMPLE, stratify=df[config.TARGET],
                                 random_state=config.SEED)
        df = df.reset_index(drop=True)
        info["subsampled_to"] = len(df)

    # --- mã hoá nhãn ---
    raw_y = df[config.TARGET]
    counts = raw_y.value_counts()
    classes = list(counts.index)
    if len(classes) < 2:
        raise ValueError("Target chỉ có 1 lớp.")
    if len(classes) == 2:
        task = "binary"
        pos = config.POSITIVE_LABEL
        if pos is None:
            pos = counts.idxmin()               # lớp thiểu số
        if pos not in classes:
            # thử so sánh dạng chuỗi (vd POSITIVE_LABEL="1" mà cột là int)
            m = {str(c): c for c in classes}
            if str(pos) in m:
                pos = m[str(pos)]
            else:
                raise ValueError(f"POSITIVE_LABEL={pos!r} không nằm trong {classes}")
        neg = [c for c in classes if c != pos][0]
        order = [neg, pos]                       # nhãn 0 = neg, 1 = pos
    else:
        task = "multiclass"
        order = sorted(classes, key=lambda v: str(v))
    mapping = {c: i for i, c in enumerate(order)}
    y = raw_y.map(mapping).astype(int).to_numpy()
    class_names = [str(c) for c in order]

    X = df.drop(columns=[config.TARGET])

    # --- nhận diện kiểu cột ---
    forced_cat = [c for c in config.CATEGORICAL_COLUMNS if c in X.columns]
    for c in forced_cat:
        X[c] = X[c].astype("object")
    num_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c not in forced_cat]
    cat_cols = [c for c in X.columns if c not in num_cols]
    for c in cat_cols:
        X[c] = X[c].astype("object")             # (bool/category -> object để imputer xử lý thống nhất)

    # --- chia train/test (phân tầng theo nhãn) ---
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
    X_train, X_test = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    y_train, y_test = y[tr], y[te]

    dist = pd.Series(y).value_counts().sort_index()
    info.update({
        "n_rows": len(X), "n_features": X.shape[1], "n_numeric": len(num_cols), "n_categorical": len(cat_cols),
        "n_train": len(tr), "n_test": len(te),
        "class_counts": {class_names[i]: int(dist[i]) for i in dist.index},
        "minority_ratio": float(dist.min() / dist.sum()),
        "positive_label": class_names[1] if task == "binary" else None,
        "n_missing_cells": int(X.isna().sum().sum()),
    })

    full = X.copy()
    full["__y__"] = y
    return Dataset(full, X_train, X_test, y_train, y_test, num_cols, cat_cols, class_names, task, info)
