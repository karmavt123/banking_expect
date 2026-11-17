"""Tiền xử lý đặt TRONG pipeline sklearn để tránh data leakage:
mọi thống kê (median, mean/std, danh sách category...) chỉ được học trên phần train của từng fold."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    parts = []
    if num_cols:
        parts.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),   # median: ít bị outlier kéo lệch
            ("scaler", StandardScaler()),                    # cần cho LogReg/SVM/MLP; vô hại với cây
        ]), list(num_cols)))
    if cat_cols:
        parts.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),  # missing thành 1 category riêng
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                     sparse_output=False)),  # gộp category hiếm (<1%) để tránh nổ chiều
        ]), list(cat_cols)))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)
