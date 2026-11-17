"""Hàm tiện ích dùng chung."""
import matplotlib

matplotlib.use("Agg")  # vẽ ra file, không cần màn hình
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def ensure_dirs():
    for d in (config.DATA_DIR, config.RESULTS_DIR, config.FIG_DIR, config.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def cv_splitter():
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)


def save_fig(fig, name: str):
    path = config.FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.relative_to(config.ROOT)}")


def save_csv(df: pd.DataFrame, name: str):
    path = config.RESULTS_DIR / name
    df.to_csv(path, index=False)
    print(f"  [csv] {path.relative_to(config.ROOT)}")


def read_csv_if_exists(name: str):
    p = config.RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def df_to_md(df: pd.DataFrame, digits: int = 4) -> str:
    """DataFrame -> bảng markdown (không cần thư viện tabulate)."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.{digits}f}"
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines)
