"""Chạy toàn bộ pipeline:  python src/run_all.py
  --steps eda benchmark tune fs ensemble final   (mặc định: tất cả theo thứ tự)
  --quick                                          chạy thử rất nhanh để kiểm tra lỗi (KHÔNG lấy số liệu báo cáo từ đây)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from utils import ensure_dirs  # noqa: E402

ALL = ["eda", "benchmark", "tune", "fs", "ensemble", "final"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", choices=ALL, default=ALL)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        config.QUICK = True
        config.CV_FOLDS = 3
        config.N_ITER_SEARCH = 4
        config.SUBSAMPLE = config.SUBSAMPLE or 4000
        config.FS_PERCENTILES = [50, 100]
        config.ENSEMBLE_TOP_K = 2
        print("*** CHẾ ĐỘ QUICK: chỉ để kiểm tra code chạy được ***")

    ensure_dirs()
    import benchmark, data, ensemble, eda, feature_selection, final, tune  # noqa: E401

    t0 = time.time()
    ds = data.prepare()
    print(f"Dữ liệu: {ds.info['n_rows']} mẫu, {ds.info['n_features']} đặc trưng "
          f"({ds.info['n_numeric']} số, {ds.info['n_categorical']} phân loại), bài toán {ds.task}, "
          f"lớp {ds.info['class_counts']}")
    steps = {"eda": eda.run, "benchmark": benchmark.run, "tune": tune.run,
             "fs": feature_selection.run, "ensemble": ensemble.run, "final": final.run}
    for s in ALL:
        if s in args.steps:
            ts = time.time()
            steps[s](ds)
            print(f"-- {s} xong sau {time.time() - ts:.0f}s\n")
    print(f"HOÀN TẤT sau {time.time() - t0:.0f}s. Xem thư mục results/ (bắt đầu từ SUMMARY.md).")


if __name__ == "__main__":
    main()
