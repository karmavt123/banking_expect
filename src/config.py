"""
CẤU HÌNH TRUNG TÂM CỦA BTL.

Khi đã có dataset, CHỈ CẦN SỬA MỤC 1 (DATASET) ở dưới. Phần còn lại để mặc định
cũng chạy được; muốn chỉnh thì đọc comment từng dòng.

Lưu ý: các module khác luôn đọc `config.TÊN` lúc chạy (không copy giá trị lúc import),
nên `run_all.py --quick` có thể ghi đè cấu hình để chạy thử nhanh.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Đường dẫn (không cần sửa)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT / "models"

# ----------------------------------------------------------------------------
# 1. DATASET  <-- SỬA Ở ĐÂY KHI CÓ DATASET
# ----------------------------------------------------------------------------
DATA_FILE = "bank+marketing/bank-additional/bank-additional-full.csv"  # tên file trong thư mục data/, vd "bank-full.csv". None = tự lấy file đầu tiên tìm thấy
READ_KWARGS = {"sep": ";"}  # tham số cho pandas.read_csv, vd {"sep": ";"} (UCI Bank Marketing dùng dấu ;)
                        # để trống -> pandas tự đoán dấu phân cách
TARGET = "y"            # BẮT BUỘC: tên cột nhãn, vd "y"
POSITIVE_LABEL = "yes"  # bài toán 2 lớp: nhãn được coi là "dương" (lớp quan tâm), vd "yes".
                        # None = lớp ít mẫu hơn (thường đúng cho bài toán imbalance)
DROP_COLUMNS = ["duration"]  # cột bỏ đi: id, tên, cột gây leakage (vd "duration" trong Bank Marketing)
NA_VALUES = ["?", "NA", "N/A", "", "null", "NULL"]  # chuỗi được coi là missing
CATEGORICAL_COLUMNS = []  # ép các cột số thành categorical (vd mã vùng, mã ngành)
NUMERIC_COLUMNS = []      # ép các cột (đang là chuỗi) thành số
SUBSAMPLE = None        # nếu data rất lớn (>100k dòng) có thể lấy mẫu phân tầng, vd 50000. None = dùng hết

# ----------------------------------------------------------------------------
# 2. THIẾT LẬP THỰC NGHIỆM
# ----------------------------------------------------------------------------
SEED = 42               # cố định để tái hiện kết quả
TEST_SIZE = 0.2         # tỉ lệ tập test giữ riêng (chỉ dùng đúng 1 lần ở bước final)
CV_FOLDS = 5            # số fold của Stratified K-Fold
N_JOBS = -1             # -1 = dùng hết CPU
N_ITER_SEARCH = 15      # số tổ hợp tham số thử cho mỗi mô hình khi tuning (RandomizedSearchCV)
USE_CLASS_WEIGHT = True # True: dùng class_weight="balanced" cho mô hình hỗ trợ (xử lý imbalance)

# Mô hình sẽ chạy. Tên hợp lệ: logreg, decision_tree, random_forest, hist_gb, svm_rbf, mlp
MODELS = ["logreg", "decision_tree", "random_forest", "hist_gb", "svm_rbf", "mlp"]
SVM_MAX_ROWS = 20000    # SVM RBF chậm ~O(n^2): nếu số dòng train vượt ngưỡng này thì tự bỏ qua SVM

# Metric chính dùng để tuning + chọn mô hình ("auto" = tự chọn theo bài toán, xem metrics.py)
# Có thể ghi đè bằng: "f1", "average_precision", "roc_auc", "f1_macro", "balanced_accuracy", "accuracy"
PRIMARY_METRIC = "auto"

# Thí nghiệm feature selection: chạy trên mô hình nào ("auto" = mô hình tốt nhất sau tuning)
FS_MODEL = "auto"
FS_PERCENTILES = [10, 25, 50, 75, 100]   # % số đặc trưng giữ lại (lọc theo ANOVA F-score)

# Số mô hình tốt nhất (có predict_proba) đưa vào ensemble
ENSEMBLE_TOP_K = 3

QUICK = False           # bật bởi `run_all.py --quick` (chạy thử cho nhanh, KHÔNG dùng để lấy số liệu báo cáo)
