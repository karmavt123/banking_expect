# BTL Học máy (CO3117) — <<Tên đề tài / tên dataset>>

**Nhóm:** <<MSSV - Họ tên>>, <<MSSV - Họ tên>>
**Giảng viên:** <<...>> | **Học kỳ:** <<...>>

Bài toán: <<1–2 câu: dự đoán gì từ gì, vd "dự đoán khách hàng có đăng ký gửi tiền kỳ hạn hay không">>.
Dataset: <<tên + link nguồn>> (<<n>> mẫu, <<d>> đặc trưng, <<k>> lớp).

## 1. Cấu trúc thư mục

```
bai-lam/
├── README.md              <- file này
├── requirements.txt       <- thư viện cần cài
├── Dockerfile             <- (tuỳ chọn) môi trường Docker
├── package.py             <- đóng gói zip để nộp
├── data/                  <- dataset (file lớn: xem mục 5)
├── src/                   <- toàn bộ mã nguồn
│   ├── config.py          <- CẤU HÌNH DUY NHẤT cần chỉnh (dataset, metric, số fold...)
│   ├── data.py            <- đọc dữ liệu, làm sạch, chia train/test
│   ├── preprocess.py      <- tiền xử lý (impute, scale, one-hot) nằm trong pipeline
│   ├── models.py          <- các mô hình + không gian tham số
│   ├── metrics.py         <- chọn & tính metric
│   ├── eda.py             <- bước 1: EDA
│   ├── benchmark.py       <- bước 2: so sánh mô hình (tham số mặc định, CV)
│   ├── tune.py            <- bước 3: tối ưu siêu tham số
│   ├── feature_selection.py <- bước 4: chọn đặc trưng
│   ├── ensemble.py        <- bước 5: voting / stacking
│   ├── final.py           <- bước 6: đánh giá trên tập test, vẽ hình
│   └── run_all.py         <- chạy toàn bộ pipeline
├── results/               <- bảng csv, SUMMARY.md, figures/ (tự sinh)
├── models/                <- mô hình đã huấn luyện .joblib (tự sinh)
├── report/                <- báo cáo
├── slides/                <- slide thuyết trình
└── notes/                 <- ghi chú giải thích các lựa chọn thiết kế
```

## 2. Cài đặt môi trường

Yêu cầu Python ≥ 3.9.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Hoặc dùng Docker:

```bash
docker build -t btl-ml .
docker run --rm -v "$PWD/results:/app/results" btl-ml
```

## 3. Cách chạy

1. Đặt dataset vào `data/`, mở `src/config.py`, chỉnh **mục 1 (DATASET)**: `DATA_FILE`, `TARGET`, ...
2. Chạy thử nhanh để kiểm tra cấu hình (vài chục giây, số liệu *không* dùng cho báo cáo):

   ```bash
   python src/run_all.py --quick
   ```
3. Chạy đầy đủ:

   ```bash
   python src/run_all.py
   ```
   Hoặc chạy từng bước: `python src/run_all.py --steps eda benchmark`
   (thứ tự: `eda → benchmark → tune → fs → ensemble → final`; `fs`, `ensemble`, `final` cần `tune` đã chạy).
4. Xem kết quả: `results/SUMMARY.md` (tổng hợp mọi bảng), `results/figures/*.png`, `results/eda_summary.md`.

Thời gian chạy đầy đủ tham khảo: <<điền sau khi chạy, vd ~10 phút trên MacBook M-series>>.

## 4. Quy trình thực nghiệm (tóm tắt)

| Bước | Nội dung |
|---|---|
| Chia dữ liệu | Hold-out test <<20>>% (stratified, seed = <<42>>), chỉ dùng **một lần** ở bước cuối |
| Tiền xử lý | Impute (median / "missing"), chuẩn hoá, one-hot — đặt trong `Pipeline` để không rò rỉ dữ liệu |
| Mô hình | <<Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM (RBF), MLP>> |
| Đánh giá | Stratified <<5>>-fold CV; metric chính **<<tên metric>>** (lý do: xem báo cáo, mục 4) |
| Tuning | RandomizedSearchCV, <<25>> tổ hợp / mô hình |
| Nâng cao | Feature selection, ensemble (soft voting + stacking), phân tích ngưỡng, bootstrap CI |

Mô hình cuối cùng được **chọn theo điểm CV trên tập train**, không chọn theo tập test.

## 5. File lớn / dữ liệu

- Dataset gốc: <<link>>
- Mô hình đã huấn luyện (`models/*.joblib`): <<link Drive/GitHub nếu có>>

## 6. Ghi chú tái hiện

- Mọi nguồn ngẫu nhiên dùng `SEED = 42` (xem `src/config.py`); kết quả có thể lệch rất nhỏ giữa các phiên bản scikit-learn / hệ điều hành.
- Phiên bản đã chạy: Python <<3.x>>, scikit-learn <<x.y>> (chạy `pip freeze | grep -i -E "scikit|pandas|numpy"` để lấy).
- Có dùng AI hỗ trợ tham khảo / debug; các thành viên đã đọc, hiểu và chịu trách nhiệm toàn bộ nội dung nộp.
