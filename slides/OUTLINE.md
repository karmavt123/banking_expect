# Outline slide thuyết trình (≈ 12 slide, 10–12 phút)

Mỗi slide: **một thông điệp**, ít chữ, hình lấy từ `results/figures/`. Slide thật sẽ dựng sau khi có kết quả.

| # | Slide | Nội dung chính | Hình / bảng |
|---|---|---|---|
| 1 | Tiêu đề | Tên đề tài, nhóm, GVHD | — |
| 2 | Bài toán & động lực | Dự đoán gì, vì sao quan trọng | — |
| 3 | Dataset & thách thức | n mẫu, d đặc trưng, imbalance/missing... | `eda_class_distribution.png` |
| 4 | Khám phá dữ liệu | 1–2 insight đáng chú ý nhất | `eda_target_rate_by_category.png` hoặc `eda_correlation.png` |
| 5 | Pipeline thực nghiệm | sơ đồ: split → pipeline → CV → tuning → test (nhấn mạnh không leakage) | sơ đồ hộp |
| 6 | Các mô hình | 6 mô hình, mỗi cái 1 dòng ý tưởng | bảng |
| 7 | Vì sao chọn metric này | accuracy sai ở đâu, metric chính là gì | ví dụ số cụ thể |
| 8 | Kết quả CV & tuning | default vs tuned | `benchmark_cv_boxplot.png` + bảng gain |
| 9 | Feature selection & ensemble | có cải thiện không | `feature_selection.png`, bảng ensemble |
| 10 | Kết quả cuối trên test | bảng + mô hình chọn (theo CV) | `test_comparison.png`, `roc_pr_curves.png` |
| 11 | Phân tích lỗi & thảo luận | confusion matrix, ngưỡng, đặc trưng quan trọng, learning curve | `confusion_matrices.png`, `permutation_importance.png` |
| 12 | Kết luận & hướng phát triển | 3 ý chính + hạn chế thật | — |
| (phụ lục) | Q&A dự phòng | learning curve, bootstrap CI, tham số tốt nhất | |

## Gợi ý phân công thuyết trình
Chia theo khối: (2–4) dữ liệu · (5–7) phương pháp · (8–10) kết quả · (11–12) thảo luận/kết luận. **Ai cũng phải trả lời được câu hỏi ở mọi khối** (xem `notes/GIAI_THICH.md` §11).
