# Đối chiếu yêu cầu đề (ML-huong-dan-BTL.pdf) — còn phần nào nhóm phải tự làm

Pipeline lo phần chạy thực nghiệm + sinh bảng/hình. Các mục **[TỰ LÀM]** cần người viết/quyết định.

| # | Yêu cầu của đề | Đã có sẵn trong code | Việc còn lại |
|---|---|---|---|
| 1 | Chọn dataset, mô tả vì sao phù hợp, thách thức chính | — | **[TỰ LÀM]** chọn dataset, **liên hệ giảng viên duyệt nếu không chắc**, viết mục 1 báo cáo |
| 2 | Phân tích bài toán & đặc điểm dữ liệu (dạng, số mẫu, phân bố, thách thức) | `eda.py` → `results/eda_summary.md` + hình | **[TỰ LÀM]** nhận xét, diễn giải các hình |
| 3 | ≥ 2 thuật toán khác nhau | 6 thuật toán (LogReg, Tree, RF, GB, SVM, MLP) | chọn giữ/bớt trong `config.MODELS`; viết phần mô tả phương pháp |
| 4 | Thực nghiệm, so sánh, **giải thích vì sao chọn metric** | `benchmark.py`, `final.py`, `metrics.py` | **[TỰ LÀM]** viết lý do chọn metric theo bối cảnh dataset (gợi ý ở `GIAI_THICH.md` §4) |
| 5 | Kỹ thuật nâng cao: feature selection, tuning, ensemble, baseline DL | `feature_selection.py`, `tune.py`, `ensemble.py`, MLP | **[TỰ LÀM]** diễn giải kết quả có cải thiện thật không |
| 6 | Báo cáo (mục tiêu, phương pháp, kết quả, thảo luận, kết luận) | template ở `report/BAO_CAO_template.md` | **[TỰ LÀM]** điền, viết thảo luận bằng lời của mình |
| 7 | Slide thuyết trình | outline ở `slides/OUTLINE.md` | làm slide (có thể nhờ Claude dựng sau khi có kết quả) |
| – | README.md cách chạy | `README.md` (còn `<<...>>` cần điền) | điền nhóm, dataset, thời gian chạy, phiên bản |
| – | requirements / Dockerfile | có sẵn | — |
| – | Nén `MSSV1-MSSV2-....zip` | `python package.py MSSV1 MSSV2` | file lớn upload Drive, dán link vào README |
| – | Hiểu bài để bảo vệ | `notes/GIAI_THICH.md` | **[TỰ LÀM]** đọc, tự trả lời 11 câu hỏi cuối |

## Thứ tự làm gợi ý
1. Chốt dataset → bỏ vào `data/` → sửa `src/config.py` → `python src/run_all.py --quick` (kiểm tra chạy được)
2. `python src/run_all.py` (chạy đầy đủ) → đọc `results/SUMMARY.md`
3. Điền báo cáo từ số liệu trong `results/`, chèn hình trong `results/figures/`
4. Làm slide, điền README, `python package.py ...`, thử giải nén & chạy lại ở máy sạch (hoặc Docker) để chắc tái hiện được
