# <<Tên đề tài>>
**Bài tập lớn Học máy (CO3117)** — Nhóm <<..>>: <<MSSV – Họ tên>>, <<MSSV – Họ tên>>
GVHD: <<...>> — <<Tháng/Năm>>

> Cách dùng template: thay mọi `<<...>>`; các dòng "*Gợi ý*" xoá đi khi viết xong. Số liệu lấy từ `results/SUMMARY.md`, hình từ `results/figures/`.
> Phần **Thảo luận** và **lý do chọn metric** phải viết bằng suy nghĩ của nhóm theo dataset cụ thể.

---

## 1. Giới thiệu bài toán
1.1. **Bối cảnh & mục tiêu**: <<bài toán thực tế là gì, ai quan tâm, dự đoán cái gì>>.
1.2. **Dataset**: <<tên, nguồn (link), số mẫu, số đặc trưng, biến mục tiêu, giấy phép nếu có>>.
1.3. **Vì sao dataset này phù hợp để áp dụng ML**: <<2–4 ý>>.
1.4. **Thách thức chính của dữ liệu**: <<vd mất cân bằng lớp (x:y), giá trị thiếu, đặc trưng phân loại nhiều giá trị, nhiễu, tương quan cao...>>.
1.5. **Đóng góp / phạm vi của báo cáo**: <<so sánh k thuật toán, tuning, feature selection, ensemble...>>.

*Gợi ý*: nói rõ bài toán là nhị phân hay đa lớp, và "dương" nghĩa là gì.

## 2. Dữ liệu & tiền xử lý
2.1. **Mô tả dữ liệu** — bảng: số mẫu, số cột số/phân loại, tỉ lệ thiếu, tỉ lệ lớp.
  → lấy từ `results/eda_summary.md`. Chèn: `eda_class_distribution.png`, `eda_numeric_hist.png`, `eda_correlation.png`, `eda_missing_values.png` (nếu có).
2.2. **Nhận xét EDA**: <<phân bố lệch? outlier? cặp đặc trưng tương quan? đặc trưng nào có vẻ phân biệt lớp (`eda_target_rate_by_category.png`)?>>
2.3. **Làm sạch**: bỏ <<n>> dòng trùng, bỏ cột <<...>> vì <<...>> (vd leakage / id).
2.4. **Tiền xử lý** (trong pipeline để tránh data leakage): impute median / "missing", chuẩn hoá, one-hot (gộp category < 1%).
2.5. **Chia dữ liệu**: train <<n_train>> / test <<n_test>> (stratified, seed 42); mọi lựa chọn mô hình dùng CV <<5>>-fold trên train, test chỉ dùng một lần ở cuối.

## 3. Mô hình & phương pháp
3.1. **Các thuật toán** (mỗi cái 2–4 câu: ý tưởng, siêu tham số chính, vì sao chọn): Logistic Regression (baseline tuyến tính), Decision Tree, Random Forest (bagging), Gradient Boosting (boosting), SVM-RBF, MLP (baseline DL). → xem `notes/GIAI_THICH.md` §6, **viết lại bằng lời của nhóm**.
3.2. **Xử lý mất cân bằng**: `class_weight="balanced"` <<(MLP không hỗ trợ)>>; chọn ngưỡng trên dự đoán out-of-fold.
3.3. **Metric & lý do chọn** *(bắt buộc giải thích)*:
  - Metric chính: **<<tên>>**, vì <<lý do gắn với bối cảnh dataset>>.
  - Metric phụ: <<accuracy (chỉ tham khảo, vì...), precision, recall, F1, ROC-AUC, MCC>>.
3.4. **Quy trình đánh giá**: Stratified <<5>>-fold CV → RandomizedSearchCV (<<25>> tổ hợp) → chọn mô hình theo điểm CV → đánh giá một lần trên test; bootstrap 95% CI.
3.5. **Kỹ thuật nâng cao**: feature selection (ANOVA-F, L1, RF importance), ensemble (soft voting, stacking của top-<<3>>).

## 4. Thực nghiệm & kết quả
4.1. **Thiết lập**: phần cứng <<...>>, Python <<..>>, scikit-learn <<..>>, thời gian chạy <<..>>.
4.2. **So sánh tham số mặc định (CV)** — bảng từ `SUMMARY.md` §Benchmark + hình `benchmark_cv_boxplot.png`. Chú ý cột *train vs CV* (overfitting).
4.3. **Sau tuning** — bảng §Tuning (default → tuned, gain) + tham số tốt nhất.
4.4. **Feature selection** — bảng + `feature_selection.png`.
4.5. **Ensemble** — bảng §Ensemble.
4.6. **Kết quả trên tập test** — bảng §Kết quả test + `test_comparison.png`, `confusion_matrices.png`, `roc_pr_curves.png`.
4.7. **Độ tin cậy** — bảng bootstrap CI 95%; `learning_curve.png`.
4.8. **Phân tích ngưỡng** (nhị phân) — bảng + `threshold_f1.png`.
4.9. **Đặc trưng quan trọng** — `permutation_importance.png`.

## 5. Thảo luận *(phần này ăn điểm — viết có lập luận, bám số liệu)*
- Mô hình nào tốt nhất? Chênh lệch với các mô hình còn lại **có ý nghĩa không** (CI chồng nhau không)?
- Vì sao nhóm mô hình đó tốt/kém trên dữ liệu này (bản chất dữ liệu bảng, phi tuyến, kích thước, nhiễu...)?
- Overfitting/underfitting ở mô hình nào? Bằng chứng (train vs CV, learning curve)?
- Tuning, feature selection, ensemble có thực sự cải thiện không? Nếu không, vì sao (cũng là kết luận hợp lệ)?
- Lỗi chủ yếu ở đâu (confusion matrix: lớp nào bị nhầm)? Đánh đổi precision–recall và ngưỡng phù hợp cho ứng dụng thực tế?
- Đặc trưng nào quan trọng nhất; có hợp lý theo kiến thức miền không?
- **Hạn chế**: không nested CV, một seed, chưa thử SMOTE/feature engineering miền, ... (xem `notes/GIAI_THICH.md` §12).

## 6. Kết luận & hướng phát triển
- Tóm tắt: <<3–4 gạch đầu dòng kết quả chính, kèm số>>.
- Hướng phát triển: <<feature engineering theo miền, nested CV, thử XGBoost/LightGBM, hiệu chỉnh xác suất (calibration), SHAP, cost-sensitive learning...>>.

## Tài liệu tham khảo
- <<Nguồn dataset>>
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011.
- Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization*, JMLR 13, 2012.
- <<Slide môn học CO3117; các bài báo/sách khác nhóm đã dùng>>

## Phụ lục
- A. Phân công công việc: <<MSSV – việc>>
- B. Link dữ liệu / mô hình lớn: <<...>>
- C. Việc sử dụng công cụ AI: dùng để tham khảo, giải thích, debug; các thành viên đã đọc hiểu và chịu trách nhiệm nội dung.
