# Giải thích các lựa chọn trong pipeline (đọc kỹ trước khi nộp / bảo vệ)

Đề cho phép dùng AI để tham khảo, nhưng **không được nộp thứ mình không hiểu** — thầy có thể gọi nhóm lên hỏi.
File này giải thích "vì sao làm vậy" cho từng bước, kèm các câu hỏi thầy hay hỏi. Đọc xong mà trả lời được hết phần cuối thì ổn.

---

## 1. Chia dữ liệu & vì sao tập test chỉ chạm một lần

- Chia **train 80% / test 20%**, `stratify=y` để tỉ lệ lớp ở hai tập giống nhau (rất quan trọng khi imbalance).
- Mọi việc chọn mô hình / tuning / chọn đặc trưng chỉ dùng **CV trên tập train**. Tập test chỉ để đo cuối cùng.
- Nếu dùng test để chọn mô hình thì con số test không còn khách quan (đã "nhìn trộm" đáp án) → gọi là *data snooping*.
- Mô hình "tốt nhất" trong `final.py` được chọn theo **điểm CV**, không phải theo điểm test.

## 2. Data leakage & vì sao dùng `Pipeline`

Leakage = thông tin của tập validation/test lọt vào quá trình huấn luyện.
Ví dụ sai: chuẩn hoá (`StandardScaler`) trên *toàn bộ* dữ liệu rồi mới CV → mean/std đã chứa thông tin của fold validation.
Cách làm đúng: đặt impute + scale + one-hot + (feature selection) + model vào **một `Pipeline`**; mỗi fold chỉ `fit` các bước tiền xử lý trên phần train của fold đó.

## 3. Tiền xử lý

| Việc | Lý do |
|---|---|
| Impute số bằng **median** | median ít bị outlier kéo lệch hơn mean |
| Impute categorical bằng `"missing"` | coi "thiếu" là một giá trị riêng — đôi khi bản thân việc thiếu đã mang thông tin |
| `StandardScaler` | Logistic Regression, SVM, MLP nhạy với thang đo; cây thì không cần nhưng không hại → dùng chung một tiền xử lý cho đơn giản |
| One-hot, gộp category hiếm (< 1%) | tránh nổ số chiều và overfit vào category xuất hiện vài lần |
| Bỏ duplicate, cột hằng số | không mang thông tin; duplicate còn làm CV lạc quan (cùng một mẫu nằm ở cả train lẫn validation) |

## 4. Metric — phần thầy rất hay hỏi ("vì sao chọn metric này?")

- **Accuracy** chỉ hợp khi các lớp cân bằng. Nếu 90% là lớp "No", đoán toàn "No" đã đạt 90% accuracy mà vô dụng.
- **Precision** = trong số dự đoán là dương, bao nhiêu đúng. **Recall** = trong số dương thật, bắt được bao nhiêu.
  **F1** = trung bình điều hoà của hai cái, phạt mạnh khi một trong hai thấp.
- **ROC-AUC** = xác suất mô hình xếp một mẫu dương cao hơn một mẫu âm ngẫu nhiên. Không phụ thuộc ngưỡng. Nhưng khi lớp dương rất hiếm, ROC-AUC có thể trông đẹp dù mô hình tệ ở lớp dương (vì số âm thật quá lớn nên FPR nhỏ).
- **Average Precision (AP)** = diện tích dưới đường Precision–Recall, chỉ nhìn vào lớp dương → nhạy hơn khi imbalance. Đây là metric chính mặc định cho nhị phân imbalance (lớp dương < 30%).
- **Macro-F1** (đa lớp): tính F1 từng lớp rồi lấy trung bình không trọng số → lớp nhỏ nặng bằng lớp lớn.
- **MCC** (Matthews): một con số cân bằng dùng cả 4 ô của confusion matrix, ổn khi imbalance.
- Chọn metric chính nên **gắn với bối cảnh**: bỏ sót ca dương đắt hơn báo nhầm → ưu tiên recall; báo nhầm tốn kém → ưu tiên precision. *Phần này nhóm phải tự viết theo dataset cụ thể.*

## 5. Xử lý imbalance

- Dùng `class_weight="balanced"`: tăng trọng số lỗi của lớp thiểu số khi huấn luyện → không đổi dữ liệu, không rò rỉ. (MLP của sklearn không hỗ trợ `class_weight` — ghi rõ trong báo cáo.)
- Đổi **ngưỡng quyết định** (mặc định 0.5): `final.py` chọn ngưỡng tối đa F1 bằng dự đoán out-of-fold trên train, rồi áp dụng lên test. Chọn ngưỡng bằng test là gian lận (lại là snooping).
- Không dùng SMOTE mặc định vì phải đặt trong pipeline của `imblearn` và dễ làm sai (oversample trước khi CV → leakage). Nếu muốn thêm thì phải nói rõ trong báo cáo.

## 6. Các mô hình (ý tưởng – siêu tham số chính – ưu/nhược)

| Mô hình | Ý tưởng | Siêu tham số quan trọng | Ghi chú |
|---|---|---|---|
| Logistic Regression | tổ hợp tuyến tính + sigmoid, tối thiểu log-loss, có L2 | `C` (nghịch đảo độ mạnh regularization) | nhanh, dễ giải thích; yếu nếu quan hệ phi tuyến → làm baseline |
| Decision Tree | chia dữ liệu liên tiếp theo ngưỡng một đặc trưng (gini/entropy) | `max_depth`, `min_samples_leaf`, `ccp_alpha` | dễ overfit nếu không giới hạn độ sâu (train ≈ 1.0 nhưng CV thấp là dấu hiệu) |
| Random Forest | nhiều cây trên bootstrap + chọn ngẫu nhiên đặc trưng, lấy trung bình → giảm variance (bagging) | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | khá bền, ít cần tuning |
| Gradient Boosting (HistGB) | cây nông thêm lần lượt, mỗi cây sửa sai số của cây trước (giảm bias) | `learning_rate`, `max_leaf_nodes`, `max_iter`, `l2_regularization` | thường mạnh nhất trên tabular; nhạy với tham số hơn RF |
| SVM (RBF) | tìm siêu phẳng lề lớn nhất trong không gian đặc trưng do kernel RBF tạo ra | `C`, `gamma` | chậm với n lớn (≈ O(n²)) nên có ngưỡng `SVM_MAX_ROWS` |
| MLP | mạng nơ-ron nhiều lớp, huấn luyện bằng backprop (Adam) | `hidden_layer_sizes`, `alpha` (L2), `learning_rate_init` | baseline "deep learning" theo gợi ý của đề; thường không hơn GB trên dữ liệu bảng |

## 7. Tuning

- `RandomizedSearchCV`: thử ngẫu nhiên N tổ hợp từ phân phối tham số. Với không gian nhiều chiều, random search thường hiệu quả hơn grid search (Bergstra & Bengio, 2012) vì không lãng phí công vào tham số ít quan trọng.
- Điểm CV tốt nhất sau tuning hơi **lạc quan** (ta chọn tham số *dựa trên* chính điểm đó). Con số khách quan = điểm trên test. Muốn chặt hơn thì dùng nested CV (ghi ở "Hạn chế").
- Nếu `tuned_cv` ≈ `default_cv` thì không sao: nghĩa là tham số mặc định đã gần tối ưu — đó cũng là một kết luận.

## 8. Feature selection (3 họ phương pháp)

- **Filter** — ANOVA F-score: xếp hạng từng đặc trưng độc lập với mô hình, giữ top p%. Nhanh nhưng bỏ qua tương tác giữa các đặc trưng.
- **Embedded (L1)** — LinearSVC với phạt L1 ép hệ số các đặc trưng vô ích về đúng 0.
- **Embedded (cây)** — giữ đặc trưng có độ quan trọng ≥ median theo Random Forest.
- Chọn đặc trưng đặt **trong pipeline** nên mỗi fold tự chọn lại (tránh leakage). Với mô hình dạng cây/boosting, giảm đặc trưng thường không cải thiện nhiều; lợi ích chính là đơn giản & nhanh hơn — nói đúng như vậy nếu kết quả cho thấy thế.
- *Permutation importance* (cuối `final.py`): xáo trộn một cột trong test, đo metric giảm bao nhiêu → đặc trưng quan trọng thật sự với mô hình (tính trên cột gốc nên không bị chia nhỏ bởi one-hot).

## 9. Ensemble

- **Soft voting**: trung bình xác suất dự đoán của các mô hình thành viên.
- **Stacking**: dùng dự đoán (out-of-fold) của các mô hình thành viên làm đặc trưng cho một mô hình meta (Logistic Regression).
- Ensemble hiệu quả khi các mô hình thành viên **khác nhau** và mỗi cái có lỗi khác nhau. Nếu thành viên đều là cây thì lợi ích thường nhỏ. SVM bị loại vì không có `predict_proba` mặc định.

## 10. Cách đọc kết quả (để viết phần Thảo luận)

- **Train vs CV**: train ≈ 1.0 mà CV thấp hơn nhiều → overfitting (cây, RF, boosting hay vậy). Cả hai đều thấp → underfitting (mô hình quá đơn giản / thiếu đặc trưng).
- **Learning curve**: khoảng cách train–validation còn lớn và validation vẫn đang đi lên khi thêm dữ liệu → thêm dữ liệu sẽ giúp; hai đường đã hội tụ ở mức thấp → cần mô hình/đặc trưng tốt hơn.
- **Bootstrap CI 95%** (bảng `bootstrap_ci.csv`): nếu khoảng tin cậy của hai mô hình chồng lên nhau nhiều thì **không được kết luận** "mô hình A tốt hơn B" chỉ vì chênh 0.005.
- **Confusion matrix** chuẩn hoá theo hàng: đọc được recall từng lớp, lớp nào bị nhầm sang lớp nào.
- **Phân tích ngưỡng**: chỉ ra đánh đổi precision ↔ recall; chọn ngưỡng theo nhu cầu thực tế.

## 11. Câu hỏi thầy có thể hỏi (tự trả lời thử, đừng học vẹt)

1. Vì sao không dùng accuracy làm metric chính? Cho ví dụ.
2. Data leakage là gì? Pipeline của nhóm tránh nó ở đâu?
3. Vì sao phải chia stratified? Điều gì xảy ra nếu không?
4. Vì sao tuning bằng CV chứ không bằng tập test?
5. Random Forest khác Gradient Boosting ở điểm cốt lõi nào (bagging vs boosting, bias vs variance)?
6. `C` trong SVM / Logistic Regression điều khiển cái gì? `C` lớn thì sao?
7. Mô hình nào overfit nhất trong kết quả của nhóm? Dựa vào đâu để nói vậy?
8. Vì sao Decision Tree mặc định lại kém xa sau khi tuning (nếu thấy vậy)?
9. Ensemble có giúp không? Vì sao (không)?
10. Nếu bài toán thực tế bỏ sót ca dương rất tốn kém thì nhóm đổi gì (metric, ngưỡng, class_weight)?
11. Hạn chế lớn nhất của thực nghiệm này là gì?

## 12. Hạn chế (nên nêu trong báo cáo — nêu thật còn hơn giấu)

- Tuning chưa dùng nested CV nên điểm CV sau tuning hơi lạc quan.
- Mỗi cấu hình chỉ chạy với một seed; chưa lặp nhiều seed để đo độ dao động do khởi tạo.
- EDA thực hiện trên toàn bộ dữ liệu (chỉ mô tả, không dùng để chọn mô hình), nhưng vẫn nên nêu rõ.
- Chưa có kiến thức miền (domain) sâu để thiết kế đặc trưng mới; chỉ dùng đặc trưng gốc.
- Chưa thử SMOTE / cost-sensitive learning chi tiết.
