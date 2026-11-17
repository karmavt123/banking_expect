# SUMMARY (tự sinh - copy số liệu từ đây sang báo cáo)

- Metric chính: **average_precision**  |  CV: 5-fold stratified  |  seed: 42
- Bài toán: binary, lớp: ['no', 'yes'], n_train=32940, n_test=8236
- Mô hình chọn theo CV: **Ensemble: stacking**


## Benchmark CV (tham số mặc định)

| model | average_precision_mean | average_precision_std | train_average_precision | fit_time_s |
|---|---|---|---|---|
| Logistic Regression | 0.4466 | 0.0131 | 0.4521 | 0.3250 |
| Decision Tree | 0.1844 | 0.0058 | 0.9989 | 0.2871 |
| Random Forest | 0.3747 | 0.0089 | 0.9994 | 3.7853 |
| Gradient Boosting (HistGB) | 0.4664 | 0.0164 | 0.5280 | 0.7846 |
| MLP (Neural Net) | 0.4359 | 0.0180 | 0.4615 | 0.8874 |

## Tuning

| model | default_cv | tuned_cv | gain | best_params |
|---|---|---|---|---|
| Logistic Regression | 0.4466 | 0.4466 | 0.0000 | {"C": "14.528246637516036"} |
| Decision Tree | 0.1844 | 0.4196 | 0.2351 | {"ccp_alpha": "0.0005808361216819946", "criterion": "entropy", "max_depth": "10", "min_samples_leaf": "36", "min_samples_split": "41"} |
| Random Forest | 0.3747 | 0.4679 | 0.0932 | {"max_depth": "25", "max_features": "sqrt", "min_samples_leaf": "7", "n_estimators": "221"} |
| Gradient Boosting (HistGB) | 0.4664 | 0.4655 | -0.0009 | {"l2_regularization": "0.07362945281639216", "learning_rate": "0.18631003721334807", "max_depth": "5", "max_iter": "120", "max_leaf_nodes": "16", "min_samples_leaf": "48"} |
| MLP (Neural Net) | 0.4359 | 0.4556 | 0.0197 | {"alpha": "0.07852755494724252", "hidden_layer_sizes": "(128,)", "learning_rate_init": "0.000816845589476017"} |

## Feature selection

| model | variant | n_features_kept | n_features_total | average_precision_mean | average_precision_std |
|---|---|---|---|---|---|
| Random Forest | Không chọn (tất cả đặc trưng) | 62 | 62 | 0.4679 | 0.0155 |
| Random Forest | ANOVA-F top 10% | 7 | 62 | 0.4450 | 0.0186 |
| Random Forest | ANOVA-F top 25% | 16 | 62 | 0.4631 | 0.0153 |
| Random Forest | ANOVA-F top 50% | 31 | 62 | 0.4668 | 0.0160 |
| Random Forest | ANOVA-F top 75% | 46 | 62 | 0.4690 | 0.0174 |
| Random Forest | L1 (LinearSVC) | 42 | 62 | 0.4678 | 0.0144 |
| Random Forest | RandomForest importance (>= median) | 31 | 62 | 0.4656 | 0.0150 |

## Ensemble (CV)

| model | key | metric | cv_mean | cv_std | members |
|---|---|---|---|---|---|
| Ensemble: voting | voting | average_precision | 0.4685 | 0.0165 | Random Forest + Gradient Boosting (HistGB) + MLP (Neural Net) |
| Ensemble: stacking | stacking | average_precision | 0.4692 | 0.0166 | Random Forest + Gradient Boosting (HistGB) + MLP (Neural Net) |

## Kết quả trên tập test

| model | accuracy | balanced_accuracy | precision | recall | f1 | mcc | roc_auc | average_precision | chosen_by_cv |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression (default) | 0.8303 | 0.7496 | 0.3591 | 0.6455 | 0.4615 | 0.3928 | 0.8002 | 0.4463 | False |
| Logistic Regression (tuned) | 0.8303 | 0.7496 | 0.3591 | 0.6455 | 0.4615 | 0.3928 | 0.8003 | 0.4462 | False |
| Decision Tree (default) | 0.8475 | 0.6267 | 0.3295 | 0.3416 | 0.3354 | 0.2494 | 0.6269 | 0.1879 | False |
| Decision Tree (tuned) | 0.8630 | 0.7587 | 0.4264 | 0.6239 | 0.5066 | 0.4408 | 0.8087 | 0.4408 | False |
| Random Forest (default) | 0.8738 | 0.6989 | 0.4439 | 0.4731 | 0.4580 | 0.3870 | 0.7841 | 0.3872 | False |
| Random Forest (tuned) | 0.8549 | 0.7611 | 0.4082 | 0.6401 | 0.4985 | 0.4330 | 0.8096 | 0.4884 | False |
| Gradient Boosting (HistGB) (default) | 0.8485 | 0.7650 | 0.3961 | 0.6573 | 0.4943 | 0.4299 | 0.8152 | 0.4952 | False |
| Gradient Boosting (HistGB) (tuned) | 0.8445 | 0.7623 | 0.3877 | 0.6562 | 0.4874 | 0.4222 | 0.8138 | 0.4866 | False |
| MLP (Neural Net) (default) | 0.8983 | 0.6120 | 0.6250 | 0.2425 | 0.3494 | 0.3464 | 0.8031 | 0.4717 | False |
| MLP (Neural Net) (tuned) | 0.8998 | 0.5922 | 0.6988 | 0.1950 | 0.3050 | 0.3340 | 0.8012 | 0.4588 | False |
| Ensemble: voting | 0.8782 | 0.7573 | 0.4685 | 0.6013 | 0.5267 | 0.4627 | 0.8129 | 0.4822 | False |
| Ensemble: stacking | 0.8417 | 0.7612 | 0.3822 | 0.6573 | 0.4834 | 0.4179 | 0.8130 | 0.4831 | True |

## Khoảng tin cậy 95% (bootstrap trên test)

| model | metric | point | ci95_low | ci95_high |
|---|---|---|---|---|
| Logistic Regression (tuned) | average_precision | 0.4462 | 0.4126 | 0.4850 |
| Decision Tree (tuned) | average_precision | 0.4408 | 0.4065 | 0.4788 |
| Random Forest (tuned) | average_precision | 0.4884 | 0.4545 | 0.5252 |
| Gradient Boosting (HistGB) (tuned) | average_precision | 0.4866 | 0.4536 | 0.5237 |
| MLP (Neural Net) (tuned) | average_precision | 0.4588 | 0.4234 | 0.4952 |
| Ensemble: voting | average_precision | 0.4822 | 0.4472 | 0.5174 |
| Ensemble: stacking | average_precision | 0.4831 | 0.4484 | 0.5183 |

## Phân tích ngưỡng

| model | setting | threshold | precision | recall | f1 |
|---|---|---|---|---|---|
| Random Forest (tuned) | ngưỡng mặc định | 0.5000 | 0.4082 | 0.6401 | 0.4985 |
| Random Forest (tuned) | ngưỡng tối ưu F1 (chọn bằng CV trên train) | 0.6521 | 0.4760 | 0.5873 | 0.5258 |

## Top đặc trưng quan trọng (permutation)

| feature | importance | std |
|---|---|---|
| emp.var.rate | 0.0422 | 0.0067 |
| euribor3m | 0.0401 | 0.0083 |
| nr.employed | 0.0330 | 0.0074 |
| contact | 0.0254 | 0.0036 |
| pdays | 0.0243 | 0.0031 |
| poutcome | 0.0222 | 0.0031 |
| month | 0.0112 | 0.0021 |
| day_of_week | 0.0097 | 0.0031 |
| campaign | 0.0074 | 0.0010 |
| cons.price.idx | 0.0069 | 0.0020 |

## Hình đã sinh

- results/figures/benchmark_cv_boxplot.png
- results/figures/confusion_matrices.png
- results/figures/eda_class_distribution.png
- results/figures/eda_correlation.png
- results/figures/eda_numeric_hist.png
- results/figures/eda_target_rate_by_category.png
- results/figures/feature_selection.png
- results/figures/learning_curve.png
- results/figures/permutation_importance.png
- results/figures/roc_pr_curves.png
- results/figures/test_comparison.png
- results/figures/threshold_f1.png