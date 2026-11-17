# Tóm tắt EDA (tự sinh)

## Tổng quan

- **file**: bank-additional-full.csv
- **n_rows_raw**: 41188
- **n_cols_raw**: 21
- **n_duplicates_removed**: 12
- **n_missing_target_removed**: 0
- **constant_cols_dropped**: []
- **n_rows**: 41176
- **n_features**: 19
- **n_numeric**: 9
- **n_categorical**: 10
- **n_train**: 32940
- **n_test**: 8236
- **class_counts**: {'no': 36537, 'yes': 4639}
- **minority_ratio**: 0.11266271614532737
- **positive_label**: yes
- **n_missing_cells**: 0
- **loại bài toán**: binary (2 lớp: ['no', 'yes'])
- **số mẫu train / test**: 32940 / 8236

## Phân bố nhãn

| lớp | số mẫu | tỉ lệ |
|---|---|---|
| no | 36537 | 0.8873 |
| yes | 4639 | 0.1127 |

Tỉ lệ lớp lớn nhất / nhỏ nhất = 7.9

## Giá trị thiếu

Không có giá trị thiếu (sau khi quy đổi NA_VALUES).

## Đặc trưng số

| cột | mean | std | min | 50% | max | skew | % outlier(IQR) |
|---|---|---|---|---|---|---|---|
| age | 40.024 | 10.421 | 17.000 | 38.000 | 98.000 | 0.785 | 1.137 |
| campaign | 2.568 | 2.770 | 1.000 | 2.000 | 56.000 | 4.762 | 5.843 |
| pdays | 962.465 | 186.937 | 0.000 | 999.000 | 999.000 | -4.921 | 3.679 |
| previous | 0.173 | 0.495 | 0.000 | 0.000 | 7.000 | 3.831 | 13.661 |
| emp.var.rate | 0.082 | 1.571 | -3.400 | 1.100 | 1.400 | -0.724 | 0.000 |
| cons.price.idx | 93.576 | 0.579 | 92.201 | 93.749 | 94.767 | -0.231 | 0.000 |
| cons.conf.idx | -40.503 | 4.628 | -50.800 | -41.800 | -26.900 | 0.303 | 1.083 |
| euribor3m | 3.621 | 1.734 | 0.634 | 4.857 | 5.045 | -0.709 | 0.000 |
| nr.employed | 5167.035 | 72.251 | 4963.600 | 5191.000 | 5228.100 | -1.044 | 0.000 |

Cặp đặc trưng tương quan cao (|r| > 0.8): emp.var.rate~euribor3m (0.97); euribor3m~nr.employed (0.95); emp.var.rate~nr.employed (0.91)

## Đặc trưng phân loại

| cột | số giá trị | giá trị phổ biến nhất |
|---|---|---|
| job | 12 | admin. (25%) |
| marital | 4 | married (61%) |
| education | 8 | university.degree (30%) |
| default | 3 | no (79%) |
| housing | 3 | yes (52%) |
| loan | 3 | no (82%) |
| contact | 2 | cellular (63%) |
| month | 10 | may (33%) |
| day_of_week | 5 | thu (21%) |
| poutcome | 3 | nonexistent (86%) |